"""ShepherdAgent - autonomous candidate evaluator using LLM reasoning.

Each ShepherdAgent evaluates a single catalyst candidate by:
1. Using an LLM to reason about which tests to run
2. Dispatching tests to SimulationAgent (or Globus Compute fallback)
3. Tracking budget and accumulated results
4. Terminating when budget exhausted or confident enough

Architecture modes:
- Academy agents: Uses LLMAgent and SimulationAgent via Academy exchange
- Remote LLM: Direct HTTP connection to remote vLLM server (e.g., on Spark)
- Direct GC: Uses direct Globus Compute function calls
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, TYPE_CHECKING

from academy.agent import action

from skills.base_agent import TrackedAgent
from orchestration.cache import JsonlCache, make_cache_key, git_short_sha
from orchestration.test_registry import (
    get_test,
    check_prerequisites,
    get_runtime_tracker,
    AVAILABLE_TESTS,
    FAST_TEST_THRESHOLD_S,
)
from orchestration.shepherd_prompts import (
    get_system_prompt,
    SYSTEM_PROMPT,
    build_reasoning_prompt,
    build_final_assessment_prompt,
)
from orchestration.narrative import get_narrative

if TYPE_CHECKING:
    from skills.llm_agent import LLMAgent
    from skills.simulation_agent import SimulationAgent

logger = logging.getLogger(__name__)


class ShepherdAgent(TrackedAgent):
    """Autonomous agent that evaluates a catalyst candidate.

    Uses LLM reasoning to decide which characterization tests to run,
    then dispatches them to SimulationAgent or Globus Compute endpoints.

    Supports three modes (auto-detected based on constructor args):
    1. Academy agents mode: Uses LLMAgent and SimulationAgent via exchange
    2. Remote LLM mode: Direct HTTP to remote vLLM + GC for simulations
    3. Direct GC mode: Uses GC function calls for everything

    Inherits from TrackedAgent for automatic history tracking.
    """

    def __init__(
        self,
        config: dict[str, Any],
        gc_function_map: dict[str, str] | None = None,
        llm_agent: "LLMAgent | None" = None,
        llm_proxy: Any | None = None,
        sim_agent: "SimulationAgent | None" = None,
        sim_agents: dict[str, Any] | None = None,
        llm_url: str | None = None,
        llm_model: str | None = None,
        redis_host: str | None = None,
        redis_port: int = 6379,
        shepherd_id: int | str | None = None,
    ):
        """Initialize ShepherdAgent.

        Args:
            config: Shepherd configuration dict with sections:
                - llm: LLM endpoint configuration (for direct mode)
                - endpoints: GC endpoint IDs (for GC mode)
                - cache: Cache configuration
                - timeouts: Timeout settings
                - budget: Default budget settings
            gc_function_map: Mapping of test name -> Globus Compute function ID.
            llm_agent: LLMAgent instance for LLM calls (Academy mode).
                If provided, uses Academy agent-based communication.
            sim_agent: SimulationAgent instance for simulations (Academy mode).
                If provided, uses Academy agent-based communication.
                DEPRECATED: Use sim_agents dict instead.
            sim_agents: Dict mapping simulation code names to their agents.
                Example: {"mace": mace_agent, "cantera": cantera_agent, ...}
                If provided, dispatches to individual agents per simulation code.
            llm_url: Direct URL to vLLM server (e.g., "http://spark:8000/v1").
                If provided, bypasses Academy and calls vLLM directly via HTTP.
                Useful when vLLM is running on Spark and accessible via tunnel.
            llm_model: Model name for direct vLLM connection.
                Defaults to "meta-llama/Meta-Llama-3-8B-Instruct".
            redis_host: Redis hostname for distributed narrative logging.
            redis_port: Redis port (default 6379).
            shepherd_id: Optional ID for this shepherd (for logging).
                Auto-generated if not provided.
        """
        super().__init__(max_history=100)
        self._config = config
        self._gc_function_map = gc_function_map or {}
        self._redis_host = redis_host
        self._redis_port = redis_port

        # Shepherd identification for logging
        if shepherd_id is not None:
            self._shepherd_id = str(shepherd_id)
        else:
            import random
            self._shepherd_id = f"S{random.randint(1, 999):03d}"
        self._current_candidate: str | None = None  # For logging context

        # Academy agent mode
        self._llm_agent = llm_agent
        self._llm_proxy = llm_proxy  # LLMProxyAgent for tracked LLM calls
        self._sim_agent = sim_agent  # Legacy single agent
        self._sim_agents = sim_agents or {}  # New: individual agents per code
        # Only use Academy LLM agent if explicitly provided
        self._use_llm_agent = llm_agent is not None
        # Use sim agents if provided
        self._use_sim_agents = sim_agent is not None or bool(sim_agents)

        # Remote LLM mode (direct HTTP to vLLM)
        self._llm_url = llm_url
        self._llm_model = llm_model or "meta-llama/Meta-Llama-3-8B-Instruct"
        self._remote_llm_client: Any = None  # AsyncOpenAI for remote mode

        # Direct GC mode (direct HTTP/GC calls)
        self._llm_client: Any = None  # LLMClient for config-based mode
        self._gc_cheap: Any = None  # GlobusComputeAdapter
        self._gc_gpu: Any = None

        # Common
        self._cache: JsonlCache | None = None
        self._version: str = "dev"
        self._system_prompt: str = ""

    @property
    def _tag(self) -> str:
        """Log tag with shepherd ID and current candidate."""
        if self._current_candidate:
            return f"[{self._shepherd_id}|{self._current_candidate}]"
        return f"[{self._shepherd_id}]"

    async def agent_on_startup(self) -> None:
        """Initialize resources after agent starts."""
        # Get system prompt WITHOUT capabilities (they confuse the LLM about test names)
        # The valid test names are already in the reasoning prompt
        self._system_prompt = get_system_prompt(include_capabilities=False)

        # Initialize LLM client (four options)
        if self._use_llm_agent:
            # Use Academy LLM agent (legacy)
            logger.info("ShepherdAgent: using Academy LLM agent")
            status = await self._llm_agent.get_status({})
            logger.info("LLMAgent connected: model=%s", status.get("model"))

        elif self._llm_proxy:
            # Use LLM Proxy agent (preferred - tracks usage)
            logger.info("ShepherdAgent: using LLM Proxy agent")
            status = await self._llm_proxy.get_status({})
            logger.info("LLMProxy connected: model=%s", status.get("model"))

        elif self._llm_url:
            # Use remote LLM (direct HTTP to vLLM)
            logger.info("ShepherdAgent: using remote vLLM at %s", self._llm_url)

            from openai import AsyncOpenAI
            self._remote_llm_client = AsyncOpenAI(
                base_url=self._llm_url,
                api_key="not-needed",  # vLLM doesn't require auth
            )

            # Test connection and auto-detect model
            try:
                models = await self._remote_llm_client.models.list()
                available = [m.id for m in models.data]
                logger.info("Remote vLLM connected. Available models: %s", available)

                if self._llm_model not in available:
                    if available:
                        old_model = self._llm_model
                        self._llm_model = available[0]
                        logger.warning(
                            "Model %s not available. Auto-selected: %s",
                            old_model, self._llm_model,
                        )
                    else:
                        raise RuntimeError("No models available on vLLM server")
                else:
                    logger.info("Using model: %s", self._llm_model)
            except Exception as e:
                logger.error("Failed to connect to remote vLLM: %s", e)
                raise

        else:
            # Use config-based LLM client
            logger.info("ShepherdAgent: using config-based LLM")
            from orchestration.llm_client import create_llm_client_from_config
            self._llm_client = create_llm_client_from_config(self._config)
            logger.info(
                "ShepherdAgent LLM initialized: model=%s base_url=%s",
                self._llm_client.model,
                self._llm_client.base_url,
            )

        # Initialize simulation backend (three options)
        if self._use_sim_agents:
            # Use Academy simulation agents
            logger.info("ShepherdAgent: using %d Academy simulation agents", len(self._sim_agents))
            if self._sim_agent:
                status = await self._sim_agent.get_status({})
                logger.info("SimulationAgent connected: codes=%s", status.get("available_codes"))
        else:
            # Initialize GC adapters for simulations
            self._init_gc_adapters()

        # Cache (common to all modes)
        cache_config = self._config.get("cache", {})
        if cache_config.get("enabled", True):
            cache_path = cache_config.get("path", "data/shepherd_cache.jsonl")
            self._cache = JsonlCache.load(cache_path)
            logger.info("Cache loaded from %s (%d entries)", cache_path, len(self._cache.index))

        # Version for cache keys
        self._version = git_short_sha()

        # Runtime tracker for adaptive test classification
        self._runtime_tracker = get_runtime_tracker(
            redis_host=self._redis_host,
            redis_port=self._redis_port,
        )
        logger.info("RuntimeTracker initialized for adaptive test classification")

    def _init_gc_adapters(self) -> None:
        """Initialize Globus Compute adapters for simulation dispatch."""
        from hpc.globus_compute import GlobusComputeAdapter
        endpoints = self._config.get("endpoints", {})
        if endpoints.get("cheap"):
            self._gc_cheap = GlobusComputeAdapter(endpoints["cheap"])
            logger.info("GC cheap endpoint: %s", endpoints["cheap"])
        if endpoints.get("gpu"):
            self._gc_gpu = GlobusComputeAdapter(endpoints["gpu"])
            logger.info("GC gpu endpoint: %s", endpoints["gpu"])

    @action
    async def evaluate(self, req: dict) -> dict:
        """Evaluate a catalyst candidate using two-phase approach.

        Phase 1: Run all cheap tests in parallel (fast, no LLM needed)
        Phase 2: LLM decides which expensive tests to run based on cheap results

        Args:
            req: Request dict with:
                - candidate: CatalystSpec dict to evaluate
                - budget: Max compute cost (optional, uses config default)
                - goal: Optional optimization goal string

        Returns:
            Evaluation result dict with:
                - candidate: The input candidate
                - results: List of test results
                - total_cost: Total compute cost spent
                - final_assessment: LLM's final assessment
                - confidence: Confidence in assessment (0-1)
                - history: Full reasoning history
        """
        candidate = req["candidate"]
        budget_config = self._config.get("budget", {})
        budget_total = req.get("budget", budget_config.get("default", 100.0))
        goal = req.get("goal", "CO2-to-methanol conversion")

        # Set current candidate for logging context
        self._current_candidate = _candidate_summary(candidate)

        # Track this evaluation
        action_tracker = self.track_action("evaluate", req)
        action_tracker.__enter__()

        logger.info(
            "[%s|%s] Starting evaluation (budget=%.1f)",
            self._shepherd_id,
            self._current_candidate,
            budget_total,
        )

        narrative = get_narrative(redis_host=self._redis_host, redis_port=self._redis_port)
        narrative.shepherd_start(candidate, budget_total)

        results: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        budget_spent = 0.0

        # ========== PHASE 1: Run all fast tests in parallel ==========
        # Use RuntimeTracker to classify tests based on observed/expected runtimes
        fast_tests = self._runtime_tracker.get_fast_tests(threshold_s=FAST_TEST_THRESHOLD_S)
        affordable_fast = [t for t in fast_tests if t.cost <= budget_total]

        logger.info("%s Phase 1: %d fast tests", self._tag, len(affordable_fast))
        narrative._write(f"   Phase 1: Running {len(affordable_fast)} fast tests in parallel...")

        if affordable_fast:
            # Create tasks for all fast tests
            async def run_fast_test(test_spec):
                """Run a single fast test and return result dict."""
                t0 = time.time()
                try:
                    test_result = await self._run_test(test_spec.name, candidate)
                    elapsed = time.time() - t0
                    return {
                        "test": test_spec.name,
                        "result": test_result,
                        "cost": test_spec.cost,
                        "elapsed": elapsed,
                        "ok": True,
                    }
                except Exception as e:
                    elapsed = time.time() - t0
                    logger.error("%s Fast test %s failed: %s", self._tag, test_spec.name, e)
                    return {
                        "test": test_spec.name,
                        "result": {"error": str(e), "ok": False},
                        "cost": test_spec.cost,
                        "elapsed": elapsed,
                        "ok": False,
                    }

            # Run all fast tests concurrently
            fast_tasks = [run_fast_test(t) for t in affordable_fast]
            fast_results = await asyncio.gather(*fast_tasks, return_exceptions=True)

            # Process results and record runtimes
            for r in fast_results:
                if isinstance(r, Exception):
                    logger.error("Fast test task exception: %s", r)
                    continue

                results.append({
                    "test": r["test"],
                    "result": r["result"],
                    "cost": r["cost"],
                })
                budget_spent += r["cost"]

                # Record runtime for adaptive classification (only for successful tests)
                # Failed tests may complete quickly due to errors, skewing classification
                if "elapsed" in r and r.get("ok", False):
                    self._runtime_tracker.record(r["test"], r["elapsed"])

                # Log to narrative
                narrative.shepherd_test_result(r["test"], r["result"], r.get("elapsed"))

            history.append({
                "phase": 1,
                "tests_run": [r["test"] for r in fast_results if not isinstance(r, Exception)],
                "budget_spent": budget_spent,
            })

            logger.info("%s Phase 1 done: %d tests, cost=%.1f", self._tag, len(fast_results), budget_spent)

        # ========== PHASE 2: LLM decides on slow tests ==========
        slow_tests = self._runtime_tracker.get_slow_tests(threshold_s=FAST_TEST_THRESHOLD_S)
        affordable_slow = [
            t for t in slow_tests
            if t.cost <= (budget_total - budget_spent)
        ]

        # Log classification for debugging
        if slow_tests:
            logger.info("%s Slow tests: %s", self._tag, [t.name for t in slow_tests])
        else:
            fast_tests = self._runtime_tracker.get_fast_tests(threshold_s=FAST_TEST_THRESHOLD_S)
            logger.info("%s All tests classified as fast: %s", self._tag, [t.name for t in fast_tests])

        if affordable_slow and budget_spent < budget_total:
            # Get names of slow tests to show in prompt (exclude fast tests)
            slow_test_names = {t.name for t in slow_tests}

            logger.info("%s Phase 2: %d slow tests available", self._tag, len(affordable_slow))
            narrative._write(f"   Phase 2: Consulting LLM about {len(affordable_slow)} slow tests...")

            # Build prompt for expensive test decision - ONLY show slow tests
            prompt = build_reasoning_prompt(
                candidate=candidate,
                results=results,
                budget_total=budget_total,
                budget_spent=budget_spent,
                only_tests=slow_test_names,  # Filter to slow tests only
            )

            iteration = 0
            max_iterations = 10  # Safety limit for expensive tests
            consecutive_invalid = 0
            max_consecutive_invalid = 3

            while budget_spent < budget_total and iteration < max_iterations:
                iteration += 1

                if consecutive_invalid >= max_consecutive_invalid:
                    logger.warning(
                        "Stopping: %d consecutive invalid LLM requests",
                        consecutive_invalid,
                    )
                    break

                # Get LLM decision
                narrative.shepherd_thinking(f"phase 2 iteration {iteration}, budget spent {budget_spent:.1f}/{budget_total}")
                try:
                    decision = await self._reason_json(prompt)
                except Exception as e:
                    logger.error("LLM reasoning failed: %s", e)
                    history.append({"phase": 2, "iteration": iteration, "error": str(e)})
                    break

                history.append({
                    "phase": 2,
                    "iteration": iteration,
                    "decision": decision,
                    "budget_remaining": budget_total - budget_spent,
                })

                action_type = decision.get("action", "stop")

                if action_type == "stop":
                    logger.info(
                        "[%s|%s] LLM stopped: confidence=%.2f",
                        self._shepherd_id,
                        self._current_candidate,
                        decision.get("confidence", 0),
                    )
                    break

                if action_type == "test":
                    test_name = decision.get("test")
                    if not test_name:
                        logger.warning("LLM returned test action without test name")
                        consecutive_invalid += 1
                        continue

                    # Validate test exists
                    try:
                        test_spec = get_test(test_name)
                    except KeyError:
                        logger.warning("Unknown test requested: %s", test_name)
                        history[-1]["error"] = f"Unknown test: {test_name}"
                        consecutive_invalid += 1
                        continue

                    # Only allow slow tests in phase 2 (fast tests already run)
                    estimated_runtime = self._runtime_tracker.get_estimated_runtime(test_name)
                    if estimated_runtime < FAST_TEST_THRESHOLD_S:
                        logger.warning("%s LLM suggested fast test %s (skipped)", self._tag, test_name)
                        history[-1]["error"] = f"Fast test '{test_name}' not allowed in phase 2"
                        consecutive_invalid += 1
                        continue

                    # Check if already run
                    completed_tests = {r["test"] for r in results if r.get("result", {}).get("ok", True)}
                    if test_name in completed_tests:
                        logger.warning("Test %s already completed", test_name)
                        history[-1]["error"] = f"Test '{test_name}' already completed"
                        consecutive_invalid += 1
                        continue

                    # Check budget
                    if test_spec.cost > (budget_total - budget_spent):
                        logger.warning(
                            "Test %s costs %.2f but only %.2f budget remaining",
                            test_name,
                            test_spec.cost,
                            budget_total - budget_spent,
                        )
                        history[-1]["error"] = f"Insufficient budget for {test_name}"
                        continue

                    # Run the expensive test
                    logger.info(
                        "[%s|%s] Running: %s (cost=%.2f)",
                        self._shepherd_id,
                        self._current_candidate,
                        test_name,
                        test_spec.cost,
                    )
                    reasoning = decision.get("reasoning", "")
                    narrative.shepherd_decision(test_name, reasoning, test_spec.cost)
                    narrative.shepherd_running_test(test_name, None)

                    t0 = time.time()
                    try:
                        test_result = await self._run_test(test_name, candidate)
                        elapsed = time.time() - t0
                        results.append({
                            "test": test_name,
                            "result": test_result,
                            "cost": test_spec.cost,
                        })
                        # Record runtime for adaptive classification
                        self._runtime_tracker.record(test_name, elapsed)
                        narrative.shepherd_test_result(test_name, test_result, elapsed)
                        budget_spent += test_spec.cost
                        history[-1]["test_result"] = test_result
                        consecutive_invalid = 0
                    except Exception as e:
                        elapsed = time.time() - t0
                        logger.error("Test %s failed: %s", test_name, e)
                        results.append({
                            "test": test_name,
                            "result": {"error": str(e), "ok": False},
                            "cost": test_spec.cost,
                        })
                        # Don't record runtime for failed tests - they may fail quickly
                        # and skew the classification (making slow tests look fast)
                        budget_spent += test_spec.cost
                        history[-1]["error"] = str(e)

                    # Rebuild prompt with new results for next iteration
                    prompt = build_reasoning_prompt(
                        candidate=candidate,
                        results=results,
                        budget_total=budget_total,
                        budget_spent=budget_spent,
                        only_tests=slow_test_names,  # Keep filter to slow tests
                    )
        else:
            # Explain why we're skipping phase 2
            if budget_spent >= budget_total:
                logger.info("%s Phase 2: Skipped - budget exhausted", self._tag)
            elif not slow_tests:
                logger.info("%s Phase 2: Skipped - no slow tests defined", self._tag)
            elif not affordable_slow:
                logger.info("%s Phase 2: Skipped - slow tests too expensive (need %.1f, have %.1f)",
                    self._tag, min(t.cost for t in slow_tests), budget_total - budget_spent)

        # Generate final assessment
        final_assessment = await self._generate_final_assessment(
            candidate, results, budget_spent
        )

        # Log completion to narrative
        narrative.shepherd_done(
            candidate,
            final_assessment.get("viability_score", 0),
            final_assessment.get("recommendation", "unknown"),
            len(results),
            budget_spent,
        )

        result = {
            "candidate": candidate,
            "results": results,
            "total_cost": budget_spent,
            "final_assessment": final_assessment,
            "confidence": final_assessment.get("viability_score", 0) / 100.0,
            "history": history,
            "phases": 2,
        }

        # Log completion with score
        score = final_assessment.get("viability_score", 0)
        rec = final_assessment.get("recommendation", "?")
        logger.info("%s Done: score=%d %s", self._tag, score, rec)

        # Clear current candidate
        self._current_candidate = None

        # Complete action tracking
        action_tracker.set_result(result)
        action_tracker.__exit__(None, None, None)

        return result

    async def _reason_json(self, prompt: str) -> dict[str, Any]:
        """Get JSON response from LLM using appropriate mode.

        Args:
            prompt: User prompt to send

        Returns:
            Parsed JSON response dict
        """
        if self._llm_agent:
            # Academy agent mode (legacy)
            response = await self._llm_agent.chat_completion_json({
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            })

            if not response.get("ok"):
                raise RuntimeError(f"LLMAgent error: {response.get('error')}")

            if "parsed" in response:
                return response["parsed"]
            else:
                return json.loads(response.get("content", "{}"))

        elif self._llm_proxy:
            # LLM Proxy mode - uses LLMProxyAgent for tracking
            response = await self._llm_proxy.chat_completion_json({
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            })

            if not response.get("ok"):
                raise RuntimeError(f"LLMProxy error: {response.get('error')}")

            if "parsed" in response:
                return response["parsed"]
            else:
                return json.loads(response.get("content", "{}"))

        elif self._remote_llm_client:
            # Remote LLM mode - direct HTTP to vLLM
            response = await self._remote_llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )

            content = response.choices[0].message.content
            return self._parse_json_response(content)

        else:
            # Direct GC mode - use config-based LLM client
            return await self._llm_client.reason_json(prompt, self._system_prompt)

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks.

        Args:
            content: Raw LLM response string

        Returns:
            Parsed JSON dict
        """
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {content[:200]}...")

    async def _run_test(
        self,
        test_name: str,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a characterization test, using cache if available.

        Args:
            test_name: Name of test to run
            candidate: Candidate specification

        Returns:
            Test result dict
        """
        # Check cache
        if self._cache:
            cache_key = make_cache_key(candidate, test_name, self._version)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit for %s", test_name)
                return cached

        # Get test spec
        test_spec = get_test(test_name)

        # Dispatch based on mode
        if self._sim_agents:
            # Academy mode - dispatch to individual simulation agents
            result = await self._dispatch_to_sim_agents(test_name, candidate, test_spec)
        elif self._sim_agent:
            # Legacy Academy agent mode - use single SimulationAgent
            result = await self._dispatch_sim_agent(test_name, candidate, test_spec)
        elif test_spec.endpoint == "gpu" and self._gc_gpu:
            # Legacy GC mode - GPU endpoint
            result = await self._dispatch_gc(self._gc_gpu, test_name, candidate, test_spec.timeout)
        elif self._gc_cheap:
            # Legacy GC mode - cheap endpoint
            result = await self._dispatch_gc(self._gc_cheap, test_name, candidate, test_spec.timeout)
        else:
            # Local fallback for development
            result = await self._run_local(test_name, candidate)

        # Cache result
        if self._cache:
            cache_key = make_cache_key(candidate, test_name, self._version)
            self._cache.set(cache_key, result)

        return result

    async def _dispatch_sim_agent(
        self,
        test_name: str,
        candidate: dict[str, Any],
        test_spec: Any,
    ) -> dict[str, Any]:
        """Dispatch test to SimulationAgent.

        Args:
            test_name: Name of test
            candidate: Candidate specification
            test_spec: Test specification

        Returns:
            Test result dict
        """
        # Map test name to simulation code and task
        code_mapping = {
            "fast_surrogate": ("surrogate", "screening"),
            "ml_screening": ("mace", "screening"),
            "ml_relaxation": ("mace", "relaxation"),
            "microkinetic_lite": ("surrogate", "microkinetics"),
            "dft_adsorption": ("quantum_espresso", "adsorption"),
            "stability_analysis": ("stability", "analysis"),
            "cantera_reactor": ("cantera", "reactor"),
            "openmm_relaxation": ("openmm", "relaxation"),
        }

        code, task = code_mapping.get(test_name, ("surrogate", "screening"))

        logger.info("Dispatching to SimulationAgent: code=%s task=%s", code, task)

        result = await self._sim_agent.run_simulation({
            "code": code,
            "task": task,
            "candidate": candidate,
            "options": {},
        })

        if not result.get("ok"):
            raise RuntimeError(f"SimulationAgent error: {result.get('error')}")

        return result

    async def _dispatch_to_sim_agents(
        self,
        test_name: str,
        candidate: dict[str, Any],
        test_spec: Any,
    ) -> dict[str, Any]:
        """Dispatch test to individual simulation agents.

        Args:
            test_name: Name of test
            candidate: Candidate specification
            test_spec: Test specification

        Returns:
            Test result dict
        """
        # Map test name to agent name and action
        dispatch_mapping = {
            "fast_surrogate": ("surrogate", "screening"),
            "ml_screening": ("mace", "screening"),
            "ml_relaxation": ("mace", "relaxation"),
            "chgnet_screening": ("chgnet", "screening"),
            "microkinetic_lite": ("surrogate", "microkinetic"),
            "dft_adsorption": ("qe", "adsorption"),
            "stability_analysis": ("stability", "analyze"),
            "cantera_reactor": ("cantera", "reactor"),
            "cantera_sensitivity": ("cantera", "sensitivity"),
            "openmm_relaxation": ("openmm", "relaxation"),
        }

        agent_name, action_name = dispatch_mapping.get(test_name, ("surrogate", "screening"))

        # Get the agent
        agent = self._sim_agents.get(agent_name)
        if not agent:
            # Try fallback to surrogate
            agent = self._sim_agents.get("surrogate")
            if not agent:
                raise RuntimeError(f"No agent available for {agent_name} or surrogate fallback")
            logger.warning("Agent %s not available, falling back to surrogate", agent_name)
            action_name = "screening"

        logger.info("%s -> %s.%s", self._tag, agent_name, action_name)

        # Get the action method
        action_method = getattr(agent, action_name, None)
        if not action_method:
            raise RuntimeError(f"Agent {agent_name} has no action '{action_name}'")

        # Call the action with timing
        t0 = time.time()
        result = await action_method({
            "candidate": candidate,
        })
        elapsed = time.time() - t0

        # Add timing to result
        result["elapsed_s"] = round(elapsed, 2)
        logger.info("%s <- %s.%s (%.1fs)", self._tag, agent_name, action_name, elapsed)

        if not result.get("ok"):
            raise RuntimeError(f"{agent_name}.{action_name} error: {result.get('error')}")

        return result

    async def _dispatch_gc(
        self,
        gc: Any,  # GlobusComputeAdapter
        test_name: str,
        candidate: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """Dispatch test to Globus Compute and wait for result.

        Args:
            gc: GlobusComputeAdapter for the endpoint
            test_name: Name of test
            candidate: Candidate specification
            timeout: Max wait time in seconds

        Returns:
            Test result dict
        """
        function_id = self._gc_function_map.get(test_name)
        if not function_id:
            raise ValueError(f"No function_id configured for test: {test_name}")

        # Submit task
        payload = {"candidate": candidate, "test": test_name}
        task = gc.submit(function_id=function_id, payload=payload)
        logger.info("Submitted GC task: %s for test %s", task.task_id, test_name)

        # Poll for result
        poll_interval = self._config.get("timeouts", {}).get("test_poll_interval", 1.0)
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Test {test_name} timed out after {timeout}s")

            result = gc.try_result(task.task_id)
            status = result.get("status", "UNKNOWN")

            if status == "SUCCEEDED":
                return result.get("result", {})
            elif status == "FAILED":
                raise RuntimeError(f"Test {test_name} failed: {result.get('error')}")
            else:
                # Still pending
                await asyncio.sleep(poll_interval)

    async def _run_local(
        self,
        test_name: str,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Run test locally (development fallback).

        Args:
            test_name: Name of test
            candidate: Candidate specification

        Returns:
            Mock test result dict
        """
        logger.warning("Running test %s locally (no GC endpoint configured)", test_name)

        # Return mock results for development
        # In production, all tests should go through Globus Compute
        await asyncio.sleep(0.1)  # Simulate some work

        mock_results = {
            "fast_surrogate": {
                "co2_conversion": 0.35,
                "methanol_selectivity": 0.72,
                "methanol_sty": 0.25,
                "uncertainty": 0.15,
            },
            "microkinetic_lite": {
                "RLS": "CO2_adsorption",
                "temp_sensitivity": 0.8,
                "pressure_sensitivity": 0.3,
                "uncertainty_reduction": 0.05,
            },
            "dft_adsorption": {
                "E_ads_CO2": -0.45,
                "E_ads_H": -0.32,
                "uncertainty_reduction": 0.08,
            },
            "stability_analysis": {
                "stability_score": 0.82,
                "degradation_risk": "low",
            },
        }

        return mock_results.get(test_name, {"status": "completed", "test": test_name})

    async def _generate_final_assessment(
        self,
        candidate: dict[str, Any],
        results: list[dict[str, Any]],
        total_cost: float,
    ) -> dict[str, Any]:
        """Generate final assessment using LLM.

        Args:
            candidate: Candidate specification
            results: List of test results
            total_cost: Total compute cost spent

        Returns:
            Assessment dict with viability_score, strengths, concerns, recommendation
        """
        if not results:
            # No tests were run
            return {
                "viability_score": 0,
                "strengths": [],
                "concerns": ["No tests were run"],
                "recommendation": "REJECT",
                "summary": "Unable to assess candidate - no tests completed.",
            }

        prompt = build_final_assessment_prompt(
            candidate=candidate,
            results=results,
            total_cost=total_cost,
        )

        try:
            assessment = await self._reason_json(prompt)
            return assessment
        except Exception as e:
            error_msg = str(e)
            logger.error("%s Final assessment failed: %s", self._tag, error_msg)

            # Return error with score=0 so it's clearly a failure, not a mediocre result
            return {
                "viability_score": 0,
                "strengths": [],
                "concerns": [f"ASSESSMENT FAILED: {error_msg}"],
                "recommendation": "ERROR",
                "summary": f"Assessment failed: {error_msg}",
                "error": error_msg,
            }

    @action
    async def get_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get agent status including history statistics."""
        stats = self._get_statistics()
        return {
            "ok": True,
            "ready": True,
            "mode": "remote_llm" if self._remote_llm_client else "config_llm" if self._llm_client else "academy_llm",
            "llm_model": self._llm_model,
            "num_sim_agents": len(self._sim_agents),
            "cache_entries": len(self._cache.index) if self._cache else 0,
            "total_actions": stats["total_actions"],
            "total_time_s": stats["total_time_s"],
            "action_counts": stats["action_counts"],
        }

    @action
    async def get_runtime_stats(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get runtime classification statistics for tests.

        Returns summary of fast vs slow test classification based on
        observed and expected runtimes.
        """
        threshold = request.get("threshold_s", FAST_TEST_THRESHOLD_S)
        summary = self._runtime_tracker.get_classification_summary(threshold_s=threshold)
        return {
            "ok": True,
            "threshold_s": threshold,
            "fast_tests": summary["fast"],
            "slow_tests": summary["slow"],
            "n_fast": len(summary["fast"]),
            "n_slow": len(summary["slow"]),
        }


def _candidate_summary(candidate: dict[str, Any]) -> str:
    """Generate short summary string for candidate."""
    support = candidate.get("support", "unknown")
    metals = candidate.get("metals", [])
    metal_str = "+".join(
        f"{m.get('element', '?')}{m.get('wt_pct', '?')}%"
        for m in metals[:3]
    )
    return f"{metal_str}/{support}"

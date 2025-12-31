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

from academy.agent import Agent, action

from orchestration.cache import JsonlCache, make_cache_key, git_short_sha
from orchestration.test_registry import (
    get_test,
    check_prerequisites,
    AVAILABLE_TESTS,
)
from orchestration.shepherd_prompts import (
    get_system_prompt,
    SYSTEM_PROMPT,
    build_reasoning_prompt,
    build_final_assessment_prompt,
)

if TYPE_CHECKING:
    from skills.llm_agent import LLMAgent
    from skills.simulation_agent import SimulationAgent

logger = logging.getLogger(__name__)


class ShepherdAgent(Agent):
    """Autonomous agent that evaluates a catalyst candidate.

    Uses LLM reasoning to decide which characterization tests to run,
    then dispatches them to SimulationAgent or Globus Compute endpoints.

    Supports three modes (auto-detected based on constructor args):
    1. Academy agents mode: Uses LLMAgent and SimulationAgent via exchange
    2. Remote LLM mode: Direct HTTP to remote vLLM + GC for simulations
    3. Direct GC mode: Uses GC function calls for everything
    """

    def __init__(
        self,
        config: dict[str, Any],
        gc_function_map: dict[str, str] | None = None,
        llm_agent: "LLMAgent | None" = None,
        sim_agent: "SimulationAgent | None" = None,
        sim_agents: dict[str, Any] | None = None,
        llm_url: str | None = None,
        llm_model: str | None = None,
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
        """
        super().__init__()
        self._config = config
        self._gc_function_map = gc_function_map or {}

        # Academy agent mode
        self._llm_agent = llm_agent
        self._sim_agent = sim_agent  # Legacy single agent
        self._sim_agents = sim_agents or {}  # New: individual agents per code
        self._use_agents = (
            llm_agent is not None or
            sim_agent is not None or
            bool(sim_agents)
        )

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

    async def agent_on_startup(self) -> None:
        """Initialize resources after agent starts."""
        # Get system prompt with capabilities
        self._system_prompt = get_system_prompt(include_capabilities=True)

        if self._use_agents:
            # Mode 1: Academy agents
            logger.info("ShepherdAgent starting in Academy agents mode")
            if self._llm_agent:
                status = await self._llm_agent.get_status({})
                logger.info("LLMAgent connected: model=%s", status.get("model"))
            if self._sim_agent:
                status = await self._sim_agent.get_status({})
                logger.info("SimulationAgent connected: codes=%s", status.get("available_codes"))

        elif self._llm_url:
            # Mode 2: Remote LLM (direct HTTP to vLLM on Spark)
            logger.info("ShepherdAgent starting in remote LLM mode")
            logger.info("Connecting to vLLM at: %s", self._llm_url)

            from openai import AsyncOpenAI
            self._remote_llm_client = AsyncOpenAI(
                base_url=self._llm_url,
                api_key="not-needed",  # vLLM doesn't require auth
            )

            # Test connection
            try:
                models = await self._remote_llm_client.models.list()
                available = [m.id for m in models.data]
                logger.info("Remote vLLM connected. Available models: %s", available)
                if self._llm_model not in available:
                    logger.warning(
                        "Requested model %s not in available models %s",
                        self._llm_model,
                        available,
                    )
            except Exception as e:
                logger.error("Failed to connect to remote vLLM: %s", e)
                raise

            # Still need GC for simulations
            self._init_gc_adapters()

        else:
            # Mode 3: Direct GC mode (config-based LLM + GC)
            logger.info("ShepherdAgent starting in direct GC mode")
            from orchestration.llm_client import create_llm_client_from_config
            self._llm_client = create_llm_client_from_config(self._config)
            logger.info(
                "ShepherdAgent LLM initialized: model=%s base_url=%s",
                self._llm_client.model,
                self._llm_client.base_url,
            )
            self._init_gc_adapters()

        # Cache (common to all modes)
        cache_config = self._config.get("cache", {})
        if cache_config.get("enabled", True):
            cache_path = cache_config.get("path", "data/shepherd_cache.jsonl")
            self._cache = JsonlCache.load(cache_path)
            logger.info("Cache loaded from %s (%d entries)", cache_path, len(self._cache.index))

        # Version for cache keys
        self._version = git_short_sha()

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
        """Evaluate a catalyst candidate using LLM-guided test selection.

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

        logger.info(
            "ShepherdAgent evaluating candidate: %s (budget=%.2f)",
            _candidate_summary(candidate),
            budget_total,
        )

        results: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        budget_spent = 0.0
        iteration = 0
        max_iterations = 20  # Safety limit

        while budget_spent < budget_total and iteration < max_iterations:
            iteration += 1

            # Build reasoning prompt
            prompt = build_reasoning_prompt(
                candidate=candidate,
                results=results,
                budget_total=budget_total,
                budget_spent=budget_spent,
            )

            # Get LLM decision
            try:
                decision = await self._reason_json(prompt)
            except Exception as e:
                logger.error("LLM reasoning failed: %s", e)
                history.append({"iteration": iteration, "error": str(e)})
                break

            history.append({
                "iteration": iteration,
                "decision": decision,
                "budget_remaining": budget_total - budget_spent,
            })

            action_type = decision.get("action", "stop")

            if action_type == "stop":
                logger.info(
                    "Shepherd stopping: confidence=%.2f reason=%s",
                    decision.get("confidence", 0),
                    decision.get("reasoning", "")[:100],
                )
                break

            if action_type == "test":
                test_name = decision.get("test")
                if not test_name:
                    logger.warning("LLM returned test action without test name")
                    continue

                # Validate test exists
                try:
                    test_spec = get_test(test_name)
                except KeyError as e:
                    logger.warning("Unknown test requested: %s", test_name)
                    history[-1]["error"] = str(e)
                    continue

                # Check prerequisites
                completed_tests = {r["test"] for r in results}
                satisfied, missing = check_prerequisites(test_name, completed_tests)
                if not satisfied:
                    logger.warning(
                        "Prerequisites not met for %s: missing %s",
                        test_name,
                        missing,
                    )
                    history[-1]["error"] = f"Prerequisites not met: {missing}"
                    continue

                # Check budget
                if test_spec.cost > (budget_total - budget_spent):
                    logger.warning(
                        "Test %s costs %.2f but only %.2f budget remaining",
                        test_name,
                        test_spec.cost,
                        budget_total - budget_spent,
                    )
                    history[-1]["error"] = "Insufficient budget"
                    continue

                # Run the test
                logger.info("Running test: %s (cost=%.2f)", test_name, test_spec.cost)
                try:
                    test_result = await self._run_test(test_name, candidate)
                    results.append({
                        "test": test_name,
                        "result": test_result,
                        "cost": test_spec.cost,
                    })
                    budget_spent += test_spec.cost
                    history[-1]["test_result"] = test_result
                except Exception as e:
                    logger.error("Test %s failed: %s", test_name, e)
                    history[-1]["error"] = str(e)

        # Generate final assessment
        final_assessment = await self._generate_final_assessment(
            candidate, results, budget_spent
        )

        return {
            "candidate": candidate,
            "results": results,
            "total_cost": budget_spent,
            "final_assessment": final_assessment,
            "confidence": final_assessment.get("viability_score", 0) / 100.0,
            "history": history,
            "iterations": iteration,
        }

    async def _reason_json(self, prompt: str) -> dict[str, Any]:
        """Get JSON response from LLM using appropriate mode.

        Args:
            prompt: User prompt to send

        Returns:
            Parsed JSON response dict
        """
        if self._llm_agent:
            # Academy agent mode
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
            "microkinetic_lite": ("surrogate", "microkinetic"),
            "dft_adsorption": ("qe", "adsorption"),
            "stability_analysis": ("stability", "analyze"),
            "cantera_reactor": ("cantera", "reactor"),
            "cantera_sensitivity": ("cantera", "sensitivity"),
            "openmm_relaxation": ("openmm", "relaxation"),
            "chgnet_screening": ("chgnet", "screening"),
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

        logger.info("Dispatching to %s.%s", agent_name, action_name)

        # Get the action method
        action_method = getattr(agent, action_name, None)
        if not action_method:
            raise RuntimeError(f"Agent {agent_name} has no action '{action_name}'")

        # Call the action
        result = await action_method({
            "candidate": candidate,
        })

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
            logger.error("Final assessment generation failed: %s", e)
            return {
                "viability_score": 50,
                "strengths": ["Tests completed successfully"],
                "concerns": [f"Assessment generation failed: {e}"],
                "recommendation": "DEPRIORITIZE",
                "summary": "Assessment generation failed; manual review required.",
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

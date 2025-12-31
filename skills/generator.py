"""GeneratorAgent - orchestrates candidate discovery using LLM-guided generation.

The GeneratorAgent:
1. Uses an LLM to propose new catalyst candidates based on evaluation history
2. Spawns ShepherdAgents on Spark via Globus Compute to evaluate each candidate
3. Collects results and uses them to guide the next generation
4. Decides when to stop (iteration limit, convergence, or LLM judgment)

Architecture:
- GeneratorAgent runs on local machine (Mac)
- LLM server (llama-cpp) runs on Spark
- ShepherdAgents run as GC tasks on Spark, connecting to shared LLM
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from orchestration.llm_client import LLMClient, create_llm_client_from_config
from orchestration.generator_state import (
    GenerationState,
    validate_candidate,
    normalize_candidate,
    candidate_to_str,
)
from orchestration.generator_prompts import (
    SYSTEM_PROMPT,
    build_proposal_prompt,
    build_convergence_prompt,
)

logger = logging.getLogger(__name__)

# Import Academy components conditionally
try:
    from academy.agent import Agent, action, loop
    from academy.manager import Manager
    from academy.exchange import LocalExchangeFactory
    ACADEMY_AVAILABLE = True
except ImportError:
    ACADEMY_AVAILABLE = False
    # Provide stub classes for when Academy isn't available
    class Agent:
        pass
    def action(f):
        return f
    def loop(f):
        return f


class GeneratorAgent(Agent):
    """Orchestrates catalyst candidate discovery using LLM and distributed ShepherdAgents.

    The generator runs an autonomous loop that:
    1. Proposes candidates via LLM
    2. Spawns ShepherdAgents on Spark via GC to evaluate each candidate
    3. Updates state and checks for convergence
    4. Continues until stopping criteria met
    """

    def __init__(
        self,
        config: dict[str, Any],
        shepherd_config: dict[str, Any] | None = None,
        gc_function_map: dict[str, str] | None = None,
        gc_endpoint: str | None = None,
        llm_url: str | None = None,
    ):
        """Initialize GeneratorAgent.

        Args:
            config: Generator-specific configuration with sections:
                - llm: LLM endpoint config (mode, model, urls)
                - generation: candidates_per_iteration, max_iterations
                - convergence: patience, min_improvement, llm_judgment
                - shepherd: budget_per_candidate, num_concurrent
                - state: checkpoint_path, results_path
            shepherd_config: Configuration passed to spawned ShepherdAgents
            gc_function_map: Globus Compute function mappings for simulations
            gc_endpoint: Globus Compute endpoint ID for running Shepherds on Spark
            llm_url: URL to shared LLM server on Spark (e.g., "http://localhost:8080/v1")
        """
        if ACADEMY_AVAILABLE:
            super().__init__()
        self._config = config
        self._shepherd_config = shepherd_config or {}
        self._gc_function_map = gc_function_map or {}
        self._gc_endpoint = gc_endpoint
        self._llm_url = llm_url

        # Initialized in agent_on_startup / run()
        self._llm: LLMClient | None = None
        self._state: GenerationState | None = None
        self._gc_executor = None
        self._shepherd_func_id: str | None = None
        self._running = False

    async def agent_on_startup(self) -> None:
        """Initialize resources after agent starts."""
        await self._initialize()

    def _initialize(self) -> None:
        """Initialize resources (can be called sync or async)."""
        # Initialize LLM client
        llm_config = self._config.get("llm", {})

        # If llm_url provided, use it directly
        if self._llm_url:
            from orchestration.llm_client import LLMClient
            self._llm = LLMClient(
                base_url=self._llm_url,
                model=llm_config.get("model", "gpt-3.5-turbo"),
            )
        else:
            self._llm = create_llm_client_from_config(self._config)

        logger.info(
            "GeneratorAgent LLM initialized: model=%s base_url=%s",
            self._llm.model,
            self._llm.base_url,
        )

        # Initialize or restore state
        state_config = self._config.get("state", {})
        checkpoint_path = state_config.get("checkpoint_path", "data/generator_state.json")

        if Path(checkpoint_path).exists():
            try:
                self._state = GenerationState.load_checkpoint(checkpoint_path)
                logger.info("Restored state from checkpoint (iteration %d)", self._state.iteration)
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s. Starting fresh.", e)
                self._state = GenerationState()
        else:
            self._state = GenerationState()

        # Setup Globus Compute for Shepherd execution on Spark
        if self._gc_endpoint:
            self._setup_gc()

    def _setup_gc(self) -> None:
        """Setup Globus Compute executor and register shepherd function."""
        from globus_compute_sdk import Client, Executor

        # Import the shepherd GC function
        from scripts.shepherd_gc import evaluate_candidate_gc

        logger.info("Setting up Globus Compute for endpoint: %s", self._gc_endpoint)

        client = Client()
        self._shepherd_func_id = client.register_function(evaluate_candidate_gc)
        logger.info("Registered shepherd function: %s", self._shepherd_func_id)

        # Create executor (will be used in _evaluate_candidates)
        self._gc_executor = Executor(endpoint_id=self._gc_endpoint)

    async def agent_on_shutdown(self) -> None:
        """Cleanup on shutdown."""
        self._running = False

        # Save final checkpoint
        if self._state:
            state_config = self._config.get("state", {})
            checkpoint_path = state_config.get("checkpoint_path", "data/generator_state.json")
            try:
                self._state.save_checkpoint(checkpoint_path)
            except Exception as e:
                logger.error("Failed to save checkpoint: %s", e)

        # Cleanup GC executor
        if self._gc_executor:
            try:
                self._gc_executor.shutdown()
            except Exception:
                pass

    @loop
    async def main_loop(self, shutdown: asyncio.Event) -> None:
        """Main generation loop - runs until convergence or shutdown."""
        self._running = True
        gen_config = self._config.get("generation", {})
        conv_config = self._config.get("convergence", {})

        max_iterations = gen_config.get("max_iterations", 20)
        candidates_per_iter = gen_config.get("candidates_per_iteration", 6)
        patience = conv_config.get("patience", 3)
        min_improvement = conv_config.get("min_improvement", 0.01)
        use_llm_judgment = conv_config.get("llm_judgment", True)

        logger.info(
            "Starting generation loop: max_iter=%d, candidates/iter=%d",
            max_iterations,
            candidates_per_iter,
        )

        while not shutdown.is_set() and self._running:
            iteration = self._state.iteration

            # Check iteration limit
            if iteration >= max_iterations:
                self._state.converged = True
                self._state.stop_reason = f"Reached max iterations ({max_iterations})"
                logger.info("Stopping: %s", self._state.stop_reason)
                break

            logger.info("=== Generation iteration %d ===", iteration + 1)

            # Step 1: Propose candidates via LLM
            candidates = await self._propose_candidates(candidates_per_iter)
            if not candidates:
                logger.warning("No valid candidates proposed, retrying...")
                await asyncio.sleep(1)
                continue

            logger.info("Proposed %d candidates", len(candidates))

            # Step 2: Evaluate candidates via ShepherdAgents
            results = await self._evaluate_candidates(candidates)

            # Step 3: Update state
            self._state.update_with_results(results)

            # Step 4: Save checkpoint
            state_config = self._config.get("state", {})
            checkpoint_path = state_config.get("checkpoint_path", "data/generator_state.json")
            self._state.save_checkpoint(checkpoint_path)

            # Append to results log
            results_path = state_config.get("results_path", "data/generator_results.jsonl")
            self._append_results(results_path, results)

            # Step 5: Check convergence
            if self._state.check_convergence(patience, min_improvement):
                logger.info("Stopping: %s", self._state.stop_reason)
                break

            # Step 6: Optional LLM convergence judgment
            if use_llm_judgment and iteration >= patience:
                should_stop = await self._check_llm_convergence()
                if should_stop:
                    logger.info("Stopping: LLM recommends stopping")
                    break

            logger.info(
                "Iteration %d complete: best_score=%.1f, total_evaluated=%d",
                iteration + 1,
                self._state.best_score,
                len(self._state.candidates_evaluated),
            )

        # Final summary
        logger.info("Generation complete: %s", json.dumps(self._state.get_summary(), indent=2))
        self.agent_shutdown()

    async def _propose_candidates(self, n: int) -> list[dict[str, Any]]:
        """Propose new candidates via LLM.

        Args:
            n: Number of candidates to propose

        Returns:
            List of validated candidate dicts
        """
        prompt = build_proposal_prompt(
            n_candidates=n,
            iteration=self._state.iteration,
            total_evaluated=len(self._state.candidates_evaluated),
            best_score=self._state.best_score,
            best_candidate=self._state.best_candidate,
            top_performers=self._state.get_top_performers(10),
            seen_candidates=self._state.seen_strings,
        )

        try:
            response = await self._llm.reason_json(prompt, SYSTEM_PROMPT)
        except Exception as e:
            logger.error("LLM proposal failed: %s", e)
            return []

        raw_candidates = response.get("candidates", [])
        reasoning = response.get("reasoning", "")
        logger.info("LLM proposal reasoning: %s", reasoning[:200])

        # Validate and normalize candidates
        validated = []
        for raw in raw_candidates:
            # Normalize first
            candidate = normalize_candidate(raw)

            # Validate
            valid, error = validate_candidate(candidate, self._state.seen_hashes)
            if not valid:
                logger.warning("Invalid candidate: %s - %s", candidate_to_str(candidate), error)
                continue

            validated.append(candidate)

        return validated

    async def _evaluate_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate candidates using ShepherdAgents on Spark via GC.

        Args:
            candidates: List of candidate dicts to evaluate

        Returns:
            List of ShepherdAgent result dicts
        """
        shepherd_config = self._config.get("shepherd", {})
        budget = shepherd_config.get("budget_per_candidate", 100.0)
        num_concurrent = shepherd_config.get("num_concurrent", 4)
        timeout = shepherd_config.get("timeout", 3600)

        if not self._gc_executor or not self._shepherd_func_id:
            logger.error("GC not configured - cannot evaluate candidates on Spark")
            return [{"candidate": c, "error": "GC not configured"} for c in candidates]

        logger.info(
            "Submitting %d candidates to Spark (max concurrent: %d)",
            len(candidates), num_concurrent
        )

        # Submit all candidates as GC tasks
        futures = []
        for candidate in candidates:
            config = {
                "candidate": candidate,
                "llm_url": self._llm_url or "http://localhost:8000/v1",
                "budget": budget,
                "gc_functions": self._gc_function_map,
                "gc_endpoint": self._gc_endpoint,  # For nested simulation calls
            }
            future = self._gc_executor.submit_to_registered_function(
                self._shepherd_func_id,
                args=(config,),
            )
            futures.append((candidate, future))
            logger.info("Submitted evaluation for: %s", candidate_to_str(candidate))

        # Collect results (with timeout)
        results = []
        for candidate, future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
                score = result.get("final_assessment", {}).get("viability_score", 0)
                logger.info(
                    "Completed: %s (score: %d)",
                    candidate_to_str(candidate), score
                )
            except TimeoutError:
                logger.error("Timeout for: %s", candidate_to_str(candidate))
                results.append({"candidate": candidate, "error": "timeout", "ok": False})
            except Exception as e:
                logger.error("Error for %s: %s", candidate_to_str(candidate), e)
                results.append({"candidate": candidate, "error": str(e), "ok": False})

        return results

    async def _check_llm_convergence(self) -> bool:
        """Ask LLM whether to continue or stop.

        Returns:
            True if LLM recommends stopping, False otherwise
        """
        prompt = build_convergence_prompt(
            iteration=self._state.iteration,
            total_evaluated=len(self._state.candidates_evaluated),
            best_score=self._state.best_score,
            score_history=self._state.score_history,
        )

        try:
            response = await self._llm.reason_json(prompt, SYSTEM_PROMPT)
        except Exception as e:
            logger.warning("LLM convergence check failed: %s", e)
            return False

        decision = response.get("decision", "CONTINUE")
        reasoning = response.get("reasoning", "")
        confidence = response.get("confidence", 0)

        logger.info(
            "LLM convergence judgment: %s (confidence=%.2f) - %s",
            decision,
            confidence,
            reasoning[:100],
        )

        if decision == "STOP" and confidence >= 0.7:
            self._state.converged = True
            self._state.stop_reason = f"LLM recommended stopping: {reasoning[:100]}"
            return True

        return False

    def _append_results(self, path: str, results: list[dict[str, Any]]) -> None:
        """Append results to JSONL log file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with p.open("a", encoding="utf-8") as f:
            for r in results:
                record = {
                    "timestamp": time.time(),
                    "iteration": self._state.iteration,
                    "result": r,
                }
                f.write(json.dumps(record) + "\n")

    @action
    async def get_status(self, req: dict) -> dict:
        """Get current generation status.

        Returns:
            Status dict with iteration, best_score, converged, etc.
        """
        if not self._state:
            return {"error": "State not initialized"}

        return {
            "running": self._running,
            **self._state.get_summary(),
        }

    @action
    async def get_results(self, req: dict) -> dict:
        """Get all evaluated candidates and their assessments.

        Args:
            req: Optional filters:
                - top_n: Return only top N by score (default: all)
                - min_score: Filter by minimum score

        Returns:
            Dict with candidates list
        """
        if not self._state:
            return {"error": "State not initialized", "candidates": []}

        top_n = req.get("top_n")
        min_score = req.get("min_score", 0)

        performers = self._state.get_top_performers(
            top_n if top_n else len(self._state.candidates_evaluated)
        )

        if min_score > 0:
            performers = [
                p for p in performers
                if p["final_assessment"].get("viability_score", 0) >= min_score
            ]

        return {
            "total_evaluated": len(self._state.candidates_evaluated),
            "candidates": performers,
        }

    @action
    async def stop(self, req: dict) -> dict:
        """Request the generator to stop after current iteration.

        Args:
            req: Optional:
                - reason: Reason for stopping

        Returns:
            Acknowledgment dict
        """
        reason = req.get("reason", "User requested stop")
        self._running = False
        self._state.stop_reason = reason
        logger.info("Stop requested: %s", reason)
        return {"acknowledged": True, "reason": reason}

    async def run(self) -> dict[str, Any]:
        """Run the generation loop (simple API without Academy).

        This is a simpler alternative to using the @loop decorator.
        Call this directly to run the generator.

        Returns:
            Final state summary dict
        """
        # Initialize if not already done
        if self._llm is None:
            self._initialize()

        self._running = True
        gen_config = self._config.get("generation", {})
        conv_config = self._config.get("convergence", {})

        max_iterations = gen_config.get("max_iterations", 20)
        candidates_per_iter = gen_config.get("candidates_per_iteration", 6)
        patience = conv_config.get("patience", 3)
        min_improvement = conv_config.get("min_improvement", 0.01)
        use_llm_judgment = conv_config.get("llm_judgment", True)

        logger.info(
            "Starting generation: max_iter=%d, candidates/iter=%d, endpoint=%s",
            max_iterations,
            candidates_per_iter,
            self._gc_endpoint or "local",
        )

        while self._running:
            iteration = self._state.iteration

            # Check iteration limit
            if iteration >= max_iterations:
                self._state.converged = True
                self._state.stop_reason = f"Reached max iterations ({max_iterations})"
                logger.info("Stopping: %s", self._state.stop_reason)
                break

            logger.info("=== Generation iteration %d ===", iteration + 1)

            # Step 1: Propose candidates via LLM
            candidates = await self._propose_candidates(candidates_per_iter)
            if not candidates:
                logger.warning("No valid candidates proposed, retrying...")
                await asyncio.sleep(1)
                continue

            logger.info("Proposed %d candidates", len(candidates))

            # Step 2: Evaluate candidates via ShepherdAgents on Spark
            results = await self._evaluate_candidates(candidates)

            # Step 3: Update state
            self._state.update_with_results(results)

            # Step 4: Save checkpoint
            state_config = self._config.get("state", {})
            checkpoint_path = state_config.get("checkpoint_path", "data/generator_state.json")
            self._state.save_checkpoint(checkpoint_path)

            # Append to results log
            results_path = state_config.get("results_path", "data/generator_results.jsonl")
            self._append_results(results_path, results)

            # Step 5: Check convergence
            if self._state.check_convergence(patience, min_improvement):
                logger.info("Stopping: %s", self._state.stop_reason)
                break

            # Step 6: Optional LLM convergence judgment
            if use_llm_judgment and iteration >= patience:
                should_stop = await self._check_llm_convergence()
                if should_stop:
                    logger.info("Stopping: LLM recommends stopping")
                    break

            logger.info(
                "Iteration %d complete: best_score=%.1f, total_evaluated=%d",
                iteration + 1,
                self._state.best_score,
                len(self._state.candidates_evaluated),
            )

        # Cleanup
        if self._gc_executor:
            try:
                self._gc_executor.shutdown()
            except Exception:
                pass

        return self._state.get_summary()

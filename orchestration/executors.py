from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExecResult:
    characterizer: str
    status: str            # "SUCCEEDED" | "FAILED" | "SKIPPED" | "CACHED"
    result: Dict[str, Any]
    latency_s: float
    provenance: Dict[str, Any] | None = None


class LocalExecutor:
    """
    Executes characterizers using local tools in ctx.
    Expected ctx callables (async):
      - encode_catalyst(support=..., metals=...)
      - predict_performance(feature_vector=[...])
      - estimate_catalyst_cost(support=..., metals=...)
      - microkinetic_lite(candidate=..., performance=...)   (optional)
    Optional cache in ctx:
      - cache_key(candidate, characterizer) -> str
      - cache_get(key) -> value|None
      - cache_set(key, value)
    """

    def __init__(self, ctx: dict):
        self.ctx = ctx

    async def run(self, characterizer: str, candidate: dict, history: dict, *, timeout_s: Optional[float] = None) -> ExecResult:
        t0 = time.time()
        try:
            # Cache check
            cache_key = None
            if "cache_key" in self.ctx and "cache_get" in self.ctx:
                cache_key = self.ctx["cache_key"](candidate, characterizer)
                cached = self.ctx["cache_get"](cache_key)
                if cached is not None:
                    return ExecResult(
                        characterizer=characterizer,
                        status="CACHED",
                        result=cached,
                        latency_s=time.time() - t0,
                        provenance={"mode": "local", "cache": True},
                    )

            # Dispatch
            if characterizer == "fast_surrogate":
                async def _do():
                    enc = await self.ctx["encode_catalyst"](support=candidate["support"], metals=candidate["metals"])
                    perf = await self.ctx["predict_performance"](feature_vector=enc["feature_vector"])
                    cost = await self.ctx["estimate_catalyst_cost"](support=candidate["support"], metals=candidate["metals"])
                    return {"encoding": enc, "performance": perf, "catalyst_cost": cost}

                computed = await asyncio.wait_for(_do(), timeout=timeout_s) if timeout_s else await _do()

            elif characterizer == "microkinetic_lite":
                if "microkinetic_lite" not in self.ctx:
                    return ExecResult(
                        characterizer=characterizer,
                        status="SKIPPED",
                        result={"reason": "tool not available"},
                        latency_s=time.time() - t0,
                        provenance={"mode": "local"},
                    )

                fs = history.get("fast_surrogate")
                if not fs:
                    return ExecResult(
                        characterizer=characterizer,
                        status="SKIPPED",
                        result={"reason": "requires fast_surrogate first"},
                        latency_s=time.time() - t0,
                        provenance={"mode": "local"},
                    )

                async def _do():
                    return await self.ctx["microkinetic_lite"](candidate=candidate, performance=fs["performance"])

                computed = await asyncio.wait_for(_do(), timeout=timeout_s) if timeout_s else await _do()

            elif characterizer == "dft_adsorption":
                # Placeholder until GC/HPC is wired
                return ExecResult(
                    characterizer=characterizer,
                    status="SKIPPED",
                    result={"reason": "HPC not wired"},
                    latency_s=time.time() - t0,
                    provenance={"mode": "local"},
                )

            else:
                return ExecResult(
                    characterizer=characterizer,
                    status="SKIPPED",
                    result={"reason": f"unknown characterizer '{characterizer}'"},
                    latency_s=time.time() - t0,
                    provenance={"mode": "local"},
                )

            # Cache write
            if cache_key is not None and "cache_set" in self.ctx:
                self.ctx["cache_set"](cache_key, computed)

            return ExecResult(
                characterizer=characterizer,
                status="SUCCEEDED",
                result=computed,
                latency_s=time.time() - t0,
                provenance={"mode": "local"},
            )

        except Exception as e:
            return ExecResult(
                characterizer=characterizer,
                status="FAILED",
                result={"error": repr(e)},
                latency_s=time.time() - t0,
                provenance={"mode": "local"},
            )

    async def run_batch(
        self,
        characterizer: str,
        candidates: List[dict],
        histories: Dict[str, dict],
        *,
        timeout_s: Optional[float] = None,
        concurrency: int = 32,
    ) -> Dict[str, ExecResult]:
        """
        Run a characterizer over many candidates with bounded concurrency.
        histories: candidate_key -> history dict
        Returns: candidate_key -> ExecResult
        """
        sem = asyncio.Semaphore(concurrency)

        async def _one(ck: str, c: dict) -> tuple[str, ExecResult]:
            async with sem:
                return ck, await self.run(characterizer, c, histories.get(ck, {}), timeout_s=timeout_s)

        tasks = [_one(ck, c) for ck, c in candidates]
        out_pairs = await asyncio.gather(*tasks)
        return {ck: res for ck, res in out_pairs}




class GCExecutor:
    """
    Executes characterizers via Globus Compute using ctx tools:
      - submit_characterization(characterizer=..., payload=...) -> {task_id, ...}
      - get_characterization(task_id=...) -> {status, result|error}

    GC characterizers are expected to return JSON-serializable dicts.

    Optional cache hooks in ctx:
      - cache_key(candidate, characterizer) -> str
      - cache_get(key) -> value|None
      - cache_set(key, value)
    """

    def __init__(
        self,
        ctx: dict,
        *,
        poll_interval_s: float = 0.25,
        timeout_s: Optional[float] = None,
    ):
        self.ctx = ctx
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s

        # Fail fast if GC tools aren't present
        if "submit_characterization" not in ctx or "get_characterization" not in ctx:
            raise ValueError("GCExecutor requires ctx['submit_characterization'] and ctx['get_characterization'].")

    async def run(
        self,
        characterizer: str,
        candidate: dict,
        history: dict,
        *,
        timeout_s: Optional[float] = None,
    ):
        t0 = time.time()
        timeout = timeout_s if timeout_s is not None else self.timeout_s

        try:
            # Cache check
            cache_key = None
            if "cache_key" in self.ctx and "cache_get" in self.ctx:
                cache_key = self.ctx["cache_key"](candidate, characterizer)
                cached = self.ctx["cache_get"](cache_key)
                if cached is not None:
                    return ExecResult(
                        characterizer=characterizer,
                        status="CACHED",
                        result=cached,
                        latency_s=time.time() - t0,
                        provenance={"mode": "globus_compute", "cache": True},
                    )

            payload = {
                "candidate": candidate,
                "history": history,   # include if you want multi-stage char; optional for fast_surrogate
            }

            submit = self.ctx["submit_characterization"]
            get = self.ctx["get_characterization"]

            sub = await submit(characterizer=characterizer, payload=payload)
            task_id = sub.get("task_id")
            if not task_id:
                return ExecResult(
                    characterizer=characterizer,
                    status="FAILED",
                    result={"error": f"submit returned no task_id: {sub}"},
                    latency_s=time.time() - t0,
                    provenance={"mode": "globus_compute"},
                )

            # Poll for completion
            async def _poll() -> Dict[str, Any]:
                while True:
                    r = await get(task_id=task_id)
                    status = r.get("status", "UNKNOWN")
                    if status == "SUCCEEDED":
                        return {"status": "SUCCEEDED", "result": r.get("result"), "meta": sub}
                    if status == "FAILED":
                        return {"status": "FAILED", "error": r.get("error", "unknown"), "meta": sub}
                    await asyncio.sleep(self.poll_interval_s)

            polled = await asyncio.wait_for(_poll(), timeout=timeout) if timeout else await _poll()

            if polled["status"] == "FAILED":
                return ExecResult(
                    characterizer=characterizer,
                    status="FAILED",
                    result={"error": polled["error"], "task": {"task_id": task_id, **polled.get("meta", {})}},
                    latency_s=time.time() - t0,
                    provenance={"mode": "globus_compute"},
                )

            computed = polled["result"]

            # Cache write
            if cache_key is not None and "cache_set" in self.ctx:
                self.ctx["cache_set"](cache_key, computed)

            return ExecResult(
                characterizer=characterizer,
                status="SUCCEEDED",
                result=computed,
                latency_s=time.time() - t0,
                provenance={"mode": "globus_compute", "task": {"task_id": task_id, **polled.get("meta", {})}},
            )

        except asyncio.TimeoutError:
            return ExecResult(
                characterizer=characterizer,
                status="FAILED",
                result={"error": f"timeout after {timeout}s"},
                latency_s=time.time() - t0,
                provenance={"mode": "globus_compute"},
            )
        except Exception as e:
            return ExecResult(
                characterizer=characterizer,
                status="FAILED",
                result={"error": repr(e)},
                latency_s=time.time() - t0,
                provenance={"mode": "globus_compute"},
            )

    async def run_batch(
        self,
        characterizer: str,
        candidates: List[Tuple[str, dict]],   # list of (candidate_key, candidate)
        histories: Dict[str, dict],           # candidate_key -> history
        *,
        timeout_s: Optional[float] = None,
        concurrency: int = 128,
    ) -> Dict[str, "ExecResult"]:
        """
        Submit all tasks (bounded concurrency), then poll until completion.
        Returns candidate_key -> ExecResult
        """
        timeout = timeout_s if timeout_s is not None else self.timeout_s
        submit = self.ctx["submit_characterization"]
        get = self.ctx["get_characterization"]

        sem = asyncio.Semaphore(concurrency)
        t0 = time.time()

        # 1) Submit phase (with cache short-circuit)
        pending: Dict[str, Dict[str, Any]] = {}   # ck -> {"task_id":..., "submitted":...}
        results: Dict[str, ExecResult] = {}

        async def _submit_one(ck: str, c: dict) -> None:
            nonlocal results, pending
            async with sem:
                # cache check
                cache_key = None
                if "cache_key" in self.ctx and "cache_get" in self.ctx:
                    cache_key = self.ctx["cache_key"](c, characterizer)
                    cached = self.ctx["cache_get"](cache_key)
                    if cached is not None:
                        results[ck] = ExecResult(
                            characterizer=characterizer,
                            status="CACHED",
                            result=cached,
                            latency_s=0.0,
                            provenance={"mode": "globus_compute", "cache": True},
                        )
                        return

                payload = {"candidate": c, "history": histories.get(ck, {})}
                sub = await submit(characterizer=characterizer, payload=payload)
                task_id = sub.get("task_id")
                if not task_id:
                    results[ck] = ExecResult(
                        characterizer=characterizer,
                        status="FAILED",
                        result={"error": f"submit returned no task_id: {sub}"},
                        latency_s=0.0,
                        provenance={"mode": "globus_compute"},
                    )
                    return

                pending[ck] = {"task_id": task_id, "meta": sub, "cache_key": cache_key, "candidate": c}

        await asyncio.gather(*[_submit_one(ck, c) for ck, c in candidates])

        # 2) Poll phase
        async def _poll_until_done() -> None:
            nonlocal results, pending
            start = time.time()
            while pending:
                # timeout
                if timeout and (time.time() - start) > timeout:
                    for ck, info in list(pending.items()):
                        results[ck] = ExecResult(
                            characterizer=characterizer,
                            status="FAILED",
                            result={"error": f"timeout after {timeout}s", "task": info["meta"]},
                            latency_s=time.time() - t0,
                            provenance={"mode": "globus_compute"},
                        )
                        pending.pop(ck, None)
                    return

                done_keys = []
                for ck, info in pending.items():
                    r = await get(task_id=info["task_id"])
                    status = r.get("status", "UNKNOWN")
                    if status == "SUCCEEDED":
                        computed = r.get("result")
                        # cache write
                        if info.get("cache_key") is not None and "cache_set" in self.ctx:
                            self.ctx["cache_set"](info["cache_key"], computed)

                        results[ck] = ExecResult(
                            characterizer=characterizer,
                            status="SUCCEEDED",
                            result=computed,
                            latency_s=time.time() - t0,
                            provenance={"mode": "globus_compute", "task": info["meta"]},
                        )
                        done_keys.append(ck)

                    elif status == "FAILED":
                        results[ck] = ExecResult(
                            characterizer=characterizer,
                            status="FAILED",
                            result={"error": r.get("error", "unknown"), "task": info["meta"]},
                            latency_s=time.time() - t0,
                            provenance={"mode": "globus_compute"},
                        )
                        done_keys.append(ck)

                for ck in done_keys:
                    pending.pop(ck, None)

                if pending:
                    await asyncio.sleep(self.poll_interval_s)

        await _poll_until_done()
        return results

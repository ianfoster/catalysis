from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import time


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_cache_key(candidate: dict, characterizer: str, version: str) -> str:
    blob = _stable_json({"candidate": candidate, "characterizer": characterizer, "version": version})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class JsonlCache:
    path: Path
    index: Dict[str, Any]

    @classmethod
    def load(cls, path: str) -> "JsonlCache":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        index: Dict[str, Any] = {}
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    index[rec["key"]] = rec["value"]
        return cls(path=p, index=index)

    def get(self, key: str) -> Optional[Any]:
        return self.index.get(key)

    def set(self, key: str, value: Any) -> None:
        # Overwrite in-memory; append to file (event-sourced)
        self.index[key] = value
        rec = {"key": key, "value": value}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(_stable_json(rec) + "\n")


def detect_version() -> str:
    # Prefer an explicit version; fall back to git if available; else "dev"
    env = os.getenv("CATALYSIS_VERSION")
    if env:
        return env


def git_short_sha() -> str:
    """Best-effort git short SHA."""
    try:
        import subprocess
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        if sha:
            return sha
    except Exception:
        pass
    return "dev"
    

@dataclass
class OpenMMJsonlCache:
    path: Path
    index: Dict[str, Dict[str, Any]]

    @classmethod
    def load(cls, path: str | Path) -> "OpenMMJsonlCache":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        index: Dict[str, Dict[str, Any]] = {}
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        k = rec.get("cache_key")
                        if k:
                            index[k] = rec
                    except Exception:
                        continue
        return cls(path=p, index=index)

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        rec = self.index.get(cache_key)
        if not rec:
            return None
        return rec.get("value")

    def set(
        self,
        cache_key: str,
        value: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec = {
            "ts": time.time(),
            "cache_key": cache_key,
            "value": value,
            "meta": {
                **(meta or {}),
                "code_sha": git_short_sha(),
            },
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.index[cache_key] = rec


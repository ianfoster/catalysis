from __future__ import annotations
import json
from pathlib import Path
from typing import Set

def load_seen(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    seen = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen.add(json.loads(line)["candidate_id"])
    return seen

def append_seen(path: str, candidate_ids: Set[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for cid in candidate_ids:
            f.write(json.dumps({"candidate_id": cid}) + "\n")

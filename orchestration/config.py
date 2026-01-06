from __future__ import annotations
from pathlib import Path
import yaml

def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(base_path="config.yaml", local_path="config.local.yaml") -> dict:
    base = {}
    if Path(base_path).exists():
        base = yaml.safe_load(Path(base_path).read_text()) or {}
    local = {}
    if Path(local_path).exists():
        local = yaml.safe_load(Path(local_path).read_text()) or {}
    return deep_merge(base, local)

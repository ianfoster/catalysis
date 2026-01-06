from __future__ import annotations
import hashlib
import json
from typing import Any, Dict
from rdkit import Chem

def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def candidate_id(candidate: Dict[str, Any]) -> str:
    # 16 hex chars is plenty for logs; keep full sha if you prefer
    h = hashlib.sha256(stable_json(candidate).encode("utf-8")).hexdigest()
    return h[:16]

def mol_id_from_smiles(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return hashlib.sha1(smiles.encode()).hexdigest()[:16]
    can = Chem.MolToSmiles(m, canonical=True)
    return hashlib.sha1(can.encode()).hexdigest()[:16]

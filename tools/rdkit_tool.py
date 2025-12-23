from __future__ import annotations

from typing import Any, Dict

from academy.agent import Agent, action

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.error")

def rdkit_descriptors(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a compact RDKit descriptor set for a SMILES string.

    payload = {"smiles": "..."}
    returns {"ok": bool, "smiles": "...", "descriptors": {...}} or {"ok": False, "error": "..."}
    """
    smiles = payload.get("smiles")
    if not smiles:
        return {"ok": False, "error": "missing 'smiles' in payload"}

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"ok": False, "smiles": smiles, "error": "RDKit failed to parse SMILES"}

        # Keep a small, stable set (avoid huge vectors initially)
        desc = {
            "MolWt": float(Descriptors.MolWt(mol)),
            "MolLogP": float(Crippen.MolLogP(mol)),
            "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "HBD": int(rdMolDescriptors.CalcNumHBD(mol)),
            "HBA": int(rdMolDescriptors.CalcNumHBA(mol)),
            "RingCount": int(rdMolDescriptors.CalcNumRings(mol)),
            "HeavyAtomCount": int(mol.GetNumHeavyAtoms()),
            "RotatableBonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        }

        return {"ok": True, "smiles": smiles, "descriptors": desc}

    except ImportError as e:
        return {"ok": False, "smiles": smiles, "error": f"RDKit not installed: {e!r}"}
    except Exception as e:
        return {"ok": False, "smiles": smiles, "error": repr(e)}


q#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange


def canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return smiles
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return smiles


def mol_id_from_smiles(smiles: str) -> str:
    can = canonical_smiles(smiles)
    return hashlib.sha1(can.encode("utf-8")).hexdigest()[:16]


def write_default_mdp(outdir: Path) -> None:
    # Conservative settings that should grompp cleanly for a single molecule in a big box.
    mdp = """\
integrator      = md
nsteps          = 500
dt              = 0.002

nstxout         = 100
nstvout         = 100
nstenergy       = 100
nstlog          = 100

continuation    = no
constraints     = none

tcoupl          = v-rescale
tc-grps         = System
tau_t           = 0.1
ref_t           = 300

pcoupl          = no

cutoff-scheme   = Verlet
nstlist         = 10
rlist           = 0.4
rcoulomb        = 0.4
rvdw            = 0.4

coulombtype     = Cut-off
vdwtype         = Cut-off

gen_vel         = yes
gen_temp        = 300
gen_seed        = 42
"""
    (outdir / "md.mdp").write_text(mdp, encoding="utf-8")


def force_box_in_gro(gro_path: Path, box_nm: float = 3.0) -> None:
    lines = gro_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return
    lines[-1] = f"{box_nm:.3f} {box_nm:.3f} {box_nm:.3f}"
    gro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_gromacs(interchange: Interchange, outdir: Path) -> Dict[str, Any]:
    """
    Robustly export GROMACS files across Interchange API variants.
    Returns export metadata for debugging.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    attempts: List[Tuple[str, str]] = []

    # Try method-based API variants
    if hasattr(interchange, "to_gromacs"):
        variants = [
            ("method: to_gromacs(prefix='topol', output_dir=...)", lambda: interchange.to_gromacs(prefix="topol", output_dir=str(outdir))),
            ("method: to_gromacs(output_dir=...)",               lambda: interchange.to_gromacs(str(outdir))),
            ("method: to_gromacs(output_dir=..., prefix='topol')",lambda: interchange.to_gromacs(str(outdir), prefix="topol")),
        ]
        for desc, fn in variants:
            try:
                ret = fn()
                return {"api": desc, "ret": str(ret)}
            except Exception as e:
                attempts.append((desc, f"{type(e).__name__}: {e}"))

    # Try functional exporter if present
    try:
        from openff.interchange.interop.gromacs.export import to_gromacs  # type: ignore
        variants2 = [
            ("func: to_gromacs(interchange, prefix='topol', output_dir=...)", lambda: to_gromacs(interchange, prefix="topol", output_dir=str(outdir))),
            ("func: to_gromacs(interchange, output_dir=...)",                 lambda: to_gromacs(interchange, str(outdir))),
        ]
        for desc, fn in variants2:
            try:
                ret = fn()
                return {"api": desc, "ret": str(ret)}
            except Exception as e:
                attempts.append((desc, f"{type(e).__name__}: {e}"))
    except Exception as e:
        attempts.append(("import: openff.interchange.interop.gromacs.export.to_gromacs", f"{type(e).__name__}: {e}"))

    raise RuntimeError("GROMACS export failed. Attempts:\n" + "\n".join([f"- {d}: {msg}" for d, msg in attempts]))


def normalize_outputs(outdir: Path) -> Dict[str, str]:
    """
    Ensure we end with:
      outdir/system.gro
      outdir/topol.top
    even if exporter wrote different names.
    """
    # topology
    if (outdir / "topol.top").exists():
        top = outdir / "topol.top"
    else:
        tops = list(outdir.glob("*.top")) + list(outdir.glob("*.topol")) + list(outdir.glob("*.itp"))  # defensive
        # exporters should produce .top or topol.top; if not, we error explicitly below
        top = outdir / "topol.top"
        # If a .top exists, prefer it
        for cand in list(outdir.glob("*.top")):
            cand.rename(outdir / "topol.top")
            top = outdir / "topol.top"
            break

    # coordinates
    if (outdir / "system.gro").exists():
        gro = outdir / "system.gro"
    else:
        gros = list(outdir.glob("*.gro"))
        if gros:
            gros[0].rename(outdir / "system.gro")
            gro = outdir / "system.gro"
        else:
            gro = outdir / "system.gro"

    # Final validation
    if not (outdir / "topol.top").exists():
        raise RuntimeError(f"Exporter did not produce topol.top (files={sorted([p.name for p in outdir.iterdir()])})")
    if not (outdir / "system.gro").exists():
        raise RuntimeError(f"Exporter did not produce system.gro (files={sorted([p.name for p in outdir.iterdir()])})")

    return {"topol.top": str((outdir / "topol.top").resolve()), "system.gro": str((outdir / "system.gro").resolve())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out-root", default="data/gromacs_inputs")
    ap.add_argument("--forcefield", default="openff_unconstrained-2.1.0.offxml")
    ap.add_argument("--box-nm", type=float, default=3.0)
    args = ap.parse_args()

    smiles = args.smiles
    can = canonical_smiles(smiles)
    mol_id = mol_id_from_smiles(smiles)

    outdir = Path(args.out_root).resolve() / mol_id
    outdir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "ok": False,
        "smiles": smiles,
        "canonical_smiles": can,
        "mol_id": mol_id,
        "forcefield": args.forcefield,
        "outdir": str(outdir),
        "files": {},
        "export": {},
        "error": None,
    }

    try:
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        top = Topology.from_molecules([mol])

        ff = ForceField(args.forcefield)
        interchange = Interchange.from_smirnoff(ff, top)

        meta["export"] = export_gromacs(interchange, outdir)

        files = normalize_outputs(outdir)
        force_box_in_gro(Path(files["system.gro"]), box_nm=float(args.box_nm))
        write_default_mdp(outdir)

        # required files check
        required = ["md.mdp", "system.gro", "topol.top"]
        missing = [f for f in required if not (outdir / f).exists()]
        if missing:
            raise RuntimeError(f"Missing required files after export: {missing}")

        meta["files"] = {
            "mdp": str((outdir / "md.mdp").resolve()),
            "gro": str((outdir / "system.gro").resolve()),
            "top": str((outdir / "topol.top").resolve()),
        }
        meta["ok"] = True

    except Exception as e:
        meta["error"] = repr(e)

    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0 if meta["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# OpenFF
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField

# Interchange
from openff.interchange import Interchange


def mol_id_from_smiles(smiles: str) -> str:
    # stable ID from canonical SMILES (fallback to raw)
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            can = smiles
        else:
            can = Chem.MolToSmiles(m, canonical=True)
    except Exception:
        can = smiles
    return hashlib.sha1(can.encode("utf-8")).hexdigest()[:16]


def write_default_mdp(outdir: Path, name: str = "md.mdp") -> None:
    # Short, safe demo MDP (small cutoffs, no PME) to avoid box issues
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
    (outdir / name).write_text(mdp, encoding="utf-8")


def force_box_in_gro(gro_path: Path, box_nm: float = 3.0) -> None:
    # Ensure last line has a valid cubic box; grompp will complain otherwise.
    lines = gro_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return
    lines[-1] = f"{box_nm:.3f} {box_nm:.3f} {box_nm:.3f}"
    gro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_to_gromacs(interchange: Interchange, outdir: Path) -> dict:
    """
    Export to GROMACS files in outdir.

    Interchange API differs across versions. Try multiple signatures.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    errors = []

    # 1) method-based exports (various versions)
    if hasattr(interchange, "to_gromacs"):
        for desc, fn in [
            ("to_gromacs(prefix, output_dir)", lambda: interchange.to_gromacs(prefix="topol", output_dir=str(outdir))),
            ("to_gromacs(output_dir)",        lambda: interchange.to_gromacs(str(outdir))),
            ("to_gromacs(output_dir, prefix)",lambda: interchange.to_gromacs(str(outdir), prefix="topol")),
        ]:
            try:
                ret = fn()
                return {"api": desc, "ret": str(ret)}
            except Exception as e:
                errors.append(f"{desc}: {type(e).__name__}: {e}")

    # 2) functional export path (some versions)
    try:
        from openff.interchange.interop.gromacs.export import to_gromacs  # type: ignore
        try:
            ret = to_gromacs(interchange, prefix="topol", output_dir=str(outdir))
            return {"api": "interop.to_gromacs(prefix, output_dir)", "ret": str(ret)}
        except Exception as e:
            errors.append(f"interop.to_gromacs(prefix, output_dir): {type(e).__name__}: {e}")

        try:
            ret = to_gromacs(interchange, str(outdir))
            return {"api": "interop.to_gromacs(output_dir)", "ret": str(ret)}
        except Exception as e:
            errors.append(f"interop.to_gromacs(output_dir): {type(e).__name__}: {e}")
    except Exception as e:
        errors.append(f"import interop exporter failed: {type(e).__name__}: {e}")

    raise RuntimeError("Could not export to GROMACS. Attempts:\n  - " + "\n  - ".join(errors))


def export_to_gromacs_OLD(interchange: Interchange, outdir: Path) -> dict:
    """
    Export to GROMACS files in outdir.

    We try a couple of Interchange export APIs because they vary across versions.
    Returns dict with paths for debugging.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Try method-based export first
    if hasattr(interchange, "to_gromacs"):
        # Some versions: to_gromacs(prefix, output_dir)
        try:
            ret = interchange.to_gromacs(prefix="topol", output_dir=str(outdir))
            return {"api": "interchange.to_gromacs", "ret": str(ret)}
        except TypeError:
            # Some versions: to_gromacs(path)
            ret = interchange.to_gromacs(str(outdir))
            return {"api": "interchange.to_gromacs", "ret": str(ret)}

    # Try functional export path
    try:
        from openff.interchange.interop.gromacs.export import to_gromacs  # type: ignore
        ret = to_gromacs(interchange, prefix="topol", output_dir=str(outdir))
        return {"api": "openff.interchange.interop.gromacs.export.to_gromacs", "ret": str(ret)}
    except Exception as e:
        raise RuntimeError(f"Could not export to GROMACS with this Interchange build: {e!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out-root", default="data/gromacs_inputs")
    ap.add_argument("--forcefield", default="openff_unconstrained-2.1.0.offxml")
    ap.add_argument("--box-nm", type=float, default=3.0)
    args = ap.parse_args()

    smiles = args.smiles
    mol_id = mol_id_from_smiles(smiles)

    outdir = Path(args.out_root).resolve() / mol_id
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Build OpenFF molecule + conformer
    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    top = Topology.from_molecules([mol])

    # 2) ForceField -> Interchange
    ff = ForceField(args.forcefield)
    interchange = Interchange.from_smirnoff(ff, top)

    # 3) Export to GROMACS
    export_info = export_to_gromacs(interchange, outdir)

    # The export should create a topology + coordinate file.
    # Common names:
    # - topol.top
    # - topol.gro
    # Some builds output conf.gro; we normalize.
    gro_candidates = list(outdir.glob("*.gro"))
    top_candidates = list(outdir.glob("*.top"))

    if not top_candidates:
        # Sometimes it writes topol.top directly
        if not (outdir / "topol.top").exists():
            raise RuntimeError(f"No .top/.topol.top written in {outdir} (files={list(outdir.iterdir())})")

    # Normalize names
    if (outdir / "topol.top").exists():
        topol_top = outdir / "topol.top"
    elif top_candidates:
        topol_top = top_candidates[0]
        topol_top.rename(outdir / "topol.top")
        topol_top = outdir / "topol.top"
    else:
        topol_top = outdir / "topol.top"

    if (outdir / "system.gro").exists():
        system_gro = outdir / "system.gro"
    elif gro_candidates:
        gro_candidates[0].rename(outdir / "system.gro")
        system_gro = outdir / "system.gro"
    else:
        raise RuntimeError(f"No .gro written in {outdir} (files={list(outdir.iterdir())})")

    # Force a safe box size
    force_box_in_gro(system_gro, box_nm=float(args.box_nm))

    # 4) Write mdp
    write_default_mdp(outdir, "md.mdp")

    meta = {
        "smiles": smiles,
        "mol_id": mol_id,
        "forcefield": args.forcefield,
        "outdir": str(outdir),
        "export": export_info,
        "files": {
            "topol": str(topol_top),
            "gro": str(system_gro),
            "mdp": str(outdir / "md.mdp"),
        },
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

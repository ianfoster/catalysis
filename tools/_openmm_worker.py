from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import openmm
import openmm.unit as unit
from openmm import Platform, LangevinIntegrator
from openmm.app import PDBFile, Simulation


def main() -> int:
    payload: Dict[str, Any] = json.loads(sys.stdin.read())

    t0 = time.time()
    run_dir = Path(payload["run_dir"]).resolve()
    system_xml = run_dir / payload.get("system_xml", "system.xml")
    positions_pdb = run_dir / payload.get("positions_pdb", "positions.pdb")
    platform_name = payload.get("platform", "OpenCL")
    max_it = int(payload.get("max_iterations", 2000))

    system = openmm.XmlSerializer.deserialize(system_xml.read_text(encoding="utf-8"))
    pdb = PDBFile(str(positions_pdb))

    integrator = LangevinIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    platform = Platform.getPlatformByName(platform_name)
    sim = Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.getPositions())

    sim.minimizeEnergy(maxIterations=max_it)

    state = sim.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    out = {
        "ok": True,
        "platform": platform_name,
        "potential_energy_kj_mol": float(pe),
        "openmm_version": getattr(openmm, "version", None).full_version if hasattr(openmm, "version") else None,
        "elapsed_s": time.time() - t0,
    }
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

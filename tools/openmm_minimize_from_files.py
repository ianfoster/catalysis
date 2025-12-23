from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import openmm
import openmm.unit as unit
from openmm import Platform, LangevinIntegrator
from openmm.app import PDBFile, Simulation


def openmm_minimize_from_files(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload = {
      "run_dir": "/home/ian/openmm-runs/<run_id>",
      "system_xml": "system.xml",
      "positions_pdb": "positions.pdb",
      "platform": "OpenCL",
      "max_iterations": 2000
    }
    """
    t0 = time.time()
    run_dir = Path(payload["run_dir"]).resolve()
    system_xml = run_dir / payload.get("system_xml", "system.xml")
    positions_pdb = run_dir / payload.get("positions_pdb", "positions.pdb")
    platform_name = payload.get("platform", "OpenCL")
    max_it = int(payload.get("max_iterations", 2000))

    if not system_xml.exists():
        return {"ok": False, "error": f"missing {system_xml}"}
    if not positions_pdb.exists():
        return {"ok": False, "error": f"missing {positions_pdb}"}

    system = openmm.XmlSerializer.deserialize(system_xml.read_text(encoding="utf-8"))
    pdb = PDBFile(str(positions_pdb))
    positions = pdb.getPositions()

    integrator = LangevinIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    platform = Platform.getPlatformByName(platform_name)
    sim = Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(positions)

    sim.minimizeEnergy(maxIterations=max_it)

    state = sim.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    return {
        "ok": True,
        "platform": platform_name,
        "potential_energy_kj_mol": float(pe),
        "runtime_s": time.time() - t0,
    }

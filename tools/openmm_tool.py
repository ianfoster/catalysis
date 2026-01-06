# tools/openmm_tool.py

import time
from openmm import unit
from openmm.app import Simulation, Topology
from openmm import LangevinIntegrator, Platform
from rdkit import Chem
from rdkit.Chem import AllChem


def run_openmm_minimization(smiles: str):
    t0 = time.time()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "error": "invalid_smiles"}

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    # Placeholder topology/system
    topology = Topology()
    integrator = LangevinIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    platform = Platform.getPlatformByName("CPU")

    simulation = Simulation(topology, None, integrator, platform)

    # Fake energy for now (replace with real forcefield)
    energy = -42.0 * unit.kilojoule_per_mole

    return {
        "ok": True,
        "energy_kj_mol": energy.value_in_unit(unit.kilojoule_per_mole),
        "runtime_s": time.time() - t0,
    }

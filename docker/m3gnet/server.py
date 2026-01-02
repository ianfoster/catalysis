#!/usr/bin/env python3
"""Simple M3GNet HTTP server for containerized deployment."""

import json
import logging
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model (loaded once at startup)
_potential = None
_calculator_class = None


def load_model():
    """Load M3GNet model at startup."""
    global _potential, _calculator_class

    import matgl
    # M3GNet model requires DGL backend (case-sensitive)
    matgl.set_backend("DGL")

    from matgl.ext.ase import PESCalculator

    logger.info("Loading M3GNet model...")
    _potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    _calculator_class = PESCalculator
    logger.info("M3GNet model loaded!")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"ok": True, "model": "M3GNet-MP-2021.2.8-PES"})


@app.route("/screening", methods=["POST"])
def screening():
    """Run M3GNet single-point energy calculation."""
    from ase.build import fcc111
    import numpy as np

    data = request.json
    candidate = data.get("candidate", {})
    metals = candidate.get("metals", [])

    # Extract metal percentages
    cu = next((m["wt_pct"] for m in metals if m["element"] == "Cu"), 60)
    zn = next((m["wt_pct"] for m in metals if m["element"] == "Zn"), 25)
    al = next((m["wt_pct"] for m in metals if m["element"] == "Al"), 15)

    try:
        # Build slab
        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        n_atoms = len(slab)
        total_wt = cu + zn + al + 0.01

        n_zn = int(n_atoms * zn / total_wt)
        n_al = int(n_atoms * al / total_wt)

        symbols = list(slab.get_chemical_symbols())
        for i in range(min(n_zn, n_atoms)):
            symbols[i] = "Zn"
        for i in range(n_zn, min(n_zn + n_al, n_atoms)):
            symbols[i] = "Al"
        slab.set_chemical_symbols(symbols)

        # Calculate
        calc = _calculator_class(potential=_potential)
        slab.calc = calc

        energy = float(slab.get_potential_energy())
        forces = slab.get_forces()
        max_force = float(np.max(np.abs(forces)))

        return jsonify({
            "ok": True,
            "method": "m3gnet",
            "total_energy_eV": round(energy, 6),
            "energy_per_atom_eV": round(energy / len(slab), 6),
            "max_force_eV_A": round(max_force, 6),
            "n_atoms": len(slab),
        })

    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8080)

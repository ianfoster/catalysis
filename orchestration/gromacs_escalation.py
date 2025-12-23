import time
import uuid
import logging
from pathlib import Path
from typing import Dict, Any
from orchestration.escalators.base import BaseEscalator
from orchestration.ids import mol_id_from_smiles

logger = logging.getLogger(__name__)

import subprocess

OPENFF311_PY = "/Users/ian/anaconda3/envs/openff311/bin/python"  # adjust if needed

def generate_gromacs_inputs(smiles: str) -> None:
    # Run the generator script in the openff311 environment
    subprocess.run(
        [OPENFF311_PY, "scripts/generate_gromacs_inputs.py", "--smiles", smiles],
        check=True,
        capture_output=False,
        text=True,
    )

class GromacsEscalator(BaseEscalator):
    def __init__(
        self,
        *,
        gc_client,
        spark_compute_ep: str,
        gmx_function_id: str,
        spark_base_dir: str,
        transfer_to_spark,
        transfer_from_spark,
    ):
        super().__init__(
            gc_client=gc_client,
            transfer_to_spark=transfer_to_spark,
            transfer_from_spark=transfer_from_spark,
        )

        self.gc = gc_client
        self.function_id = gmx_function_id
        self.spark_compute_ep = spark_compute_ep
        self.spark_base = Path(spark_base_dir)
        self.tx_to = transfer_to_spark
        self.tx_from = transfer_from_spark

    async def run(self, smiles: str) -> Dict[str, Any]:
        """
        Assumes GROMACS inputs already exist locally under:
          data/gromacs_inputs/<mol_id>/
        """
        run_id = uuid.uuid4().hex[:16]
        mol_id = mol_id_from_smiles(smiles)
        local_in = (Path("data/gromacs_inputs") / mol_id).resolve()
        spark_run = self.spark_base / run_id
        spark_in = spark_run / "input"
        spark_out = spark_run / "output"

        if not local_in.exists():
            if not local_in.exists():
                logger.info("[GROMACS] inputs missing; generating | smiles=%s | dir=%s", smiles, local_in)
                try:
                    generate_gromacs_inputs(smiles)
                except Exception as e:
                    return {"ok": False, "stage": "generate_inputs", "error": repr(e)}

            # re-check
            if not local_in.exists():
                return {"ok": False, "stage": "inputs", "error": f"missing_gromacs_inputs:{local_in}"}

        logger.info("[GROMACS] %s | transfer start", smiles)

        # ---- transfer inputs ----
        t_tx0 = time.time()
        await self.tx_to(
            src_path=str(local_in),
            dst_path=str(spark_in),
            recursive=True,
        )
        t_tx = time.time() - t_tx0

        # ---- submit compute ----
        payload = {
            "run_id": str(run_id), # XXXX
            "run_dir": str(spark_run),
            "input_dir": str(spark_in),
            "output_dir": str(spark_out),
        }

        t_gc0 = time.time()
        task_id = self.gc.run(
            function_id=self.function_id,
            endpoint_id=self.spark_compute_ep,
            payload=payload,
        )

        result = self._poll(task_id)
        t_gc = time.time() - t_gc0

        # ---- transfer outputs back ----
        await self.tx_from(
            src_path=str(spark_out),
            dst_path=f"data/gromacs_results/{run_id}",
            recursive=True,
        )

        result["cost"] = {
            "transfer_s": t_tx,
            "compute_s": t_gc,
        }
        result["run_id"] = run_id
        return result

    def _poll(self, task_id):
        while True:
            t = self.gc.get_task(task_id)
            if t["status"] == "success":
                return self.gc.get_result(task_id)
            if t["status"] == "failed":
                return {"ok": False, "error": t}
            time.sleep(1.0)

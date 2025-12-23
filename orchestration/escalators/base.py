from typing import Callable, Any

class BaseEscalator:
    """
    Shared infrastructure for all GC-based escalators
    (OpenMM, GROMACS, etc.)
    """

    def __init__(
        self,
        *,
        gc_client,
        transfer_to_spark: Callable[..., Any],
        transfer_from_spark: Callable[..., Any],
    ):
        self.gc = gc_client
        self.transfer_to_spark = transfer_to_spark
        self.transfer_from_spark = transfer_from_spark

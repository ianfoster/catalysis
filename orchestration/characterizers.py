from dataclasses import dataclass

@dataclass(frozen=True)
class Characterizer:
    name: str
    cost_level: str        # "cheap" | "medium" | "expensive"
    expected_runtime_s: int
    outputs: list[str]
    reduces_uncertainty: bool

CHARACTERIZERS = {
    "fast_surrogate": Characterizer(
        name="fast_surrogate",
        cost_level="cheap",
        expected_runtime_s=1,
        outputs=["co2_conversion", "methanol_selectivity", "methanol_sty"],
        reduces_uncertainty=False,
    ),
    "microkinetic_lite": Characterizer(
        name="microkinetic_lite",
        cost_level="medium",
        expected_runtime_s=60,
        outputs=["rate_constants", "RLS"],
        reduces_uncertainty=True,
    ),
    "dft_adsorption": Characterizer(
        name="dft_adsorption",
        cost_level="expensive",
        expected_runtime_s=3600,
        outputs=["E_ads_CO2", "E_ads_H"],
        reduces_uncertainty=True,
    ),
}

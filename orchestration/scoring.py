def score_candidate(history: dict) -> float:
    if "fast_surrogate" not in history:
        return float("-inf")

    fs = history["fast_surrogate"]
    perf = fs["performance"]
    cost = fs.get("catalyst_cost", {})

    score = (
        perf["methanol_sty"]
        + 2.0 * perf["methanol_selectivity"]
        + 0.5 * perf["co2_conversion"]
        - 0.001 * cost.get("usd_per_kg", 0.0)
    )

    if "microkinetic_lite" in history:
        score += 0.25
    if "dft_adsorption" in history:
        score += 0.5

    return score

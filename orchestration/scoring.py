def score_candidate(history: dict) -> float:
    fs = history.get("fast_surrogate")
    if not isinstance(fs, dict):
        return float("-inf")

    perf = fs.get("performance")
    if not isinstance(perf, dict):
        # fast_surrogate missing (timeout/failed/cancelled)
        return float("-inf")

    cost = fs.get("catalyst_cost", {}) if isinstance(fs.get("catalyst_cost"), dict) else {}
    usd_per_kg = float(cost.get("usd_per_kg", 0.0))

    score = (
        1.0 * float(perf.get("methanol_sty", 0.0))
        + 2.0 * float(perf.get("methanol_selectivity", 0.0))
        + 0.5 * float(perf.get("co2_conversion", 0.0))
        - 0.5 * float(perf.get("uncertainty", 0.0))
        - 0.001 * usd_per_kg
    )

    # Optional bonuses
    if "microkinetic_lite" in history:
        score += 0.25
    if "dft_adsorption" in history:
        score += 0.50

    return score

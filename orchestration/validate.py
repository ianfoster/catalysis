from orchestration.characterizers import CHARACTERIZERS

def validate_result(characterizer: str, result: dict) -> dict:
    expected = CHARACTERIZERS[characterizer].outputs
    missing = [k for k in expected if k not in result and k not in result.get("performance", {})]
    if missing:
        return {
            "status": "FAILED",
            "error": f"Missing outputs {missing}",
            "raw": result,
        }
    return result

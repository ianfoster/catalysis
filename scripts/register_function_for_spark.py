from globus_compute_sdk import Client

gcc = Client()

def score_smiles(smiles: str) -> dict:
    import requests
    r = requests.post(
        "http://localhost:8000/predict",
        json={"smiles": smiles},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

# Register function (do this once; reuse the ID)
function_id = gcc.register_function(score_smiles)
print("function_id:", function_id)


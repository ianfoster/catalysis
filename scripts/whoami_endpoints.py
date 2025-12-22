from globus_compute_sdk import Client
import json

gcc = Client()
eps = gcc.get_endpoints()
print("num_endpoints:", len(eps))

for e in eps:
    print(json.dumps(e, indent=2, default=str))


from globus_compute_sdk import Client

gcc = Client()

FUNCTION_ID = "4ba064ea-abdd-4ec3-a2d7-3d7331e517e2"
ENDPOINT_ID = "0213ed14-85d5-4a51-9a0f-5bc6b0e5f9d5"

task_id = gcc.run(
    endpoint_id=ENDPOINT_ID,
    function_id=FUNCTION_ID,
    arguments=("CCO",),
)

print("endpoint_id:", ENDPOINT_ID)
print("task_id:", task_id)
print("result:", gcc.get_result(task_id))


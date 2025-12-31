#!/usr/bin/env python3
"""Start vLLM server on Spark (DGX GB10).

Key requirement for GB10: --enforce-eager true flag!

Working models (from December 2025 benchmarks):
- Qwen/Qwen3-4B-FP8: 33.30 tok/s
- Qwen/Qwen3-14B-FP8: 12.73 tok/s
- Qwen/Qwen3-30B-A3B-FP8 (MoE): 37.23 tok/s
- Qwen/Qwen3-32B-FP8: 6.13 tok/s
- Meta-Llama-3.1-8B-Instruct-FP8-dynamic: 94.90 tok/s
- Meta-Llama-3.1-8B-Instruct-quantized.w4a16: 189.81 tok/s

Usage:
    # On Spark directly (recommended approach):
    docker run -it --gpus all -p 8000:8000 \\
        nvcr.io/nvidia/vllm:25.11-py3 \\
        vllm serve "meta-llama/Llama-3.1-8B-Instruct" \\
        --enforce-eager --trust-remote-code

    # Or use this script:
    python scripts/start_vllm_server.py --model meta-llama/Llama-3.1-8B-Instruct

    # Via Globus Compute (from Mac):
    python scripts/start_vllm_server.py --endpoint $GC_ENDPOINT \\
        --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import subprocess
import sys
import time


def start_vllm_server_gc(config: dict) -> dict:
    """Start vLLM server on Spark via Docker.

    This function is COMPLETELY SELF-CONTAINED for GC serialization.

    Args:
        config: Dict with:
            - model: HuggingFace model name/path
            - port: Server port (default: 8000)
            - host: Server host (default: 0.0.0.0)
            - container: Docker container image
            - enforce_eager: Use eager mode (required for GB10!)
            - trust_remote_code: Trust remote code
            - tensor_parallel_size: TP size (default: 1)
            - max_model_len: Max context length (optional)
            - quantization: Quantization method (optional)

    Returns:
        Dict with server status and connection info
    """
    import subprocess
    import time
    import logging
    import os
    import signal

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("vllm_server")

    model = config.get("model")
    if not model:
        return {"ok": False, "error": "model required"}

    port = config.get("port", 8000)
    host = config.get("host", "0.0.0.0")
    container = config.get("container", "nvcr.io/nvidia/vllm:25.11-py3")
    enforce_eager = config.get("enforce_eager", True)  # Required for GB10!
    trust_remote_code = config.get("trust_remote_code", True)
    tensor_parallel_size = config.get("tensor_parallel_size", 1)
    max_model_len = config.get("max_model_len")
    quantization = config.get("quantization")
    run_duration = config.get("run_duration", 0)

    result = {
        "model": model,
        "host": host,
        "port": port,
        "container": container,
        "server_started": False,
    }

    # Build vLLM serve command
    vllm_args = [
        "vllm", "serve", model,
        "--host", host,
        "--port", str(port),
    ]

    if enforce_eager:
        vllm_args.append("--enforce-eager")

    if trust_remote_code:
        vllm_args.append("--trust-remote-code")

    if tensor_parallel_size > 1:
        vllm_args.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

    if max_model_len:
        vllm_args.extend(["--max-model-len", str(max_model_len)])

    if quantization:
        vllm_args.extend(["--quantization", quantization])

    # Build docker command
    docker_cmd = [
        "docker", "run",
        "--gpus", "all",
        "-p", f"{port}:{port}",
        "--rm",  # Remove container when done
        "-d",    # Detach mode
        "--name", "vllm-server",
        container,
    ] + vllm_args

    logger.info(f"Starting vLLM server...")
    logger.info(f"  Model: {model}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Enforce eager: {enforce_eager}")
    logger.info(f"  Command: {' '.join(docker_cmd)}")

    # Check if container already running
    try:
        check = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=vllm-server"],
            capture_output=True, text=True
        )
        if check.stdout.strip():
            logger.info("Stopping existing vllm-server container...")
            subprocess.run(["docker", "stop", "vllm-server"], capture_output=True)
            time.sleep(2)
    except Exception as e:
        logger.warning(f"Error checking existing container: {e}")

    # Start container
    try:
        proc = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            result["error"] = f"Docker failed: {proc.stderr}"
            return result
        result["container_id"] = proc.stdout.strip()
        logger.info(f"Container started: {result['container_id'][:12]}")
    except subprocess.TimeoutExpired:
        result["error"] = "Docker start timed out"
        return result
    except Exception as e:
        result["error"] = f"Failed to start Docker: {e}"
        return result

    # Wait for server to be ready
    logger.info("Waiting for vLLM server to load model...")

    import urllib.request
    import urllib.error

    health_url = f"http://127.0.0.1:{port}/health"
    models_url = f"http://127.0.0.1:{port}/v1/models"
    start_time = time.time()
    timeout = config.get("startup_timeout", 300)  # 5 min for model loading

    while time.time() - start_time < timeout:
        # Check if container still running
        try:
            check = subprocess.run(
                ["docker", "ps", "-q", "-f", "name=vllm-server"],
                capture_output=True, text=True
            )
            if not check.stdout.strip():
                # Container died, get logs
                logs = subprocess.run(
                    ["docker", "logs", "vllm-server"],
                    capture_output=True, text=True
                )
                result["error"] = f"Container died: {logs.stderr[-1000:]}"
                return result
        except Exception:
            pass

        # Check health endpoint
        try:
            req = urllib.request.urlopen(health_url, timeout=5)
            if req.status == 200:
                result["server_started"] = True
                logger.info(f"Server healthy after {time.time() - start_time:.1f}s")
                break
        except urllib.error.URLError:
            pass
        except Exception:
            pass

        # Also check models endpoint (vLLM reports ready when model loaded)
        try:
            req = urllib.request.urlopen(models_url, timeout=5)
            if req.status == 200:
                import json as json_mod
                data = json_mod.loads(req.read().decode())
                if data.get("data"):
                    result["server_started"] = True
                    result["models"] = [m.get("id") for m in data.get("data", [])]
                    logger.info(f"Model loaded after {time.time() - start_time:.1f}s")
                    break
        except Exception:
            pass

        time.sleep(5)
        elapsed = time.time() - start_time
        logger.info(f"Still waiting... ({elapsed:.0f}s)")

    if not result["server_started"]:
        result["error"] = f"Server not ready after {timeout}s"
        # Get container logs for debugging
        try:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "50", "vllm-server"],
                capture_output=True, text=True
            )
            result["logs"] = logs.stdout + logs.stderr
        except Exception:
            pass
        return result

    # Test the server
    logger.info("Testing server with chat completion...")

    try:
        import json as json_mod
        test_url = f"http://127.0.0.1:{port}/v1/chat/completions"
        test_data = json_mod.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 20,
        }).encode()

        req = urllib.request.Request(
            test_url,
            data=test_data,
            headers={"Content-Type": "application/json"},
        )
        response = urllib.request.urlopen(req, timeout=120)
        response_data = json_mod.loads(response.read().decode())

        result["test_response"] = response_data["choices"][0]["message"]["content"]
        result["ok"] = True
        logger.info(f"Server test successful: {result['test_response'][:50]}...")

    except Exception as e:
        result["error"] = f"Server test failed: {e}"
        result["ok"] = False
        return result

    # Keep server running or return
    if run_duration > 0:
        logger.info(f"Server will run for {run_duration}s...")
        time.sleep(run_duration)
        logger.info("Stopping server...")
        subprocess.run(["docker", "stop", "vllm-server"], capture_output=True)
    else:
        result["note"] = "Server running in background. Use 'docker stop vllm-server' to stop."

    result["server_url"] = f"http://{host}:{port}/v1"
    return result


def get_spark_ip() -> str:
    """Get Spark's IP address for external access."""
    import socket
    try:
        # Get hostname
        hostname = socket.gethostname()
        # Get IP
        ip = socket.gethostbyname(hostname)
        return ip
    except Exception:
        return "localhost"


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM server on Spark (DGX GB10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended models for GB10 (with --enforce-eager):
  - meta-llama/Llama-3.1-8B-Instruct
  - neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic (faster)
  - Qwen/Qwen3-14B-FP8
  - Qwen/Qwen3-30B-A3B-FP8 (MoE, good speed)

Examples:
  # Start locally on Spark
  python scripts/start_vllm_server.py --model meta-llama/Llama-3.1-8B-Instruct

  # Via Globus Compute
  python scripts/start_vllm_server.py --endpoint $GC_ENDPOINT \\
      --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic

  # Direct docker command (alternative):
  docker run -it --gpus all -p 8000:8000 \\
      nvcr.io/nvidia/vllm:25.11-py3 \\
      vllm serve meta-llama/Llama-3.1-8B-Instruct \\
      --enforce-eager --trust-remote-code
        """,
    )

    parser.add_argument(
        "--endpoint",
        help="Globus Compute endpoint (for remote start)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--container",
        default="nvcr.io/nvidia/vllm:25.11-py3",
        help="Docker container image",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Disable --enforce-eager (NOT recommended for GB10)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Maximum context length",
    )
    parser.add_argument(
        "--quantization",
        help="Quantization method (e.g., fp8, awq)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--run-duration",
        type=int,
        default=0,
        help="How long to run (seconds). 0 = keep running in background",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=300,
        help="Timeout for model loading (default: 300s)",
    )
    parser.add_argument(
        "--output",
        default="vllm_server_info.json",
        help="Output file for server info",
    )

    args = parser.parse_args()

    config = {
        "model": args.model,
        "port": args.port,
        "container": args.container,
        "enforce_eager": not args.no_enforce_eager,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "quantization": args.quantization,
        "run_duration": args.run_duration,
        "startup_timeout": args.startup_timeout,
    }

    if args.endpoint:
        # Run via Globus Compute
        from globus_compute_sdk import Client, Executor

        print(f"Starting vLLM server on endpoint: {args.endpoint}")
        print(f"Model: {args.model}")
        print(f"Container: {args.container}")
        print(f"Enforce eager: {not args.no_enforce_eager}")

        client = Client()
        func_id = client.register_function(start_vllm_server_gc)
        print(f"Function ID: {func_id}")

        total_timeout = args.startup_timeout + 120 + args.run_duration

        with Executor(endpoint_id=args.endpoint) as ex:
            future = ex.submit_to_registered_function(func_id, args=(config,))
            try:
                result = future.result(timeout=total_timeout)
            except Exception as e:
                result = {"ok": False, "error": str(e)}
    else:
        # Run locally
        print(f"Starting vLLM server locally...")
        print(f"Model: {args.model}")
        print(f"Container: {args.container}")
        print(f"Enforce eager: {not args.no_enforce_eager}")
        result = start_vllm_server_gc(config)

    print(f"\nResult:")
    print(json.dumps(result, indent=2))

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")

    if result.get("ok"):
        print(f"\n{'='*60}")
        print("vLLM server is running!")
        print(f"Server URL: {result.get('server_url', f'http://localhost:{args.port}/v1')}")
        print(f"\nTo use with discovery:")
        print(f"  python scripts/run_discovery.py --endpoint $GC_ENDPOINT \\")
        print(f"      --llm-url http://<spark-ip>:{args.port}/v1")
        print(f"\nTo stop:")
        print(f"  docker stop vllm-server")
        print(f"{'='*60}")

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

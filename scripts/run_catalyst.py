#!/usr/bin/env python3
"""Run Catalyst Discovery - unified launcher for Mac.

This script handles the full workflow:
1. Starts agents on Spark via GC (if not already running)
2. Sets up SSH tunnel to Spark's Redis
3. Runs Generator locally, connecting to Spark agents

Usage:
    # Full workflow
    python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --spark-host spark

    # Just run generator (agents already running, tunnel already open)
    python scripts/run_catalyst.py --redis-port 6380 --skip-agents --skip-tunnel
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def clear_caches(endpoint: str | None = None) -> dict:
    """Clear local and remote caches."""
    result = {"ok": True, "cleared": {}}

    # Clear local cache files
    local_files = [
        "data/generator_state.json",
        "data/generator_results.jsonl",
        "data/shepherd_cache.jsonl",
        "data/char_cache.jsonl",
        "data/discovery_results.json",
    ]

    for f in local_files:
        path = Path(f)
        if path.exists():
            path.unlink()
            result["cleared"][f] = "deleted"
        else:
            result["cleared"][f] = "not found"

    # Clear remote Redis caches if endpoint provided
    if endpoint:
        from globus_compute_sdk import Client, Executor

        def clear_redis_cache_gc(cfg: dict) -> dict:
            """Clear Redis caches on Spark."""
            result = {"ok": False, "cleared": {}}
            try:
                import redis
                r = redis.Redis(host="localhost", port=6379)

                # Clear narrative logs
                narrative_keys = r.keys("narrative:*")
                if narrative_keys:
                    r.delete(*narrative_keys)
                result["cleared"]["narrative"] = len(narrative_keys)

                # Clear test_runtime keys
                runtime_keys = r.keys("test_runtime:*")
                if runtime_keys:
                    r.delete(*runtime_keys)
                result["cleared"]["test_runtime"] = len(runtime_keys)

                # Clear any result caches
                cache_keys = r.keys("cache:*")
                if cache_keys:
                    r.delete(*cache_keys)
                result["cleared"]["cache"] = len(cache_keys)

                result["ok"] = True
            except Exception as e:
                result["error"] = str(e)
            return result

        client = Client()
        func_id = client.register_function(clear_redis_cache_gc)

        with Executor(endpoint_id=endpoint) as ex:
            future = ex.submit_to_registered_function(func_id, args=({},))
            remote_result = future.result(timeout=60)
            result["remote"] = remote_result

    return result


def check_agents_on_spark(endpoint: str) -> dict:
    """Check status of agents on Spark via Globus Compute."""
    from globus_compute_sdk import Client, Executor

    def check_agents_gc(cfg: dict) -> dict:
        """Check Academy agents status on Spark."""
        import subprocess
        import urllib.request

        result = {"ok": False}

        # Check vLLM
        try:
            resp = urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
            import json
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            result["vllm"] = {"running": True, "models": models}
        except Exception as e:
            result["vllm"] = {"running": False, "error": str(e)}

        # Check Redis and agent registrations
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379)
            r.ping()
            result["redis"] = {"running": True}

            # Count agents by type (from agent:* keys) and status (from active:* keys)
            agent_types = {}
            active_count = 0
            total_count = 0

            for key in r.keys("agent:*"):
                try:
                    agent_id = key.decode().replace("agent:", "")
                    type_data = r.get(key)
                    status = r.get(f"active:{agent_id}")
                    is_active = status == b"ACTIVE"

                    if is_active:
                        active_count += 1

                    # Extract agent class name
                    if type_data:
                        # Format: "skills.shepherd.ShepherdAgent,skills.base_agent.TrackedAgent,..."
                        first_class = type_data.decode().split(",")[0]
                        class_name = first_class.split(".")[-1]  # Just the class name
                        key_name = f"{class_name} ({'active' if is_active else 'inactive'})"
                        agent_types[key_name] = agent_types.get(key_name, 0) + 1
                    total_count += 1
                except Exception:
                    pass

            result["agents"] = {
                "total": total_count,
                "active": active_count,
                "types": dict(sorted(agent_types.items())),
            }
        except Exception as e:
            result["redis"] = {"running": False, "error": str(e)}
            result["agents"] = {"count": 0}

        # Check if run_spark_agents.py is running
        try:
            proc = subprocess.run(
                ["pgrep", "-f", "run_spark_agents.py"],
                capture_output=True,
                text=True,
            )
            pids = proc.stdout.strip().split("\n") if proc.stdout.strip() else []
            result["agent_process"] = {"running": len(pids) > 0, "pids": pids}
        except Exception as e:
            result["agent_process"] = {"error": str(e)}

        result["ok"] = True
        return result

    logging.info("Checking agents on Spark via GC...")

    client = Client()
    func_id = client.register_function(check_agents_gc)

    with Executor(endpoint_id=endpoint) as ex:
        future = ex.submit_to_registered_function(func_id, args=({},))
        result = future.result(timeout=60)

    return result


def stop_agents_on_spark(endpoint: str) -> dict:
    """Stop running agents on Spark via Globus Compute."""
    from globus_compute_sdk import Client, Executor

    def stop_agents_gc(cfg: dict) -> dict:
        """Stop Academy agents on Spark."""
        import subprocess

        result = {"ok": False, "cleared": {}}

        # Kill any running agent processes
        try:
            # Kill run_spark_agents.py
            proc = subprocess.run(
                ["pkill", "-f", "run_spark_agents.py"],
                capture_output=True,
            )
            result["killed_process"] = proc.returncode == 0

            # Clear ALL agent-related Redis keys
            try:
                import redis
                r = redis.Redis(host="localhost", port=6379)

                # Clear active:* keys
                active_keys = r.keys("active:*")
                if active_keys:
                    r.delete(*active_keys)
                result["cleared"]["active"] = len(active_keys)

                # Clear agent:* keys (agent type registrations)
                agent_keys = r.keys("agent:*")
                if agent_keys:
                    r.delete(*agent_keys)
                result["cleared"]["agent"] = len(agent_keys)

                # Clear queue:* keys (message queues)
                queue_keys = r.keys("queue:*")
                if queue_keys:
                    r.delete(*queue_keys)
                result["cleared"]["queue"] = len(queue_keys)

            except Exception as e:
                result["redis_error"] = str(e)

            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    logging.info("Stopping agents on Spark via GC...")

    client = Client()
    func_id = client.register_function(stop_agents_gc)

    with Executor(endpoint_id=endpoint) as ex:
        future = ex.submit_to_registered_function(func_id, args=({},))
        result = future.result(timeout=60)

    return result


def update_code_on_spark(endpoint: str) -> dict:
    """Update code on Spark via git pull."""
    from globus_compute_sdk import Client, Executor

    def update_code_gc(cfg: dict) -> dict:
        """Update catalysis code on Spark."""
        import os
        import subprocess

        catalysis_dir = os.path.expanduser("~/catalysis")
        result = {"ok": False, "dir": catalysis_dir}

        try:
            # Git pull
            proc = subprocess.run(
                ["git", "pull"],
                cwd=catalysis_dir,
                capture_output=True,
                text=True,
            )
            result["git_pull"] = proc.stdout.strip()
            result["git_stderr"] = proc.stderr.strip() if proc.stderr else None

            # Reinstall package
            proc = subprocess.run(
                ["pip", "install", "-e", "."],
                cwd=catalysis_dir,
                capture_output=True,
                text=True,
            )
            result["pip_install"] = "success" if proc.returncode == 0 else proc.stderr[:200]

            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    logging.info("Updating code on Spark via GC...")

    client = Client()
    func_id = client.register_function(update_code_gc)

    with Executor(endpoint_id=endpoint) as ex:
        future = ex.submit_to_registered_function(func_id, args=({},))
        result = future.result(timeout=120)

    return result


def start_agents_on_spark(endpoint: str, config: dict) -> dict:
    """Start agents on Spark via Globus Compute.

    Returns dict with status and agent info.
    """
    from globus_compute_sdk import Client, Executor

    # Self-contained GC function
    def start_agents_gc(cfg: dict) -> dict:
        """Start Academy agents on Spark."""
        import asyncio
        import logging
        import subprocess
        import time
        import urllib.request

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("start_agents")

        llm_model = cfg.get("llm_model", "meta-llama/Llama-3.1-8B-Instruct")
        llm_port = cfg.get("llm_port", 8000)
        num_shepherds = cfg.get("num_shepherds", 4)
        device = cfg.get("device", "cuda")

        result = {"ok": False, "llm_port": llm_port}

        # Check/start vLLM
        logger.info("Checking vLLM server...")
        try:
            urllib.request.urlopen(f"http://localhost:{llm_port}/v1/models", timeout=5)
            logger.info("vLLM already running")
            result["vllm"] = "already_running"
        except Exception:
            logger.info("Starting vLLM...")
            try:
                # Remove existing container if any
                subprocess.run(["docker", "rm", "-f", "vllm-server"],
                             capture_output=True)
                # Start new container
                cmd = [
                    "docker", "run", "--gpus", "all",
                    "-p", f"{llm_port}:{llm_port}",
                    "--rm", "-d", "--name", "vllm-server",
                    "nvcr.io/nvidia/vllm:25.11-py3",
                    "vllm", "serve", llm_model,
                    "--host", "0.0.0.0", "--port", str(llm_port),
                    "--enforce-eager", "--trust-remote-code",
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # Wait for ready
                for i in range(60):
                    time.sleep(5)
                    try:
                        urllib.request.urlopen(f"http://localhost:{llm_port}/v1/models", timeout=5)
                        logger.info("vLLM ready")
                        result["vllm"] = "started"
                        break
                    except Exception:
                        pass
                else:
                    result["error"] = "vLLM failed to start"
                    return result
            except Exception as e:
                result["error"] = f"vLLM error: {e}"
                return result

        # Check Redis
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379)
            r.ping()
            result["redis"] = "running"
        except Exception as e:
            result["error"] = f"Redis not running: {e}"
            return result

        # Start agents in background using nohup
        logger.info("Starting Academy agents...")
        try:
            import os
            r = redis.Redis(host="localhost", port=6379)

            # Check if process already running
            proc_check = subprocess.run(
                ["pgrep", "-f", "run_spark_agents.py"],
                capture_output=True, text=True
            )
            if proc_check.returncode == 0:
                pids = proc_check.stdout.strip()
                keys = r.keys("active:*")
                active = sum(1 for k in keys if r.get(k) == b"ACTIVE")
                result["agents"] = "already_running"
                result["pids"] = pids
                result["agent_count"] = active
                result["ok"] = True
                return result

            # Start agents via subprocess (detached)
            catalysis_dir = os.path.expanduser("~/catalysis")
            log_file = "/tmp/spark_agents.log"

            # Clear old log
            with open(log_file, "w") as f:
                f.write(f"Starting agents at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            cmd = f"""cd {catalysis_dir} && nohup python scripts/run_spark_agents.py \
                --llm-url http://localhost:{llm_port}/v1 \
                --redis-host localhost \
                --num-shepherds {num_shepherds} \
                --device {device} \
                >> {log_file} 2>&1 &"""

            logger.info(f"Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

            # Wait for agents to register
            for i in range(30):
                time.sleep(2)

                # Check if process started
                proc_check = subprocess.run(
                    ["pgrep", "-f", "run_spark_agents.py"],
                    capture_output=True, text=True
                )
                if proc_check.returncode != 0:
                    # Process died - read log
                    try:
                        with open(log_file) as f:
                            result["log_tail"] = f.read()[-500:]
                    except:
                        pass
                    result["error"] = "Agent process died"
                    return result

                # Check for active agents
                keys = r.keys("active:*")
                active = sum(1 for k in keys if r.get(k) == b"ACTIVE")
                if active >= num_shepherds:
                    result["agents"] = "started"
                    result["agent_count"] = active
                    result["ok"] = True
                    return result

            # Timed out but process running
            keys = r.keys("active:*")
            active = sum(1 for k in keys if r.get(k) == b"ACTIVE")
            result["agents"] = "starting"
            result["agent_count"] = active
            result["ok"] = True

        except Exception as e:
            result["error"] = f"Agent startup error: {e}"

        return result

    logging.info("Starting agents on Spark via GC...")

    client = Client()
    func_id = client.register_function(start_agents_gc)

    with Executor(endpoint_id=endpoint) as ex:
        future = ex.submit_to_registered_function(func_id, args=(config,))
        result = future.result(timeout=600)  # 10 min timeout

    return result


def setup_ssh_tunnel(spark_host: str, local_port: int = 6380, remote_port: int = 6379) -> subprocess.Popen:
    """Set up SSH tunnel to Spark's Redis.

    Returns the subprocess handle (keep alive to maintain tunnel).
    """
    logging.info(f"Setting up SSH tunnel: localhost:{local_port} -> {spark_host}:{remote_port}")

    # Start SSH tunnel in foreground-ish mode (will prompt for password)
    proc = subprocess.Popen(
        ["ssh", "-N", "-L", f"{local_port}:localhost:{remote_port}", spark_host],
        stdin=None,  # Allow password prompt
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for tunnel to establish (longer wait after password prompt)
    logging.info("Waiting for tunnel to establish...")

    # Try connecting to Redis through tunnel with retries
    import redis
    for attempt in range(15):  # Try for 30 seconds
        time.sleep(2)

        # Check if SSH process died
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode()
            raise RuntimeError(f"SSH tunnel failed: {stderr}")

        try:
            r = redis.Redis(host="localhost", port=local_port, socket_timeout=3)
            r.ping()
            logging.info("SSH tunnel established and Redis accessible")
            return proc
        except redis.ConnectionError:
            logging.debug(f"Tunnel not ready yet (attempt {attempt + 1}/15)...")
            continue
        except Exception as e:
            logging.debug(f"Connection attempt {attempt + 1} failed: {e}")
            continue

    # If we get here, tunnel didn't work
    proc.terminate()
    raise RuntimeError("Could not connect to Redis through tunnel after 30 seconds. Is Redis running on Spark?")


async def run_generator_with_academy(
    redis_port: int,
    generator_llm: str,
    generator_model: str | None,
    max_iterations: int,
    candidates_per_iteration: int,
    budget: float,
) -> dict:
    """Run GeneratorAgent connecting to Spark agents via Redis."""
    from academy.manager import Manager
    from academy.exchange import RedisExchangeFactory
    from skills.generator import GeneratorAgent

    # Build config
    config = {
        "llm": {},
        "generation": {
            "max_iterations": max_iterations,
            "candidates_per_iteration": candidates_per_iteration,
        },
        "convergence": {
            "patience": 3,
            "min_improvement": 0.01,
            "llm_judgment": True,
        },
        "shepherd": {
            "budget_per_candidate": budget,
            "num_concurrent": 8,
            "timeout": 3600,
        },
        "state": {
            "checkpoint_path": "data/generator_state.json",
            "results_path": "data/generator_results.jsonl",
        },
    }

    # Setup Generator LLM
    if generator_llm == "openai":
        config["llm"]["base_url"] = "https://api.openai.com/v1"
        config["llm"]["model"] = generator_model or "gpt-4o"
        config["llm"]["api_key_env"] = "OPENAI_API_KEY"
    else:
        config["llm"]["base_url"] = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
        config["llm"]["model"] = generator_model or "meta-llama/Meta-Llama-3.1-70B-Instruct"
        config["llm"]["api_key_env"] = "ARGONNE_ACCESS_TOKEN"

    logging.info("Connecting to Academy exchange...")
    exchange_factory = RedisExchangeFactory(hostname="localhost", port=redis_port)

    async with await Manager.from_exchange_factory(exchange_factory) as manager:
        logging.info("Launching GeneratorAgent...")
        generator = await manager.launch(
            GeneratorAgent,
            kwargs={
                "config": config,
                "shepherd_config": {},
                "redis_host": "localhost",
                "redis_port": redis_port,
            },
        )

        logging.info("Starting discovery loop...")
        results = await generator.run_discovery({})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Catalyst Discovery (unified launcher)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full workflow (start agents, tunnel, run generator)
    python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --spark-host spark

    # Restart agents with updated code
    python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --spark-host spark --restart-agents

    # Just update code on Spark (no restart)
    python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --update-code

    # Skip agent startup (already running)
    python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --spark-host spark --skip-agents

    # Skip tunnel (already have one on port 6380)
    python scripts/run_catalyst.py --redis-port 6380 --skip-agents --skip-tunnel

    # Use Argonne instead of OpenAI
    python scripts/run_catalyst.py --endpoint $GC_ENDPOINT --spark-host spark --generator-llm argonne
        """,
    )

    # Connection
    parser.add_argument("--endpoint", help="Globus Compute endpoint ID on Spark")
    parser.add_argument("--spark-host", default="spark", help="Spark hostname for SSH (default: spark)")
    parser.add_argument("--redis-port", type=int, default=6380, help="Local Redis port (tunnel target, default: 6380)")

    # Agent management
    parser.add_argument("--check-agents", action="store_true", help="Check agent status on Spark and exit")
    parser.add_argument("--stop-agents", action="store_true", help="Stop agents on Spark and exit")
    parser.add_argument("--update-code", action="store_true", help="Update code on Spark via git pull")
    parser.add_argument("--restart-agents", action="store_true", help="Stop agents, update code, restart agents")
    parser.add_argument("--start-agents-only", action="store_true", help="Start agents on Spark and exit (no generator)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear local caches and Spark Redis caches")

    # Skip options
    parser.add_argument("--skip-agents", action="store_true", help="Skip starting agents (already running)")
    parser.add_argument("--skip-tunnel", action="store_true", help="Skip SSH tunnel (already open)")

    # Agent config
    parser.add_argument("--llm-model", default="meta-llama/Llama-3.1-8B-Instruct", help="LLM model on Spark")
    parser.add_argument("--num-shepherds", type=int, default=4, help="Number of ShepherdAgents")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device for ML agents")

    # Generator config
    parser.add_argument("--generator-llm", default="openai", choices=["openai", "argonne"], help="Generator LLM")
    parser.add_argument("--generator-model", help="Generator model name")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max iterations")
    parser.add_argument("--candidates-per-iteration", type=int, default=6, help="Candidates per iteration")
    parser.add_argument("--budget", type=float, default=100.0, help="Budget per candidate")

    parser.add_argument("--output", default="data/discovery_results.json", help="Output file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate args
    needs_endpoint = (
        args.check_agents or args.stop_agents or args.update_code or
        args.restart_agents or args.start_agents_only or not args.skip_agents
    )
    if needs_endpoint and not args.endpoint:
        parser.error("--endpoint required for agent management")
    if not args.skip_tunnel and not args.spark_host and not (args.check_agents or args.stop_agents or args.update_code or args.start_agents_only):
        parser.error("--spark-host required unless --skip-tunnel")

    # Handle agent management commands (these exit early)
    if args.check_agents:
        result = check_agents_on_spark(args.endpoint)
        print("\n" + "=" * 60)
        print("Agent Status on Spark")
        print("=" * 60)
        vllm = result.get("vllm", {})
        print(f"vLLM: {'running' if vllm.get('running') else 'NOT running'}")
        if vllm.get("models"):
            print(f"  Models: {', '.join(vllm['models'])}")
        redis_info = result.get("redis", {})
        print(f"Redis: {'running' if redis_info.get('running') else 'NOT running'}")
        agents = result.get("agents", {})
        print(f"Agents: {agents.get('total', 0)} total, {agents.get('active', 0)} active")
        if agents.get("types"):
            for agent_type, count in agents["types"].items():
                print(f"  - {agent_type}: {count}")
        proc = result.get("agent_process", {})
        print(f"Agent process: {'running' if proc.get('running') else 'NOT running'}")
        print("=" * 60)
        sys.exit(0)

    if args.clear_cache:
        logging.info("Clearing caches...")
        result = clear_caches(args.endpoint)
        print("\n" + "=" * 60)
        print("Cache Cleared")
        print("=" * 60)
        print("Local files:")
        for f, status in result.get("cleared", {}).items():
            print(f"  - {f}: {status}")
        if result.get("remote"):
            print("Remote Redis:")
            for k, v in result["remote"].get("cleared", {}).items():
                print(f"  - {k}: {v} keys")
        print("=" * 60)
        sys.exit(0)

    if args.stop_agents:
        result = stop_agents_on_spark(args.endpoint)
        logging.info(f"Stop agents result: {result}")
        sys.exit(0 if result.get("ok") else 1)

    if args.update_code:
        result = update_code_on_spark(args.endpoint)
        logging.info(f"Update code result: {result}")
        if not args.restart_agents:
            sys.exit(0 if result.get("ok") else 1)

    if args.restart_agents:
        # Stop existing agents
        logging.info("=" * 60)
        logging.info("Stopping existing agents...")
        logging.info("=" * 60)
        stop_result = stop_agents_on_spark(args.endpoint)
        logging.info(f"Stop result: {stop_result}")

        # Update code
        logging.info("=" * 60)
        logging.info("Updating code on Spark...")
        logging.info("=" * 60)
        update_result = update_code_on_spark(args.endpoint)
        logging.info(f"Update result: git_pull={update_result.get('git_pull')}, pip={update_result.get('pip_install')}")

        if not update_result.get("ok"):
            logging.error(f"Code update failed: {update_result.get('error')}")
            sys.exit(1)

        # Continue to start agents (don't skip)
        args.skip_agents = False

    # Check API key
    if args.generator_llm == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY not set")
            sys.exit(1)
    else:
        if not os.environ.get("ARGONNE_ACCESS_TOKEN"):
            logging.warning("ARGONNE_ACCESS_TOKEN not set - run scripts/argonne_auth.py")

    tunnel_proc = None

    try:
        # Step 1: Start agents on Spark
        if not args.skip_agents:
            logging.info("=" * 60)
            logging.info("Step 1: Starting agents on Spark")
            logging.info("=" * 60)

            agent_config = {
                "llm_model": args.llm_model,
                "llm_port": 8000,
                "num_shepherds": args.num_shepherds,
                "device": args.device,
            }

            result = start_agents_on_spark(args.endpoint, agent_config)

            if not result.get("ok"):
                logging.error(f"Failed to start agents: {result.get('error')}")
                if result.get("log_tail"):
                    logging.error(f"Log output:\n{result.get('log_tail')}")
                sys.exit(1)

            logging.info(f"Agents status: vLLM={result.get('vllm')}, agents={result.get('agents')}, count={result.get('agent_count')}")
            if result.get("pids"):
                logging.info(f"Agent PIDs: {result.get('pids')}")

            if args.start_agents_only:
                logging.info("Agents started. Exiting (--start-agents-only).")
                sys.exit(0)

        # Step 2: Setup SSH tunnel
        if not args.skip_tunnel:
            logging.info("=" * 60)
            logging.info("Step 2: Setting up SSH tunnel")
            logging.info("=" * 60)

            tunnel_proc = setup_ssh_tunnel(args.spark_host, args.redis_port)

        # Step 3: Run Generator
        logging.info("=" * 60)
        logging.info("Step 3: Running Generator")
        logging.info("=" * 60)
        logging.info(f"Generator LLM: {args.generator_llm}")
        logging.info(f"Redis port: {args.redis_port}")
        logging.info(f"Max iterations: {args.max_iterations}")
        logging.info(f"Candidates/iteration: {args.candidates_per_iteration}")

        results = asyncio.run(
            run_generator_with_academy(
                redis_port=args.redis_port,
                generator_llm=args.generator_llm,
                generator_model=args.generator_model,
                max_iterations=args.max_iterations,
                candidates_per_iteration=args.candidates_per_iteration,
                budget=args.budget,
            )
        )

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Results saved to {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("Discovery Summary")
        print("=" * 60)
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Iterations: {results.get('iteration', 0)}")
            print(f"Evaluated: {results.get('total_evaluated', 0)}")
            print(f"Best score: {results.get('best_score', 0):.1f}")
            if results.get("best_candidate"):
                bc = results["best_candidate"]
                if isinstance(bc, str):
                    print(f"Best: {bc}")
                else:
                    metals = "+".join(f"{m['element']}{m['wt_pct']}%" for m in bc.get("metals", [])[:3])
                    print(f"Best: {metals}/{bc.get('support', '?')}")
        print("=" * 60)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        # Cleanup tunnel
        if tunnel_proc:
            logging.info("Closing SSH tunnel...")
            tunnel_proc.terminate()
            tunnel_proc.wait()


if __name__ == "__main__":
    main()

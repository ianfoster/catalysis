#!/usr/bin/env python3
"""Setup Globus Compute endpoint on Polaris.

Run this script ON POLARIS to configure and register the endpoint.

Usage:
    # SSH to Polaris first
    ssh <username>@polaris.alcf.anl.gov

    # Then run this script
    python setup_endpoint.py --account <allocation> --project <project_name>

    # The script will output the endpoint ID to add to config.yaml
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ENDPOINT_NAME = "polaris-catalyst"

CONFIG_TEMPLATE = '''
display_name: Polaris Catalyst DFT Endpoint

engine:
  type: HighThroughputEngine

  provider:
    type: PBSProProvider
    queue: {queue}
    account: {account}
    select_options: "ngpus=4"
    nodes_per_block: 1
    walltime: "{walltime}"
    init_blocks: 0
    min_blocks: 0
    max_blocks: {max_blocks}
    scheduler_options: |
      #PBS -l filesystems=home:eagle:grand
      #PBS -l place=scatter
    worker_init: |
      module use /soft/modulefiles
      module load conda
      conda activate {conda_env}
      module load cudatoolkit-standalone/12.4.1
      module load quantum-espresso/7.3
      export TMPDIR=/local/scratch
      export CUDA_VISIBLE_DEVICES=0,1,2,3

  max_workers_per_node: 4
  address:
    type: address_by_interface
    ifname: bond0

heartbeat_period: 30
idle_heartbeats_soft: 5
idle_heartbeats_hard: 10

working_dir: /eagle/projects/{project}/catalyst_workdir

log_level: INFO
'''


def check_environment():
    """Verify we're running on Polaris."""
    hostname = os.uname().nodename
    if not hostname.startswith(("polaris", "x3")):
        print(f"Warning: This script should be run on Polaris (current host: {hostname})")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            sys.exit(1)


def check_globus_compute():
    """Check if globus-compute-endpoint is installed."""
    try:
        result = subprocess.run(
            ["globus-compute-endpoint", "version"],
            capture_output=True,
            text=True,
        )
        print(f"Globus Compute Endpoint version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("ERROR: globus-compute-endpoint not found.")
        print("Install with: pip install globus-compute-endpoint")
        return False


def configure_endpoint(args):
    """Create endpoint configuration."""
    config_dir = Path.home() / ".globus_compute" / ENDPOINT_NAME
    config_dir.mkdir(parents=True, exist_ok=True)

    config_content = CONFIG_TEMPLATE.format(
        queue=args.queue,
        account=args.account,
        walltime=args.walltime,
        max_blocks=args.max_blocks,
        conda_env=args.conda_env,
        project=args.project,
    )

    config_path = config_dir / "config.yaml"
    config_path.write_text(config_content)
    print(f"Wrote configuration to: {config_path}")

    return config_dir


def start_endpoint():
    """Start the endpoint and get its ID."""
    print(f"\nStarting endpoint '{ENDPOINT_NAME}'...")

    # First configure if not exists
    subprocess.run(
        ["globus-compute-endpoint", "configure", ENDPOINT_NAME],
        capture_output=True,
    )

    # Start the endpoint
    result = subprocess.run(
        ["globus-compute-endpoint", "start", ENDPOINT_NAME],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error starting endpoint: {result.stderr}")
        return None

    # Get endpoint ID
    result = subprocess.run(
        ["globus-compute-endpoint", "list"],
        capture_output=True,
        text=True,
    )

    # Parse output to find endpoint ID
    for line in result.stdout.split("\n"):
        if ENDPOINT_NAME in line:
            parts = line.split()
            for part in parts:
                # UUID format
                if len(part) == 36 and part.count("-") == 4:
                    return part

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Setup Globus Compute endpoint on Polaris for Catalyst DFT jobs"
    )
    parser.add_argument(
        "--account", "-a",
        required=True,
        help="ALCF allocation/project name for job charging",
    )
    parser.add_argument(
        "--project", "-p",
        required=True,
        help="Project directory name on /eagle/projects/",
    )
    parser.add_argument(
        "--queue", "-q",
        default="debug",
        choices=["debug", "debug-scaling", "prod", "preemptable"],
        help="PBS queue to use (default: debug)",
    )
    parser.add_argument(
        "--walltime", "-w",
        default="01:00:00",
        help="Job walltime (default: 01:00:00)",
    )
    parser.add_argument(
        "--max-blocks", "-m",
        type=int,
        default=2,
        help="Maximum number of nodes (default: 2)",
    )
    parser.add_argument(
        "--conda-env", "-e",
        default="catalyst",
        help="Conda environment name (default: catalyst)",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the endpoint after configuration",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Polaris Globus Compute Endpoint Setup")
    print("=" * 60)

    check_environment()

    if not check_globus_compute():
        sys.exit(1)

    config_dir = configure_endpoint(args)
    print(f"\nEndpoint configured at: {config_dir}")

    if args.start:
        endpoint_id = start_endpoint()
        if endpoint_id:
            print("\n" + "=" * 60)
            print("SUCCESS! Endpoint is running.")
            print("=" * 60)
            print(f"\nEndpoint ID: {endpoint_id}")
            print("\nAdd this to your config.yaml:")
            print(f"""
shepherd:
  gc_endpoints:
    polaris:
      id: "{endpoint_id}"
      capabilities: ["qe", "gpaw", "mace", "chgnet"]
      priority: 2
""")
        else:
            print("\nCould not determine endpoint ID. Check with:")
            print(f"  globus-compute-endpoint list")
    else:
        print("\nTo start the endpoint, run:")
        print(f"  globus-compute-endpoint start {ENDPOINT_NAME}")
        print("\nOr re-run this script with --start")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Example: Run catalyst evaluation with remote vLLM on Spark.

This example shows the full workflow:
1. Start vLLM on Spark via Globus Compute
2. Set up SSH tunnel to access vLLM
3. Run ShepherdAgent with remote LLM connection

Prerequisites:
    - Globus Compute endpoint running on Spark
    - vLLM and dependencies installed on Spark
    - SSH access to Spark for tunneling

Usage:
    # Step 1: Start vLLM on Spark (in one terminal)
    python scripts/start_llm_agent.py --endpoint $GC_ENDPOINT --run-duration 3600

    # Step 2: Set up SSH tunnel (in another terminal)
    ssh -L 8000:localhost:8000 spark.alcf.anl.gov

    # Step 3: Run this example (in another terminal)
    python scripts/example_remote_llm.py

Or for testing without Spark, use a local vLLM:
    # Start local vLLM
    python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct

    # Run example
    python scripts/example_remote_llm.py --llm-url http://localhost:8000/v1
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def run_evaluation(
    llm_url: str,
    llm_model: str,
    candidate: dict,
    gc_endpoint: str | None = None,
):
    """Run candidate evaluation with remote LLM.

    Args:
        llm_url: URL to vLLM server (e.g., http://localhost:8000/v1)
        llm_model: Model name on vLLM server
        candidate: Catalyst candidate to evaluate
        gc_endpoint: Globus Compute endpoint for simulations (optional)
    """
    import yaml

    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    shepherd_config = config.get("shepherd", {})

    # Override endpoint if provided
    if gc_endpoint:
        shepherd_config["endpoints"] = {
            "cheap": gc_endpoint,
            "gpu": gc_endpoint,
        }

    # Load GC function map
    gc_functions_path = PROJECT_ROOT / "gc_functions.json"
    if gc_functions_path.exists():
        with open(gc_functions_path) as f:
            gc_data = json.load(f)
            gc_function_map = gc_data.get("functions", {})
    else:
        gc_function_map = config.get("globus_compute", {}).get("functions", {})

    print(f"Loaded {len(gc_function_map)} GC functions")

    # Create ShepherdAgent with remote LLM
    from skills.shepherd import ShepherdAgent

    shepherd = ShepherdAgent(
        config=shepherd_config,
        gc_function_map=gc_function_map,
        llm_url=llm_url,
        llm_model=llm_model,
    )

    # Start the agent
    print(f"\nStarting ShepherdAgent with remote LLM at {llm_url}")
    await shepherd.agent_on_startup()

    # Run evaluation
    print(f"\nEvaluating candidate: {json.dumps(candidate, indent=2)}")

    result = await shepherd.evaluate({
        "candidate": candidate,
        "budget": 100.0,
        "goal": "CO2-to-methanol conversion",
    })

    return result


async def test_llm_connection(llm_url: str, llm_model: str):
    """Test connection to remote vLLM."""
    from openai import AsyncOpenAI

    print(f"Testing connection to {llm_url}...")

    client = AsyncOpenAI(
        base_url=llm_url,
        api_key="not-needed",
    )

    try:
        # List models
        models = await client.models.list()
        print(f"Available models: {[m.id for m in models.data]}")

        # Test completion
        response = await client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
        )
        print(f"Test response: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"Connection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Example: Run evaluation with remote vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--llm-url",
        default="http://localhost:8000/v1",
        help="URL to vLLM server (default: localhost:8000 for SSH tunnel)",
    )
    parser.add_argument(
        "--llm-model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name on vLLM server",
    )
    parser.add_argument(
        "--endpoint",
        help="Globus Compute endpoint for simulations",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test LLM connection, don't run evaluation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Test LLM connection first
    if not asyncio.run(test_llm_connection(args.llm_url, args.llm_model)):
        print("\nFailed to connect to vLLM. Make sure:")
        print("1. vLLM is running on Spark (use start_llm_agent.py)")
        print("2. SSH tunnel is set up: ssh -L 8000:localhost:8000 spark")
        print("3. Or specify --llm-url with the correct address")
        return 1

    if args.test_only:
        print("\nLLM connection test successful!")
        return 0

    # Example candidate
    candidate = {
        "support": "ZrO2",
        "metals": [
            {"element": "Cu", "wt_pct": 55},
            {"element": "Zn", "wt_pct": 30},
            {"element": "Al", "wt_pct": 15},
        ],
    }

    # Run evaluation
    try:
        result = asyncio.run(run_evaluation(
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            candidate=candidate,
            gc_endpoint=args.endpoint,
        ))

        print("\n" + "=" * 60)
        print("EVALUATION RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))

        # Summary
        assessment = result.get("final_assessment", {})
        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)
        print(f"Viability Score: {assessment.get('viability_score', 'N/A')}/100")
        print(f"Recommendation: {assessment.get('recommendation', 'N/A')}")
        print(f"Total Cost: {result.get('total_cost', 0):.2f}")
        print(f"Tests Run: {len(result.get('results', []))}")

        if assessment.get("strengths"):
            print(f"\nStrengths:")
            for s in assessment["strengths"]:
                print(f"  - {s}")

        if assessment.get("concerns"):
            print(f"\nConcerns:")
            for c in assessment["concerns"]:
                print(f"  - {c}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

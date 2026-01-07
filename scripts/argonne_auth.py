#!/usr/bin/env python3
"""Argonne ALCF Inference Service Authentication.

This wraps the ALCF inference_auth_token.py script for easy integration.

Setup (one-time):
    # Download ALCF auth script
    wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py

    # Authenticate (opens browser for Globus login)
    python inference_auth_token.py authenticate

    # Test
    python scripts/argonne_auth.py --test

Usage in code:
    from scripts.argonne_auth import get_alcf_token
    token = get_alcf_token()

    # Or set as env var
    export ARGONNE_ACCESS_TOKEN=$(python scripts/argonne_auth.py)

Usage with OpenAI client:
    from openai import OpenAI
    from scripts.argonne_auth import get_alcf_client

    client = get_alcf_client()
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[{"role": "user", "content": "Hello"}]
    )
"""

import os
import sys
import subprocess
from pathlib import Path

# ALCF endpoints
SOPHIA_URL = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
METIS_URL = "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"

# Default to Sophia (vLLM, supports Llama models)
DEFAULT_URL = SOPHIA_URL

# Path to ALCF auth script (download if needed)
AUTH_SCRIPT = Path(__file__).parent / "inference_auth_token.py"
AUTH_SCRIPT_URL = "https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py"


def ensure_auth_script():
    """Download ALCF auth script if not present."""
    if not AUTH_SCRIPT.exists():
        print(f"Downloading ALCF auth script to {AUTH_SCRIPT}...")
        import urllib.request
        urllib.request.urlretrieve(AUTH_SCRIPT_URL, AUTH_SCRIPT)
        print("Done. Run 'python scripts/inference_auth_token.py authenticate' to login.")


def get_alcf_token() -> str:
    """Get current ALCF access token.

    Returns:
        Access token string

    Raises:
        RuntimeError if token cannot be obtained
    """
    ensure_auth_script()

    # Try to get token from the auth script
    try:
        result = subprocess.run(
            [sys.executable, str(AUTH_SCRIPT), "get_access_token"],
            capture_output=True,
            text=True,
            cwd=AUTH_SCRIPT.parent,
        )

        if result.returncode == 0:
            token = result.stdout.strip()
            if token:
                return token

        # Check if we need to authenticate
        if "authenticate" in result.stderr.lower() or not result.stdout.strip():
            raise RuntimeError(
                "Not authenticated with ALCF. Run:\n"
                f"  python {AUTH_SCRIPT} authenticate"
            )

        raise RuntimeError(f"Failed to get token: {result.stderr}")

    except FileNotFoundError:
        raise RuntimeError(
            f"Auth script not found at {AUTH_SCRIPT}. Run:\n"
            f"  wget {AUTH_SCRIPT_URL} -O {AUTH_SCRIPT}"
        )


def get_alcf_client(base_url: str = DEFAULT_URL):
    """Get OpenAI client configured for ALCF inference.

    Args:
        base_url: ALCF inference endpoint URL

    Returns:
        OpenAI client instance
    """
    from openai import OpenAI

    token = get_alcf_token()
    return OpenAI(
        api_key=token,
        base_url=base_url,
    )


def authenticate():
    """Run interactive authentication."""
    ensure_auth_script()
    subprocess.run(
        [sys.executable, str(AUTH_SCRIPT), "authenticate"],
        cwd=AUTH_SCRIPT.parent,
    )


def list_models():
    """List available models on ALCF inference service."""
    import urllib.request
    import json

    token = get_alcf_token()
    url = "https://inference-api.alcf.anl.gov/resource_server/list-endpoints"

    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        return {"error": str(e)}


def test_connection():
    """Test the ALCF inference connection."""
    print("Testing ALCF inference connection...")
    print(f"  Endpoint: {DEFAULT_URL}")

    try:
        token = get_alcf_token()
        print(f"  Token: {token[:20]}...")

        client = get_alcf_client()

        # Skip model listing - not all endpoints support it
        # Go straight to testing completion
        print("  Testing chat completion with Llama-3.1-8B...")
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Say 'ALCF test successful' in exactly 4 words."}],
            max_tokens=20,
        )
        print(f"  Response: {response.choices[0].message.content}")

        # Also test 70B since that's what we'll use
        print("  Testing chat completion with Llama-3.1-70B...")
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=10,
        )
        print(f"  Response: {response.choices[0].message.content}")

        print("\nALCF inference connection successful!")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ALCF Inference Authentication")
    parser.add_argument("--test", action="store_true", help="Test the connection")
    parser.add_argument("--authenticate", action="store_true", help="Run authentication")
    parser.add_argument("--models", action="store_true", help="List available models")
    parser.add_argument("--url", default=DEFAULT_URL, help="Inference endpoint URL")

    args = parser.parse_args()

    if args.authenticate:
        authenticate()
    elif args.test:
        test_connection()
    elif args.models:
        import json
        data = list_models()
        print(json.dumps(data, indent=2))
    else:
        # Just print the token (for use with export)
        try:
            print(get_alcf_token())
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

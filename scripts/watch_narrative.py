#!/usr/bin/env python3
"""Watch the distributed narrative via Redis pub/sub.

Run this to see all narrative events from both Mac and Spark in real-time.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Watch catalyst discovery narrative")
    parser.add_argument("--redis-host", default="spark", help="Redis host (default: spark)")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port (default: 6379)")
    args = parser.parse_args()

    from orchestration.narrative import subscribe_narrative

    try:
        subscribe_narrative(args.redis_host, args.redis_port)
    except KeyboardInterrupt:
        print("\nStopped watching narrative.")
        sys.exit(0)


if __name__ == "__main__":
    main()

# src/agent/__main__.py
import asyncio
import sys

from agent.cli import parse_args, run_interactive, run_autonomous


def main():
    args = parse_args()
    if args.auto:
        exit_code = asyncio.run(run_autonomous(args))
        sys.exit(exit_code)
    else:
        asyncio.run(run_interactive(args))


if __name__ == "__main__":
    main()

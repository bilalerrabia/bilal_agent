"""Convenience shim: `uv run main.py` works like `uv run bilal`."""
from bilal_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

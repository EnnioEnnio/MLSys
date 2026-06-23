"""Enables `python -m mlsys <subcommand>`."""

from src.mlsys.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())

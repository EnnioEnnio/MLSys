.PHONY: setup lint format typecheck test check clean

setup:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run ty check

test:
	uv run pytest -q

check: lint typecheck test
	uv run ruff format --check .

clean:
	rm -rf .venv .ruff_cache .pytest_cache __pycache__ src/mlsys.egg-info

.PHONY: setup lint format typecheck test test-integration check clean

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

# Real-model end-to-end tests (network + heavier deps); excluded from `test`/`check`.
test-integration:
	uv run pytest -m integration

check: lint typecheck test
	uv run ruff format --check .

clean:
	rm -rf .venv .ruff_cache .pytest_cache __pycache__ src/mlsys.egg-info

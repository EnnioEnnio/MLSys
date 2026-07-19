.PHONY: setup lint format typecheck test test-integration check clean results

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

# Regenerate every results/ artifact (plots/tables/SUMMARY.md), e.g. after a theme/plot change.
# full_eval_noise, dataset_comparison, and activation_sweep are keyed to a scripts/*.py side
# report instead of generic `mlsys analyze` (see CLAUDE.md "Analysis"); everything else goes
# through `mlsys analyze` and is skipped with a note if it has no frozen+finetune/r1+r3 pair
# (e.g. a finetune-only sweep).
results:
	@for d in results/*/; do \
		name=$$(basename "$$d"); \
		case "$$name" in \
			full_eval_noise|dataset_comparison|activation_sweep) continue ;; \
		esac; \
		echo "==> mlsys analyze $$d"; \
		uv run python -m mlsys analyze "$$d" || echo "==> skipped $$d (no analysable pair)"; \
	done
	uv run python scripts/noise_report.py results/full_eval_noise
	uv run python scripts/cross_dataset_report.py
	uv run python scripts/activation_sweep_report.py results/activation_sweep --out results/activation_sweep/report.md

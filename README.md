# MLSys — Just-in-Time Model Replacement

Seminar project at HPI implementing the pipeline from [Poodle: Seamlessly Scaling Down Large Language Models with Just-in-Time Model Replacement](./Poodle-%20Seamlessly%20Scaling%20Down%20Large%20Language%20Models%20with%20Just-in-Time%20Model%20Replacement.pdf) by Strassenburg et al.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) installed on your machine.

```bash
make setup   # uv sync + pre-commit install
```

## Common commands

| Command           | What it does                          |
|-------------------|---------------------------------------|
| `make setup`      | Install deps + pre-commit hooks       |
| `make lint`       | Run ruff linter                       |
| `make format`     | Run ruff formatter + auto-fix         |
| `make typecheck`  | Run ty type checker                   |
| `make test`       | Run pytest                            |
| `make check`      | lint + typecheck + test (same as CI)  |
| `make clean`      | Remove .venv and caches               |

## Layout

```
src/mlsys/
├── core/          # Shared types (Row, Label, Prediction) and Protocols (Dataset, LLMClient, CandidateModel)
├── datasets/      # Dataset adapters behind the Dataset Protocol
├── llm/           # LLMClient Protocol + prompt/response plumbing (no concrete backend yet)
├── model_search/  # Candidate pool, scoring, selection — Poodle/Alsatian-style hooks
├── surrogate/     # Fine-tuning + serving the chosen surrogate
├── monitoring/    # Parallel A/B comparison + replacement decision
├── pipeline/      # Orchestration: wires all stages together
└── cli/           # `python -m mlsys <subcommand>` entrypoints
```

## Tooling at a glance

- **uv** — package manager and task runner; deps locked in `uv.lock`
- **ruff** — linter and formatter (replaces black + flake8)
- **ty** — Astral's type checker (pre-1.0; swap to pyright with one config change if needed)
- **pytest** — smoke + critical-path tests only; no coverage gate
- **pre-commit** — runs `ruff check --fix` and `ruff format` on every commit
- **GitHub Actions** — runs `make check` on push and PR

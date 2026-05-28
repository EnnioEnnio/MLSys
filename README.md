# MLSys — Just-in-Time Model Replacement

Seminar project at HPI researching model search for non-classification tasks.

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
//TODO: add layout here
```

## Tooling at a glance

- **uv** — package manager and task runner; deps locked in `uv.lock`
- **ruff** — linter and formatter (replaces black + flake8)
- **ty** — Astral's type checker (pre-1.0; swap to pyright with one config change if needed)
- **pytest** — smoke + critical-path tests only; no coverage gate
- **pre-commit** — runs `ruff check --fix` and `ruff format` on every commit
- **GitHub Actions** — runs `make check` on push and PR

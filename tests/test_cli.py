"""CLI: list-* print the seed pool; bad args exit non-zero (incl. the P3 exit-code guard)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from mlsys.cli import main


def test_list_models_prints_seed_pool(capsys) -> None:
    rc = main(["list-models"])
    assert rc == 0
    out = capsys.readouterr().out
    for name in ("all-MiniLM-L6-v2", "potion-base-8M"):
        assert name in out


def test_list_datasets_prints_entries(capsys) -> None:
    rc = main(["list-datasets"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "wine_reviews" in out


def test_unknown_strategy_exits_nonzero() -> None:
    # argparse rejects an out-of-choices --strategy at parse time (SystemExit != 0).
    with pytest.raises(SystemExit) as exc:
        main(["search", "--dataset", "wine_reviews", "--strategy", "nope"])
    assert exc.value.code != 0


def test_unknown_dataset_raises() -> None:
    with pytest.raises((KeyError, SystemExit)):
        main(["search", "--dataset", "definitely-not-a-dataset"])


def test_python_dash_m_returns_nonzero_on_bad_args() -> None:
    # P3 regression guard: `raise SystemExit(main())` must propagate a failure as a
    # non-zero process exit code (SLURM/CI rely on it).
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlsys",
            "search",
            "--dataset",
            "definitely-not-a-dataset",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0

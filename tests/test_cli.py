"""CLI: list-* print the seed pool; bad args exit non-zero (incl. the P3 exit-code guard)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from mlsys.cli import main


@pytest.mark.parametrize(
    "hidden,strategy,expected",
    [
        (512, "full_eval", "2296342_wine_reviews_fulleval_16_model_MLP_512"),
        (None, "full_eval", "2296342_wine_reviews_fulleval_16_model_FCH"),
        (0, "frozen", "2296342_wine_reviews_frozen_16_model_FCH"),
        (128, "finetune", "2296342_wine_reviews_finetune_16_model_MLP_128"),
    ],
)
def test_wandb_run_name_grammar(hidden, strategy, expected) -> None:
    # `main` is the entrypoint function; reach the submodule via sys.modules to call
    # its module-level helper (same name-clash as test_strategy_routes_to_run_strategy).
    import sys

    cli_main = sys.modules["mlsys.cli.main"]
    assert cli_main._wandb_run_name("2296342", "wine_reviews", strategy, 16, hidden) == expected


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


@pytest.mark.parametrize("strategy", ["frozen", "finetune", "full_eval"])
def test_strategy_routes_to_run_strategy(strategy, tmp_path, monkeypatch) -> None:
    # --strategy must accept all three choices and forward the name to run_strategy,
    # without touching the network/heavy deps (load_dataset + run_strategy stubbed).
    # The `main` function and the `mlsys.cli.main` submodule share a name on the
    # package, so reach the module object through sys.modules to patch its globals.
    import sys

    cli_main = sys.modules["mlsys.cli.main"]
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli_main, "load_dataset", lambda name: object())

    def fake_run_strategy(name, dataset, **kwargs):
        captured["strategy"] = name
        return []

    monkeypatch.setattr(cli_main, "run_strategy", fake_run_strategy)

    rc = main(
        [
            "search",
            "--dataset",
            "wine_reviews",
            "--strategy",
            strategy,
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    assert rc == 0
    assert captured["strategy"] == strategy


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

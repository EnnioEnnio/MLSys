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


def test_list_models_count_matches_pool(capsys) -> None:
    from mlsys.models.registry import load_specs

    rc = main(["list-models", "--count"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == str(len(load_specs()))


def test_list_models_index_prints_bare_name(capsys) -> None:
    from mlsys.models.registry import load_specs

    rc = main(["list-models", "--index", "0"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == next(iter(load_specs()))


def test_list_models_index_out_of_range_exits_nonzero(capsys) -> None:
    # A bad --array bound must fail the task loudly so afterok blocks consolidation.
    rc = main(["list-models", "--index", "9999"])
    assert rc != 0
    assert "out of range" in capsys.readouterr().err


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


@pytest.mark.parametrize("extra_args,expected", [([], 0), (["--warmup-epochs", "2"], 2)])
def test_warmup_epochs_reaches_finetune_config(extra_args, expected, tmp_path, monkeypatch) -> None:
    # --warmup-epochs threads through to run_strategy's finetune_config; default is 0.
    import sys

    cli_main = sys.modules["mlsys.cli.main"]
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli_main, "load_dataset", lambda name: object())

    def fake_run_strategy(name, dataset, **kwargs):
        captured["warmup_epochs"] = kwargs["finetune_config"].warmup_epochs
        return []

    monkeypatch.setattr(cli_main, "run_strategy", fake_run_strategy)

    rc = main(
        [
            "search",
            "--dataset",
            "wine_reviews",
            "--strategy",
            "finetune",
            "--output-dir",
            str(tmp_path / "run"),
            *extra_args,
        ]
    )
    assert rc == 0
    assert captured["warmup_epochs"] == expected


@pytest.mark.parametrize("extra_args,expected", [([], 0.0), (["--grad-clipping", "1.0"], 1.0)])
def test_grad_clipping_reaches_finetune_config(extra_args, expected, tmp_path, monkeypatch) -> None:
    # --grad-clipping threads through to run_strategy's finetune_config; default is 0 (off).
    import sys

    cli_main = sys.modules["mlsys.cli.main"]
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli_main, "load_dataset", lambda name: object())

    def fake_run_strategy(name, dataset, **kwargs):
        captured["grad_clipping"] = kwargs["finetune_config"].grad_clipping
        return []

    monkeypatch.setattr(cli_main, "run_strategy", fake_run_strategy)

    rc = main(
        [
            "search",
            "--dataset",
            "wine_reviews",
            "--strategy",
            "finetune",
            "--output-dir",
            str(tmp_path / "run"),
            *extra_args,
        ]
    )
    assert rc == 0
    assert captured["grad_clipping"] == expected


@pytest.mark.parametrize(
    "extra_args,expected",
    [([], True), (["--no-standardize-targets"], False), (["--standardize-targets"], True)],
)
def test_standardize_targets_reaches_head_config(
    extra_args, expected, tmp_path, monkeypatch
) -> None:
    # --[no-]standardize-targets threads through to run_strategy's head_config;
    # default is on (regression recipe, issue #32).
    import sys

    cli_main = sys.modules["mlsys.cli.main"]
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli_main, "load_dataset", lambda name: object())

    def fake_run_strategy(name, dataset, **kwargs):
        captured["standardize_targets"] = kwargs["head_config"].standardize_targets
        return []

    monkeypatch.setattr(cli_main, "run_strategy", fake_run_strategy)

    rc = main(
        [
            "search",
            "--dataset",
            "wine_reviews",
            "--strategy",
            "frozen",
            "--output-dir",
            str(tmp_path / "run"),
            *extra_args,
        ]
    )
    assert rc == 0
    assert captured["standardize_targets"] is expected


def _fake_consolidation_result(tmp_path, rows=(), summary=None):
    from mlsys.search.consolidate import ConsolidationResult

    return ConsolidationResult(
        dataset="wine_reviews",
        run_name="42_wine_reviews_fulleval_2_model_FCH",
        rows=list(rows),
        results_path=tmp_path / "results.jsonl",
        regret_path=tmp_path / "regret.json" if summary is not None else None,
        summary=summary,
        csv_paths=[],
    )


def test_consolidate_routes_flags_to_consolidate_run(tmp_path, monkeypatch) -> None:
    # Mirrors test_strategy_routes_to_run_strategy: stub the worker, assert forwarding.
    import mlsys.search.consolidate as consolidate_mod

    captured: dict[str, object] = {}

    def fake_consolidate_run(run_dir, **kwargs):
        captured["run_dir"] = run_dir
        captured.update(kwargs)
        return _fake_consolidation_result(tmp_path)

    monkeypatch.setattr(consolidate_mod, "consolidate_run", fake_consolidate_run)

    rc = main(["consolidate", str(tmp_path), "--hidden", "512", "--cleanup", "--allow-partial"])
    assert rc == 0
    assert captured == {
        "run_dir": str(tmp_path),
        "hidden": 512,
        "cleanup": True,
        "allow_partial": True,
    }


def test_consolidate_wandb_logs_both_pass_tables(tmp_path, monkeypatch) -> None:
    # The consolidated run must log results_frozen + results_finetune — the same table
    # names as a single-node full_eval — so the analysis CSV-download workflow holds.
    import sys
    import types

    from mlsys.search.regret import summarize_regret

    logged: dict[str, object] = {}

    class _FakeTable:
        def __init__(self, columns=(), data=()):
            self.columns = list(columns)
            self.data = list(data)

        def add_data(self, *row):
            pass

    fake_wandb = types.SimpleNamespace(
        init=lambda **kw: types.SimpleNamespace(finish=lambda: None, **kw),
        log=lambda payload: logged.update(payload),
        Table=_FakeTable,
        plot=types.SimpleNamespace(line=lambda *a, **kw: "line-plot"),
        run=None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    rows = []
    for strategy in ("frozen", "finetune"):
        for model in ("m1", "m2"):
            rows.append(
                {
                    "model": model,
                    "dataset": "wine_reviews",
                    "strategy": strategy,
                    "metrics": {"r2": 0.5},
                    "timing": {"train_head_s": 1.0},
                    "head_train_curve": [],
                    "head_val_curve": [],
                    "epochs_run": 1,
                }
            )
    summary = summarize_regret("wine_reviews", {"m1": 0.5, "m2": 0.4}, {"m1": 0.6, "m2": 0.7})

    import mlsys.search.consolidate as consolidate_mod

    monkeypatch.setattr(
        consolidate_mod,
        "consolidate_run",
        lambda run_dir, **kw: _fake_consolidation_result(tmp_path, rows, summary),
    )

    rc = main(["consolidate", str(tmp_path), "--wandb"])
    assert rc == 0
    for name in ("results_frozen", "results_finetune"):
        table = logged[name]
        assert isinstance(table, _FakeTable)
        assert table.columns[:3] == ["model", "dataset", "strategy"]
        assert len(table.data) == 2
    assert "regret_vs_budget" in logged


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

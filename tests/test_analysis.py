"""Analysis module: filename grammar, triple discovery, regret-recompute parity, flag
detection, table builders, a plot smoke test, and CLI smokes.

CPU-only. The pandas/seaborn-dependent tests ``importorskip`` their dep so a suite installed
without the optional ``analysis`` group skips cleanly (mirrors the integration-skip
convention)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlsys.analysis import loader
from mlsys.analysis.loader import head_sort_key, parse_filename

# --- column layout matching what full_eval writes to results.jsonl / the CSV dumps ---
_COLS = [
    "model",
    "dataset",
    "strategy",
    "mse",
    "mae",
    "r2",
    "spearman",
    "prepare_model_s",
    "prepare_data_s",
    "inference_s",
    "train_head_s",
    "eval_s",
    "peak_gpu_mem_mb",
    "epochs_run",
    "embedding_dim",
    "head_type",
    "head_repeats",
]


def _row(model, r2, *, inference_s, train_head_s, epochs_run, strategy, spearman=0.8):
    return {
        "model": model,
        "dataset": "wine_reviews",
        "strategy": strategy,
        "mse": 1.0,
        "mae": 0.5,
        "r2": r2,
        "spearman": spearman,
        "prepare_model_s": 1.0,
        "prepare_data_s": 2.0,
        "inference_s": inference_s,
        "train_head_s": train_head_s,
        "eval_s": 0.1,
        "peak_gpu_mem_mb": 1000.0,
        "epochs_run": epochs_run,
        "embedding_dim": 384,
        "head_type": "mlp",
        "head_repeats": 1 if strategy == "finetune" else 3,
    }


def _write_triple(folder: Path, run_id: str, head: str, *, with_regret: bool, head_type="mlp"):
    """Write a frozen+finetune (+optional regret) CSV trio with the filename grammar."""
    import pandas as pd

    # 3 real backbones (inference_s==0 in finetune) + 1 model2vec fallback (inference_s>0).
    frozen = [
        _row("alpha", 0.50, inference_s=10, train_head_s=5, epochs_run=20, strategy="frozen"),
        _row("beta", 0.40, inference_s=10, train_head_s=5, epochs_run=20, strategy="frozen"),
        _row("gamma", 0.30, inference_s=10, train_head_s=5, epochs_run=20, strategy="frozen"),
        _row("m2v", 0.20, inference_s=2, train_head_s=5, epochs_run=20, strategy="frozen"),
    ]
    finetune = [
        # alpha diverges (r2<0) but keeps high spearman.
        _row(
            "alpha",
            -0.10,
            inference_s=0,
            train_head_s=50,
            epochs_run=3,
            strategy="finetune",
            spearman=0.9,
        ),
        _row("beta", 0.70, inference_s=0, train_head_s=50, epochs_run=3, strategy="finetune"),
        _row("gamma", 0.60, inference_s=0, train_head_s=50, epochs_run=3, strategy="finetune"),
        # m2v skipped: inference_s>0 + early-stop epochs != budget. Its negative "finetune" r²
        # is the reused frozen score, not a real divergence — must NOT be flagged diverged.
        _row("m2v", -0.05, inference_s=2, train_head_s=5, epochs_run=20, strategy="finetune"),
    ]
    for kind, rows in (("frozen", frozen), ("finetune", finetune)):
        for r in rows:
            r["head_type"] = head_type
        df = pd.DataFrame(rows, columns=_COLS)
        df.to_csv(folder / f"{run_id}_fulleval_4_model_{head}_{kind}.csv", index=False)
    if with_regret:
        from mlsys.analysis.regret_recompute import recompute_regret

        rc = recompute_regret(pd.DataFrame(frozen), pd.DataFrame(finetune))
        rc.to_csv(folder / f"{run_id}_fulleval_4_model_{head}_regret.csv", index=False)


# ----------------------------------------------------------------- filename grammar


def test_parse_filename_fch_and_mlp():
    a = parse_filename("2296332_fulleval_16_model_FCH_frozen.csv")
    assert (a.run_id, a.head, a.kind) == ("2296332", "FCH", "frozen")
    b = parse_filename("2296342_fulleval_16_model_MLP_512_regret.csv")
    assert (b.run_id, b.head, b.kind) == ("2296342", "MLP_512", "regret")


def test_parse_filename_rejects_bad_name():
    with pytest.raises(ValueError, match="grammar"):
        parse_filename("garbage.csv")
    with pytest.raises(ValueError, match="kind"):
        parse_filename("1_fulleval_4_model_FCH_unknownkind.csv")


def test_head_sort_key_orders_by_capacity():
    heads = ["MLP_512", "FCH", "MLP_128", "MLP_256"]
    assert sorted(heads, key=head_sort_key) == ["FCH", "MLP_128", "MLP_256", "MLP_512"]


# ----------------------------------------------------------------- discovery / loading


def test_discover_triples_groups_and_orders(tmp_path):
    pytest.importorskip("pandas")
    _write_triple(tmp_path, "100", "MLP_128", with_regret=True)
    _write_triple(tmp_path, "200", "FCH", with_regret=True, head_type="linear")
    tfs = loader.discover_triples(tmp_path)
    assert [tf.head for tf in tfs] == ["FCH", "MLP_128"]  # capacity order
    assert all("frozen" in tf.paths and "finetune" in tf.paths for tf in tfs)


def test_load_triple_flags_diverged_and_skipped(tmp_path):
    pytest.importorskip("pandas")
    _write_triple(tmp_path, "100", "MLP_128", with_regret=False)
    (tf,) = loader.discover_triples(tmp_path)
    assert "regret" not in tf.paths  # missing-regret case
    triple = loader.load_triple(tf)
    assert triple.diverged == {"alpha": True, "beta": False, "gamma": False, "m2v": False}
    assert triple.finetune_skipped == {
        "alpha": False,
        "beta": False,
        "gamma": False,
        "m2v": True,
    }


# ----------------------------------------------------------------- regret parity


def test_recompute_regret_parity_with_regret_curve(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    from mlsys.analysis.regret_recompute import proxy_ranking_from_frozen, recompute_regret
    from mlsys.search.regret import regret_curve

    frozen = pd.DataFrame(
        [
            _row("a", 0.5, inference_s=1, train_head_s=1, epochs_run=5, strategy="frozen"),
            _row("b", 0.3, inference_s=1, train_head_s=1, epochs_run=5, strategy="frozen"),
            _row("c", 0.4, inference_s=1, train_head_s=1, epochs_run=5, strategy="frozen"),
        ]
    )
    finetune = pd.DataFrame(
        [
            _row("a", 0.6, inference_s=0, train_head_s=1, epochs_run=3, strategy="finetune"),
            _row("b", 0.9, inference_s=0, train_head_s=1, epochs_run=3, strategy="finetune"),
            _row("c", 0.4, inference_s=0, train_head_s=1, epochs_run=3, strategy="finetune"),
        ]
    )
    proxy = proxy_ranking_from_frozen(frozen)
    assert proxy == ["a", "c", "b"]  # frozen r2 desc
    expected = regret_curve(proxy, {"a": 0.6, "b": 0.9, "c": 0.4})
    got = recompute_regret(frozen, finetune)
    assert list(got["budget"]) == [p.budget for p in expected]
    assert got["regret"].tolist() == pytest.approx([p.regret for p in expected])
    assert got["normalized_regret"].tolist() == pytest.approx(
        [p.normalized_regret for p in expected]
    )


# ----------------------------------------------------------------- table builders


def test_table_builders_on_tiny_frames(tmp_path):
    pytest.importorskip("pandas")
    from mlsys.analysis import tables

    _write_triple(tmp_path, "100", "MLP_128", with_regret=True)
    (tf,) = loader.discover_triples(tmp_path)
    triple = loader.load_triple(tf)

    per = tables.per_triple_table(triple)
    assert set(per["model"]) == {"alpha", "beta", "gamma", "m2v"}
    assert "delta_r2" in per.columns

    summary = tables.per_triple_summary(triple)
    assert summary.best_finetune_model == "beta"  # 0.70 is max finetune r2
    assert summary.n_diverged == 1
    assert 1 <= summary.budget_to_zero <= len(triple.models)

    md = tables.df_to_markdown(per)
    assert md.startswith("| model |")


# ----------------------------------------------------------------- plot smoke


def test_plot_smoke_savefig(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    from mlsys.analysis import plots

    _write_triple(tmp_path, "100", "MLP_128", with_regret=True)
    (tf,) = loader.discover_triples(tmp_path)
    triple = loader.load_triple(tf)
    out = tmp_path / "plots"
    p = plots.plot_regret_curve(triple, out)
    assert p.exists() and p.stat().st_size > 0
    p2 = plots.plot_heatmap_frozen_r2([triple], out)
    assert p2.exists() and p2.stat().st_size > 0


# ----------------------------------------------------------------- CLI smokes


def test_cli_regret_smoke(tmp_path, capsys):
    pytest.importorskip("pandas")
    from mlsys.cli import main

    _write_triple(tmp_path, "100", "MLP_128", with_regret=True)
    out = tmp_path / "recovered.csv"
    rc = main(
        [
            "regret",
            "--frozen",
            str(tmp_path / "100_fulleval_4_model_MLP_128_frozen.csv"),
            "--finetune",
            str(tmp_path / "100_fulleval_4_model_MLP_128_finetune.csv"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    import pandas as pd

    assert list(pd.read_csv(out).columns) == ["budget", "regret", "normalized_regret"]


def test_cli_analyze_smoke_and_crash_recovery(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    from mlsys.cli import main

    # One head with regret, one without (exercises recompute-on-the-fly recovery).
    _write_triple(tmp_path, "100", "FCH", with_regret=True, head_type="linear")
    _write_triple(tmp_path, "200", "MLP_512", with_regret=False)
    rc = main(["analyze", str(tmp_path)])
    assert rc == 0
    out = tmp_path / "analysis"
    assert (out / "SUMMARY.md").exists()
    assert (out / "FCH" / "tables.csv").exists()
    assert (out / "MLP_512" / "tables.csv").exists()
    assert (out / "comparison" / "per_head_summary.csv").exists()
    # The missing MLP_512 regret CSV was recomputed and written back into the folder.
    assert (tmp_path / "200_fulleval_4_model_MLP_512_regret.csv").exists()

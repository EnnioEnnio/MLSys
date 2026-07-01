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


def _row(
    model,
    r2,
    *,
    inference_s,
    train_head_s,
    epochs_run,
    strategy,
    spearman=0.8,
    peak_gpu_mem_mb=1000.0,
):
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
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "epochs_run": epochs_run,
        "embedding_dim": 384,
        "head_type": "mlp",
        "head_repeats": 1 if strategy == "finetune" else 3,
    }


def _write_repeat_pair(
    folder: Path,
    run_id: str,
    head: str,
    *,
    with_regret: bool,
    head_type="mlp",
):
    """Write an r1+r3 (+optional regret) CSV pair for a frozen repeat-count comparison."""
    import pandas as pd

    # r1: lower-variance proxy (repeat=1); r3: higher-variance truth (repeat=3).
    # Same model pool as _write_triple but both passes are "frozen" (inference_s > 0 for both).
    r1_rows = [
        _row("alpha", 0.48, inference_s=10, train_head_s=5, epochs_run=20, strategy="frozen"),
        _row("beta", 0.38, inference_s=10, train_head_s=5, epochs_run=20, strategy="frozen"),
        _row("gamma", 0.28, inference_s=10, train_head_s=5, epochs_run=20, strategy="frozen"),
        _row("m2v", 0.18, inference_s=2, train_head_s=5, epochs_run=20, strategy="frozen"),
    ]
    r3_rows = [
        _row("alpha", 0.50, inference_s=10, train_head_s=15, epochs_run=20, strategy="frozen"),
        _row("beta", 0.40, inference_s=10, train_head_s=15, epochs_run=20, strategy="frozen"),
        _row("gamma", 0.30, inference_s=10, train_head_s=15, epochs_run=20, strategy="frozen"),
        _row("m2v", 0.20, inference_s=2, train_head_s=15, epochs_run=20, strategy="frozen"),
    ]
    for kind, rows in (("r1", r1_rows), ("r3", r3_rows)):
        for r in rows:
            r["head_type"] = head_type
        df = pd.DataFrame(rows, columns=_COLS)
        df.to_csv(folder / f"{run_id}_wine_reviews_frozen_4_model_{head}_{kind}.csv", index=False)
    if with_regret:
        from mlsys.analysis.regret_recompute import recompute_regret

        rc = recompute_regret(pd.DataFrame(r1_rows), pd.DataFrame(r3_rows))
        rc.to_csv(folder / f"{run_id}_wine_reviews_frozen_4_model_{head}_regret.csv", index=False)


def _write_triple(
    folder: Path,
    run_id: str,
    head: str,
    *,
    with_regret: bool,
    head_type="mlp",
    peak_gpu_mem_mb=1000.0,
):
    """Write a frozen+finetune (+optional regret) CSV trio with the filename grammar."""
    import pandas as pd

    mem = peak_gpu_mem_mb
    # 3 real backbones (inference_s==0 in finetune) + 1 model2vec fallback (inference_s>0).
    frozen = [
        _row(
            "alpha",
            0.50,
            inference_s=10,
            train_head_s=5,
            epochs_run=20,
            strategy="frozen",
            peak_gpu_mem_mb=mem,
        ),
        _row(
            "beta",
            0.40,
            inference_s=10,
            train_head_s=5,
            epochs_run=20,
            strategy="frozen",
            peak_gpu_mem_mb=mem,
        ),
        _row(
            "gamma",
            0.30,
            inference_s=10,
            train_head_s=5,
            epochs_run=20,
            strategy="frozen",
            peak_gpu_mem_mb=mem,
        ),
        _row(
            "m2v",
            0.20,
            inference_s=2,
            train_head_s=5,
            epochs_run=20,
            strategy="frozen",
            peak_gpu_mem_mb=mem,
        ),
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
            peak_gpu_mem_mb=mem,
        ),
        _row(
            "beta",
            0.70,
            inference_s=0,
            train_head_s=50,
            epochs_run=3,
            strategy="finetune",
            peak_gpu_mem_mb=mem,
        ),
        _row(
            "gamma",
            0.60,
            inference_s=0,
            train_head_s=50,
            epochs_run=3,
            strategy="finetune",
            peak_gpu_mem_mb=mem,
        ),
        # m2v skipped: inference_s>0 + early-stop epochs != budget. Its negative "finetune" r²
        # is the reused frozen score, not a real divergence — must NOT be flagged diverged.
        _row(
            "m2v",
            -0.05,
            inference_s=2,
            train_head_s=5,
            epochs_run=20,
            strategy="finetune",
            peak_gpu_mem_mb=mem,
        ),
    ]
    for kind, rows in (("frozen", frozen), ("finetune", finetune)):
        for r in rows:
            r["head_type"] = head_type
        df = pd.DataFrame(rows, columns=_COLS)
        df.to_csv(folder / f"{run_id}_wine_reviews_fulleval_4_model_{head}_{kind}.csv", index=False)
    if with_regret:
        from mlsys.analysis.regret_recompute import recompute_regret

        rc = recompute_regret(pd.DataFrame(frozen), pd.DataFrame(finetune))
        rc.to_csv(folder / f"{run_id}_wine_reviews_fulleval_4_model_{head}_regret.csv", index=False)


# ----------------------------------------------------------------- filename grammar


def test_parse_filename_fch_and_mlp():
    a = parse_filename("2296332_wine_reviews_fulleval_16_model_FCH_frozen.csv")
    assert (a.run_id, a.dataset, a.head, a.kind) == ("2296332", "wine_reviews", "FCH", "frozen")
    b = parse_filename("2296342_wine_reviews_fulleval_16_model_MLP_512_regret.csv")
    assert (b.run_id, b.dataset, b.head, b.kind) == ("2296342", "wine_reviews", "MLP_512", "regret")


def test_parse_filename_single_token_dataset():
    p = parse_filename("42_wine_frozen_8_model_MLP_256_finetune.csv")
    assert (p.run_id, p.dataset, p.strategy, p.num) == ("42", "wine", "frozen", "8")


def test_parse_filename_r1_r3():
    p = parse_filename("exp001_wine_reviews_frozen_4_model_FCH_r1.csv")
    assert (p.run_id, p.head, p.kind) == ("exp001", "FCH", "r1")
    p3 = parse_filename("exp001_wine_reviews_frozen_4_model_MLP_256_r3.csv")
    assert (p3.run_id, p3.head, p3.kind) == ("exp001", "MLP_256", "r3")


def test_parse_filename_rejects_bad_name():
    with pytest.raises(ValueError, match="grammar"):
        parse_filename("garbage.csv")
    # no dataset token (runid, strategy, num, model, head) → fails the model_idx >= 4 guard.
    with pytest.raises(ValueError, match="grammar"):
        parse_filename("1_fulleval_4_model_FCH_frozen.csv")
    with pytest.raises(ValueError, match="kind"):
        parse_filename("1_wine_fulleval_4_model_FCH_unknownkind.csv")


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


# ----------------------------------------------------------------- new table builders


def test_frozen_distribution_table(tmp_path):
    pytest.importorskip("pandas")
    from mlsys.analysis import tables

    _write_triple(tmp_path, "100", "FCH", with_regret=True, head_type="linear")
    _write_triple(tmp_path, "200", "MLP_512", with_regret=True)
    tfs = loader.discover_triples(tmp_path)
    triples = [loader.load_triple(tf) for tf in tfs]

    df = tables.frozen_distribution_table(triples)
    expected_cols = [
        "head",
        "mean_frozen_r2",
        "std_frozen_r2",
        "min_frozen_r2",
        "max_frozen_r2",
        "n_negative",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 2  # one row per head
    # population std: frozen r2 = [0.5, 0.4, 0.3, 0.2], mean=0.35
    # variance = ((0.15^2 + 0.05^2 + 0.05^2 + 0.15^2) / 4)
    import math

    expected_std = math.sqrt((0.0225 + 0.0025 + 0.0025 + 0.0225) / 4)
    import pytest as _pytest

    actual = df.loc[df["head"] == "FCH", "std_frozen_r2"].iloc[0]
    assert actual == _pytest.approx(expected_std, abs=1e-6)


def test_head_gain_table(tmp_path):
    pytest.importorskip("pandas")
    from mlsys.analysis import tables

    _write_triple(tmp_path, "100", "FCH", with_regret=True, head_type="linear")
    _write_triple(tmp_path, "200", "MLP_512", with_regret=True)
    tfs = loader.discover_triples(tmp_path)
    triples = [loader.load_triple(tf) for tf in tfs]

    df = tables.head_gain_table(triples)
    assert set(df.columns) >= {"model", "narrow_r2", "wide_r2", "gain"}
    assert len(df) == 4  # 4 models
    # Both heads have identical frozen data in _write_triple → gain = 0 for every model
    assert all(abs(g) < 1e-9 for g in df["gain"])


def test_epochs_table(tmp_path):
    pytest.importorskip("pandas")
    from mlsys.analysis import tables

    _write_triple(tmp_path, "100", "FCH", with_regret=True, head_type="linear")
    _write_triple(tmp_path, "200", "MLP_512", with_regret=True)
    tfs = loader.discover_triples(tmp_path)
    triples = [loader.load_triple(tf) for tf in tfs]

    df = tables.epochs_table(triples)
    assert set(df.columns) >= {
        "head",
        "mean_frozen_epochs",
        "n_frozen_at_cap",
        "frozen_cap",
        "mean_finetune_epochs",
    }
    assert len(df) == 2
    # All frozen rows have epochs_run=20, cap=20 → n_frozen_at_cap=4 (all 4 models hit cap)
    assert df["frozen_cap"].iloc[0] == 20
    assert df.loc[df["head"] == "FCH", "n_frozen_at_cap"].iloc[0] == 4


def test_head_rank_agreement_matrix(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    from mlsys.analysis import tables

    _write_triple(tmp_path, "100", "FCH", with_regret=True, head_type="linear")
    _write_triple(tmp_path, "200", "MLP_512", with_regret=True)
    tfs = loader.discover_triples(tmp_path)
    triples = [loader.load_triple(tf) for tf in tfs]

    df = tables.head_rank_agreement_matrix(triples)
    # head column + one column per head → 3 columns for 2 heads
    assert "head" in df.columns
    assert len(df) == 2  # 2 rows (one per head)
    # Diagonal must be 1.0 (identical series -> Spearman rho=1)
    for _i, row in df.iterrows():
        head = row["head"]
        assert abs(float(row[head]) - 1.0) < 1e-9


def test_frozen_timing_share_table(tmp_path):
    pytest.importorskip("pandas")
    from mlsys.analysis import tables

    _write_triple(tmp_path, "100", "MLP_128", with_regret=True)
    (tf,) = loader.discover_triples(tmp_path)
    triple = loader.load_triple(tf)

    df = tables.frozen_timing_share_table([triple])
    assert "head" in df.columns
    assert "inference_pct" in df.columns
    assert len(df) == 1
    # All pct columns sum to 100
    pct_cols = [
        "prepare_model_pct",
        "prepare_data_pct",
        "inference_pct",
        "train_head_pct",
        "eval_pct",
    ]
    row_sum = sum(float(df[c].iloc[0]) for c in pct_cols)
    assert abs(row_sum - 100.0) < 1e-6


def test_value_frontier_table(tmp_path):
    pytest.importorskip("pandas")
    from mlsys.analysis import tables

    _write_triple(tmp_path, "100", "FCH", with_regret=True, head_type="linear")
    _write_triple(tmp_path, "200", "MLP_512", with_regret=True)
    tfs = loader.discover_triples(tmp_path)
    triples = [loader.load_triple(tf) for tf in tfs]

    df = tables.value_frontier_table(triples)
    assert set(df.columns) >= {
        "model",
        "frozen_inference_s",
        "frozen_r2",
        "finetune_r2",
        "frozen_peak_gpu_mem_mb",
    }
    assert len(df) == 4  # 4 models, widest head
    # Sorted by inference_s asc: m2v (2s) should appear before alpha/beta/gamma (10s)
    assert df.iloc[0]["model"] == "m2v"


def test_per_triple_table_has_frozen_epochs(tmp_path):
    pytest.importorskip("pandas")
    from mlsys.analysis import tables

    _write_triple(tmp_path, "100", "MLP_128", with_regret=True)
    (tf,) = loader.discover_triples(tmp_path)
    triple = loader.load_triple(tf)

    df = tables.per_triple_table(triple)
    assert "frozen_epochs" in df.columns
    assert list(df["frozen_epochs"]) == [20, 20, 20, 20]


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


def test_new_plots_smoke(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    pytest.importorskip("scipy")
    from mlsys.analysis import plots

    _write_triple(tmp_path, "100", "FCH", with_regret=True, head_type="linear")
    _write_triple(tmp_path, "200", "MLP_512", with_regret=True)
    tfs = loader.discover_triples(tmp_path)
    triples = [loader.load_triple(tf) for tf in tfs]
    out = tmp_path / "plots"

    p_epochs = plots.plot_epochs_vs_head(triples, out)
    assert p_epochs.exists() and p_epochs.stat().st_size > 0

    p_rank = plots.plot_head_rank_agreement(triples, out)
    assert p_rank.exists() and p_rank.stat().st_size > 0

    p_timing = plots.plot_frozen_timing_share(triples, out)
    assert p_timing.exists() and p_timing.stat().st_size > 0

    p_frontier = plots.plot_value_frontier(triples, out)
    assert p_frontier.exists() and p_frontier.stat().st_size > 0


# ----------------------------------------------------------------- r1/r3 repeat comparison


def test_load_triple_r1_r3_labels(tmp_path):
    """r1+r3 pair → proxy_label='repeat=1', truth_label='repeat=3', finetune_skipped empty."""
    pytest.importorskip("pandas")
    _write_repeat_pair(tmp_path, "exp001", "FCH", with_regret=False, head_type="linear")
    (tf,) = loader.discover_triples(tmp_path)
    assert tf.paths.keys() >= {"r1", "r3"}
    triple = loader.load_triple(tf)
    assert triple.proxy_label == "repeat=1"
    assert triple.truth_label == "repeat=3"
    # finetune_skipped must be empty — both passes are frozen (inference_s > 0 for all)
    assert triple.finetune_skipped == {}
    # diverged is still meaningful: truth r² < 0 flags a poor model
    assert set(triple.diverged.keys()) == {"alpha", "beta", "gamma", "m2v"}
    assert all(not v for v in triple.diverged.values())  # all r3 r² > 0


def test_cli_analyze_r1_r3_smoke(tmp_path):
    """Full analyze pipeline works end-to-end for a frozen r1/r3 repeat comparison."""
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    from mlsys.cli import main

    _write_repeat_pair(tmp_path, "exp001", "FCH", with_regret=False, head_type="linear")
    rc = main(["analyze", str(tmp_path)])
    assert rc == 0
    out = tmp_path / "analysis"
    summary = (out / "SUMMARY.md").read_text()
    assert "repeat=1" in summary
    assert "repeat=3" in summary
    # The regret CSV should have been recomputed and written back.
    assert (tmp_path / "exp001_wine_reviews_frozen_4_model_FCH_regret.csv").exists()


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
            str(tmp_path / "100_wine_reviews_fulleval_4_model_MLP_128_frozen.csv"),
            "--finetune",
            str(tmp_path / "100_wine_reviews_fulleval_4_model_MLP_128_finetune.csv"),
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
    assert (tmp_path / "200_wine_reviews_fulleval_4_model_MLP_512_regret.csv").exists()


def test_cli_analyze_cpu_run_zero_gpu_mem(tmp_path):
    """CPU-only runs record peak_gpu_mem_mb=0; the synthesis section must not ZeroDivide.

    Regression: ``_synthesis_section`` divided finetune/frozen peak GPU memory unguarded, so a
    zero-memory CPU run crashed *after* every table+plot was written, leaving the analysis
    folder with no SUMMARY.md. The ratio must degrade to ``n/a`` instead.
    """
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    from mlsys.cli import main

    _write_triple(tmp_path, "100", "FCH", with_regret=True, peak_gpu_mem_mb=0.0)
    assert main(["analyze", str(tmp_path)]) == 0
    summary = (tmp_path / "analysis" / "SUMMARY.md").read_text()
    # The memory ratio falls back to n/a; the table/plot artifacts are still emitted.
    assert "peak GPU mem ratio:** n/a" in summary

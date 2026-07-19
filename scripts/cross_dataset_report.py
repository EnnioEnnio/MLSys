"""Cross-dataset comparison at the fixed final-recipe config (MLP_256, warmup-2, z-scoring).

`mlsys analyze` treats one folder as one experiment (single dataset, heads as the comparison
axis); this side-script compares along the *dataset/scale* axis instead — the second-dataset
question (METHOD.md "The second dataset has a job": does the pool separate?) and the
model-replacement question (does the proxy ranking survive a ~2k-row budget?):

- wine_full     = results/full_eval_noise (per-model mean over the 5 seed repeats)
- wine_tiny     = results/wine_tiny_test (2k rows)
- housing_full  = results/full_eval_usa_housing_no_early_stopping (200k rows)
- housing_tiny  = results/housing_tiny_test (2k rows)

Tables: per-dataset quality/separation summary + rank-survival Spearmans. Figures: per-model
r2 strips per dataset, rank-survival bars, and the housing diagnosis scatter (per-model
prediction-vs-target Spearman vs r2 — high rho at r2 ~ 0 means the models rank prices fine
and the long-tailed target destroys the squared error, which no affine z-scoring can fix).

Deterministic: no RNG, sorted discovery, no timestamps in outputs.

Usage:
    uv run python scripts/cross_dataset_report.py
    uv run python scripts/cross_dataset_report.py --out-dir results/dataset_comparison
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Sibling side-script (scripts/ is not a package; script-dir on sys.path makes this work).
from noise_report import load_seed_triples, r2_matrix, trainable_models

from mlsys.analysis import theme
from mlsys.analysis.loader import discover_triples, load_triple, resolve_role_pair
from mlsys.analysis.regret_recompute import triple_regret_df
from mlsys.analysis.tables import df_to_markdown, rank_spearman, write_table

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_DIRS = {
    "wine_full": Path("results/full_eval_noise"),
    "wine_tiny": Path("results/wine_tiny_test"),
    "housing_full": Path("results/full_eval_usa_housing_no_early_stopping"),
    "housing_tiny": Path("results/housing_tiny_test"),
}


@dataclass
class DatasetRun:
    """One dataset's per-model signals at the shared MLP_256 recipe (seed-averaged if 5x)."""

    label: str
    frozen_r2: pd.Series
    finetune_r2: pd.Series
    frozen_spearman: pd.Series  # per-model predictions-vs-targets rank corr
    finetune_spearman: pd.Series
    trainable: list[str]
    cross_pass_rho: float  # frozen-vs-finetune rank agreement (mean over seeds if 5x)
    regret_at_1: float  # mean over seeds if 5x


def _load_single(label: str, directory: Path) -> DatasetRun:
    candidates = [tf for tf in discover_triples(directory) if resolve_role_pair(tf.paths)]
    if len(candidates) != 1:
        raise ValueError(f"{directory}: expected exactly 1 analysable run, got {len(candidates)}")
    t = load_triple(candidates[0])
    skipped = {m for m, flag in t.ref_skipped.items() if flag}
    return DatasetRun(
        label=label,
        frozen_r2=t.proxy.set_index("model")["r2"],
        finetune_r2=t.reference.set_index("model")["r2"],
        frozen_spearman=t.proxy.set_index("model")["spearman"],
        finetune_spearman=t.reference.set_index("model")["spearman"],
        trainable=[m for m in t.models if m not in skipped],
        cross_pass_rho=rank_spearman(t),
        regret_at_1=float(triple_regret_df(t).iloc[0]["regret"]),
    )


def _load_seed_averaged(label: str, directory: Path) -> DatasetRun:
    import numpy as np
    import pandas as pd

    triples = load_seed_triples(directory)
    spearman = {
        which: pd.DataFrame(
            {
                seed: (t.proxy if which == "frozen" else t.reference).set_index("model")["spearman"]
                for seed, t in triples.items()
            }
        ).mean(axis=1)
        for which in ("frozen", "finetune")
    }
    return DatasetRun(
        label=label,
        frozen_r2=r2_matrix(triples, "frozen").mean(axis=1),
        finetune_r2=r2_matrix(triples, "finetune").mean(axis=1),
        frozen_spearman=spearman["frozen"],
        finetune_spearman=spearman["finetune"],
        trainable=trainable_models(triples),
        cross_pass_rho=float(np.mean([rank_spearman(t) for t in triples.values()])),
        regret_at_1=float(
            np.mean([triple_regret_df(t).iloc[0]["regret"] for t in triples.values()])
        ),
    )


def load_runs(dirs: dict[str, Path]) -> dict[str, DatasetRun]:
    runs: dict[str, DatasetRun] = {}
    for label, directory in dirs.items():
        loader = _load_seed_averaged if label == "wine_full" else _load_single
        runs[label] = loader(label, directory)
    return runs


# --------------------------------------------------------------------------- tables


def comparison_table(runs: dict[str, DatasetRun]) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for run in runs.values():
        ft_trainable = run.finetune_r2.loc[run.trainable]
        rows.append(
            {
                "dataset": run.label,
                "n_models": len(run.frozen_r2),
                "fz_r2_mean": float(run.frozen_r2.mean()),
                "fz_r2_min": float(run.frozen_r2.min()),
                "fz_r2_max": float(run.frozen_r2.max()),
                "ft_r2_mean": float(run.finetune_r2.mean()),
                "ft_r2_min": float(run.finetune_r2.min()),
                "ft_r2_max": float(run.finetune_r2.max()),
                "ft_band_trainable": float(ft_trainable.max() - ft_trainable.min()),
                "ft_sigma_between": float(ft_trainable.std(ddof=1)),
                "ft_pred_spearman_mean": float(run.finetune_spearman.loc[run.trainable].mean()),
                "n_diverged": int((ft_trainable < 0).sum()),
                "top1_frozen": str(run.frozen_r2.idxmax()),
                "top1_finetune": str(run.finetune_r2.idxmax()),
                "regret_at_1": run.regret_at_1,
            }
        )
    return pd.DataFrame(rows)


def _rho(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import spearmanr

    common = [m for m in a.index if m in b.index]
    return float(spearmanr(a.loc[common], b.loc[common]).correlation)


def rank_survival_table(runs: dict[str, DatasetRun]) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for dataset in ("wine", "housing"):
        full, tiny = runs[f"{dataset}_full"], runs[f"{dataset}_tiny"]
        rows.append(
            {
                "dataset": dataset,
                "tiny_fz_vs_full_fz": _rho(tiny.frozen_r2, full.frozen_r2),
                "tiny_fz_vs_full_ft": _rho(tiny.frozen_r2, full.finetune_r2),
                "tiny_ft_vs_full_ft": _rho(tiny.finetune_r2, full.finetune_r2),
                "tiny_fz_vs_tiny_ft": tiny.cross_pass_rho,
                "full_fz_vs_full_ft": full.cross_pass_rho,
                "tiny_regret_at_1": tiny.regret_at_1,
                "full_regret_at_1": full.regret_at_1,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    theme.apply_theme(plt, sns)
    return plt


def plot_dataset_bands(runs: dict[str, DatasetRun], out: Path) -> None:
    plt = _setup()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, run in enumerate(runs.values()):
        for offset, series, color, marker in (
            (-0.13, run.frozen_r2, theme.PROXY_COLOR, theme.PROXY_MARKER),
            (+0.13, run.finetune_r2, theme.REFERENCE_COLOR, theme.REFERENCE_MARKER),
        ):
            ax.scatter(
                [i + offset] * len(series),
                series,
                color=color,
                marker=marker,
                s=30,
                alpha=0.7,
                zorder=3,
            )
    ax.set_xticks(range(len(runs)), list(runs), fontsize=10)
    ax.set_ylabel("test r² (one point per model)")
    ax.axhline(0.0, color=theme.GRID, lw=1)
    ax.set_title("Per-model r² by dataset — frozen proxy (tan) vs finetune reference (burgundy)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_rank_survival(survival: pd.DataFrame, out: Path) -> None:
    import numpy as np

    plt = _setup()
    metrics = [
        ("tiny_fz_vs_full_fz", "tiny frozen ↔ full frozen"),
        ("tiny_ft_vs_full_ft", "tiny finetune ↔ full finetune"),
        ("tiny_fz_vs_tiny_ft", "tiny frozen ↔ tiny finetune"),
        ("full_fz_vs_full_ft", "full frozen ↔ full finetune"),
    ]
    colors = [
        theme.PASS_COLORS["frozen"],
        theme.SUBSTEP_COLORS[3],
        theme.SUBSTEP_COLORS[2],
        theme.STATUS_COLORS["ok"],
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(survival))
    width = 0.2
    for j, ((column, label), color) in enumerate(zip(metrics, colors, strict=True)):
        ax.bar(x + (j - 1.5) * width, survival[column], width, label=label, color=color)
    ax.set_xticks(x, list(survival["dataset"]))
    ax.axhline(0.0, color=theme.INK, lw=0.8)
    ax.set_ylabel("Spearman rho (16-model rankings)")
    ax.set_ylim(-0.35, 1.0)
    ax.set_title("Does the ranking survive a 2k-row budget?")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_housing_diagnosis(runs: dict[str, DatasetRun], out: Path) -> None:
    plt = _setup()
    housing, wine = runs["housing_full"], runs["wine_full"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        wine.finetune_r2,
        wine.finetune_spearman,
        color=theme.STATUS_COLORS["ok"],
        marker="^",
        s=32,
        alpha=0.55,
        label="wine_full finetune (healthy: rho tracks r²)",
    )
    ax.scatter(
        housing.frozen_r2,
        housing.frozen_spearman,
        color=theme.PROXY_COLOR,
        marker=theme.PROXY_MARKER,
        s=32,
        alpha=0.8,
        label="housing_full frozen",
    )
    ax.scatter(
        housing.finetune_r2,
        housing.finetune_spearman,
        color=theme.REFERENCE_COLOR,
        marker=theme.REFERENCE_MARKER,
        s=32,
        alpha=0.8,
        label="housing_full finetune",
    )
    ax.set_xlabel("test r²")
    ax.set_ylabel("prediction-vs-target Spearman rho (per model)")
    ax.set_xlim(-0.06, 1.0)
    ax.set_title("usa_housing: models rank prices well, but r² ≈ 0 — a long-tail target problem")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- report


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=Path("results/dataset_comparison"))
    for label, default in DEFAULT_DIRS.items():
        parser.add_argument(f"--{label.replace('_', '-')}", type=Path, default=default)
    args = parser.parse_args()
    dirs = {label: getattr(args, label) for label in DEFAULT_DIRS}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(dirs)
    comparison = comparison_table(runs)
    survival = rank_survival_table(runs)

    write_table(comparison, args.out_dir / "dataset_comparison")
    write_table(survival, args.out_dir / "rank_survival")
    plot_dataset_bands(runs, args.out_dir / "dataset_r2_bands.png")
    plot_rank_survival(survival, args.out_dir / "rank_survival.png")
    plot_housing_diagnosis(runs, args.out_dir / "housing_diagnosis.png")

    housing = runs["housing_full"]
    ft_tr = housing.finetune_r2.loc[housing.trainable]
    sp_tr = housing.finetune_spearman.loc[housing.trainable]
    headline = "\n".join(
        [
            f"- housing_full: finetune r² in [{ft_tr.min():.3f}, {ft_tr.max():.3f}] while "
            f"prediction-vs-target rho reaches {sp_tr.max():.2f} — ordering is learnable, the "
            "squared error is not (long-tailed target; affine z-scoring cannot fix this)",
            f"- wine rank survival at 2k rows: frozen↔frozen rho = "
            f"{survival.loc[survival.dataset == 'wine', 'tiny_fz_vs_full_fz'].iloc[0]:.2f}",
        ]
    )
    sections = [
        "# Dataset comparison report (fixed MLP_256 final recipe)\n",
        "## Headline\n\n" + headline + "\n",
        "## Per-dataset summary\n\n" + df_to_markdown(comparison),
        "## Rank survival (full pool, 16 models)\n\n" + df_to_markdown(survival),
    ]
    (args.out_dir / "DATASET_SUMMARY.md").write_text("\n".join(sections))
    print("\n\n".join(sections))
    print(f"wrote tables + figures to {args.out_dir}")


if __name__ == "__main__":
    main()

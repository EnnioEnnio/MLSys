"""Noise-floor report over the 5-seed full_eval repeats (METHOD.md "The noise floor", #67/#68).

`mlsys analyze` keys its per-head artifacts on the head label, so five repeats of the *same*
config (MLP_256noise1..5) would be treated as five different heads — semantically wrong for a
seed study. This side-script instead groups by run-id via the analysis loader and computes the
five statistics METHOD.md specifies, per pass (frozen / finetune):

1. noise floor      sigma_within  = sqrt(mean_m var_seeds(r2))
2. signal           sigma_between = std_m(mean_seeds(r2)), plus the raw band (max - min)
3. discriminativeness  sigma_between / sigma_within and ICC = s2_b / (s2_b + s2_w)
4. ranking stability   mean pairwise Spearman between the seeds' rankings
5. random-ranking null Monte-Carlo random permutations as the "proxy" over each seed's
   finetune scores -> regret@k null band, against the observed proxy regret curves

The finetune rows are also reported for the trainable subset only (model2vec candidates fall
back to their frozen score, `ref_skipped`; their stable pseudo-reference would otherwise
inflate reference-ranking stability).

Deterministic: fixed MC seed, sorted file discovery, no timestamps in outputs.

Usage:
    uv run python scripts/noise_report.py                      # results/full_eval_noise
    uv run python scripts/noise_report.py results/full_eval_noise --out-dir ... --seed 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mlsys.analysis import theme
from mlsys.analysis.loader import Triple, discover_triples, load_triple
from mlsys.analysis.regret_recompute import recompute_regret, triple_regret_df
from mlsys.analysis.tables import df_to_markdown, rank_spearman, write_table

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

DEFAULT_DIR = Path("results/full_eval_noise")
SEED_RE = re.compile(r"noise(\d+)$")


# --------------------------------------------------------------------------- loading


def load_seed_triples(directory: Path) -> dict[str, Triple]:
    """{seed label -> Triple}, ordered by seed number (labels like 'noise3')."""
    triples: dict[int, Triple] = {}
    for tf in discover_triples(directory):
        match = SEED_RE.search(tf.head)
        if match is None:
            raise ValueError(
                f"run-id {tf.run_id}: head {tf.head!r} has no trailing 'noise<N>' seed token"
            )
        num = int(match.group(1))
        if num in triples:
            raise ValueError(f"duplicate seed noise{num} in {directory}")
        triples[num] = load_triple(tf)
    if len(triples) < 2:
        raise FileNotFoundError(f"need >=2 seed triples in {directory}, found {len(triples)}")
    return {f"noise{num}": triples[num] for num in sorted(triples)}


def r2_matrix(triples: dict[str, Triple], which: str) -> pd.DataFrame:
    """models x seeds r2 frame ('frozen' -> .proxy, 'finetune' -> .reference)."""
    import pandas as pd

    frames = {
        seed: (t.proxy if which == "frozen" else t.reference).set_index("model")["r2"]
        for seed, t in triples.items()
    }
    matrix = pd.DataFrame(frames)
    if matrix.isna().any().any():
        raise ValueError(f"model sets differ across seeds in the {which} pass")
    return matrix


def trainable_models(triples: dict[str, Triple]) -> list[str]:
    """Models actually fine-tuned in every seed (excludes model2vec `ref_skipped` fallbacks)."""
    first, *rest = triples.values()
    skipped = {m for m, flag in first.ref_skipped.items() if flag}
    for t in rest:
        if {m for m, flag in t.ref_skipped.items() if flag} != skipped:
            raise ValueError("ref_skipped set differs across seeds")
    return [m for m in first.models if m not in skipped]


# --------------------------------------------------------------------------- statistics


def _variance_stats(matrix: pd.DataFrame, label: str) -> dict[str, object]:
    """The METHOD.md statistics 1-3 for one models x seeds r2 matrix."""
    within_var = matrix.var(axis=1, ddof=1)  # per-model variance across seeds
    means = matrix.mean(axis=1)
    sigma_within = float((within_var.mean()) ** 0.5)
    sigma_between = float(means.std(ddof=1))
    return {
        "pass": label,
        "n_models": len(matrix),
        "n_seeds": int(matrix.shape[1]),
        "sigma_within": sigma_within,
        "max_within_std": float(within_var.max() ** 0.5),
        "sigma_between": sigma_between,
        "ratio": sigma_between / sigma_within,
        "icc": sigma_between**2 / (sigma_between**2 + sigma_within**2),
        "band_min": float(means.min()),
        "band_max": float(means.max()),
        "band_width": float(means.max() - means.min()),
    }


def noise_stats_table(
    frozen: pd.DataFrame, finetune: pd.DataFrame, trainable: list[str]
) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        [
            _variance_stats(frozen, "frozen"),
            _variance_stats(finetune, "finetune"),
            _variance_stats(finetune.loc[trainable], "finetune (trainable)"),
        ]
    )


def per_model_table(
    frozen: pd.DataFrame, finetune: pd.DataFrame, trainable: list[str]
) -> pd.DataFrame:
    """Per model: mean/std r2 and rank spread (1 = best within a seed) for both passes."""
    import pandas as pd

    fz_ranks = frozen.rank(ascending=False, method="min")
    ft_ranks = finetune.rank(ascending=False, method="min")
    table = pd.DataFrame(
        {
            "frozen_mean": frozen.mean(axis=1),
            "frozen_std": frozen.std(axis=1, ddof=1),
            "finetune_mean": finetune.mean(axis=1),
            "finetune_std": finetune.std(axis=1, ddof=1),
            "fz_rank_min": fz_ranks.min(axis=1).astype(int),
            "fz_rank_max": fz_ranks.max(axis=1).astype(int),
            "ft_rank_min": ft_ranks.min(axis=1).astype(int),
            "ft_rank_max": ft_ranks.max(axis=1).astype(int),
            "finetune_skipped": [m not in trainable for m in frozen.index],
        }
    )
    table = table.sort_values("finetune_mean", ascending=False)
    return table.rename_axis("model").reset_index()


def _pairwise_spearman(matrix: pd.DataFrame) -> pd.DataFrame:
    """seeds x seeds Spearman over the per-model r2 rankings."""
    import pandas as pd
    from scipy.stats import spearmanr

    seeds = list(matrix.columns)
    corr = pd.DataFrame(index=seeds, columns=seeds, dtype=float)
    for a in seeds:
        for b in seeds:
            corr.loc[a, b] = float(spearmanr(matrix[a], matrix[b]).correlation)
    return corr


def _offdiag(corr: pd.DataFrame) -> list[float]:
    seeds = list(corr.columns)
    return [float(corr.loc[a, b]) for i, a in enumerate(seeds) for b in seeds[i + 1 :]]


def rank_stability_tables(
    frozen: pd.DataFrame,
    finetune: pd.DataFrame,
    trainable: list[str],
    triples: dict[str, Triple],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(summary, frozen seeds x seeds matrix, finetune seeds x seeds matrix)."""
    import pandas as pd

    fz_corr = _pairwise_spearman(frozen)
    ft_corr = _pairwise_spearman(finetune)
    ft_corr_trainable = _pairwise_spearman(finetune.loc[trainable])
    cross = [rank_spearman(t) for t in triples.values()]

    def row(label: str, values: list[float]) -> dict[str, object]:
        import numpy as np

        arr = np.asarray(values)
        return {
            "comparison": label,
            "mean_rho": float(arr.mean()),
            "min_rho": float(arr.min()),
            "max_rho": float(arr.max()),
            "n_pairs": len(values),
        }

    summary = pd.DataFrame(
        [
            row("frozen vs frozen (seed pairs)", _offdiag(fz_corr)),
            row("finetune vs finetune (seed pairs)", _offdiag(ft_corr)),
            row("finetune vs finetune, trainable only", _offdiag(ft_corr_trainable)),
            row("frozen vs finetune (within seed)", cross),
        ]
    )
    return summary, fz_corr, ft_corr


# --------------------------------------------------------------------------- regret vs null


def observed_regret(triples: dict[str, Triple]) -> pd.DataFrame:
    """budgets x seeds absolute-regret frame from the committed (or recomputed) curves."""
    import pandas as pd

    curves = {
        seed: triple_regret_df(t).set_index("budget")["regret"] for seed, t in triples.items()
    }
    return pd.DataFrame(curves)


def observed_regret_trainable(triples: dict[str, Triple], trainable: list[str]) -> pd.DataFrame:
    """Regret curves with the non-finetunable (model2vec fallback) candidates removed.

    Attribution diagnostic: the frozen proxy tends to rank the static embedders on top, but
    their "reference" is capped at the frozen score — this isolates how much of the full-pool
    regret is that pool-heterogeneity effect rather than ranking error among real finetunes.
    """
    import pandas as pd

    curves = {}
    for seed, t in triples.items():
        proxy = t.proxy[t.proxy["model"].isin(trainable)]
        reference = t.reference[t.reference["model"].isin(trainable)]
        curve = recompute_regret(proxy, reference)
        curves[seed] = curve.set_index("budget")["regret"]
    return pd.DataFrame(curves)


def null_regret_samples(finetune: pd.DataFrame, n_permutations: int, seed: int) -> np.ndarray:
    """(n_seeds * n_permutations, n_budgets) regret samples for random 'proxy' rankings."""
    import numpy as np

    rng = np.random.default_rng(seed)
    samples = []
    for column in finetune.columns:
        scores = finetune[column].to_numpy()
        best = scores.max()
        perms = rng.permuted(
            np.tile(np.arange(len(scores)), (n_permutations, 1)), axis=1
        )  # (P, M) random orderings
        shortlist_best = np.maximum.accumulate(scores[perms], axis=1)
        samples.append(best - shortlist_best)
    return np.concatenate(samples, axis=0)


def regret_null_table(observed: pd.DataFrame, null: np.ndarray) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "budget": observed.index,
            "obs_mean": observed.mean(axis=1).to_numpy(),
            "obs_min": observed.min(axis=1).to_numpy(),
            "obs_max": observed.max(axis=1).to_numpy(),
            "null_mean": null.mean(axis=0),
            "null_p05": np.quantile(null, 0.05, axis=0),
            "null_p95": np.quantile(null, 0.95, axis=0),
        }
    )


# --------------------------------------------------------------------------- figures

#: Run-level print-preset switch, set from ``main()`` by ``--paper``. Mirrors ``plots.PAPER``.
PAPER: bool = False


def _setup():
    """Lazy import + themed style. Returns the (plt, sns) pair. Mirrors ``plots.py:_setup``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    theme.apply_theme(plt, sns, paper=PAPER)
    return plt, sns


def _save(fig, out: Path) -> None:
    """Write ``out``, retargeted to .pdf in paper mode. See ``plots.py:_save`` for the why."""
    import matplotlib.pyplot as plt

    if theme.is_paper():
        fig.savefig(out.with_suffix(".pdf"))
    else:
        fig.tight_layout()
        fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_seed_variability(
    frozen: pd.DataFrame, finetune: pd.DataFrame, trainable: list[str], out: Path
) -> None:
    plt, _ = _setup()
    paper = theme.is_paper()
    order = finetune.mean(axis=1).sort_values(ascending=False).index
    # Body figure. Goes to full \textwidth in paper mode: 16 rotated model names at 9 pt need
    # ~1.4 in of vertical space, which only pays for itself if the width grows too — and the
    # figure was being included at 0.75\textwidth anyway, so the width was already there.
    fig, ax = plt.subplots(
        figsize=theme.size((12, 5.5), (theme.TEXT_W_IN, 2.6)),
        constrained_layout=paper,
    )
    for i, model in enumerate(order):
        # The colour key lived in the title, which paper mode suppresses — so in paper mode it
        # becomes a real legend (labelled once, on the first model only).
        keyed = paper and i == 0
        ax.scatter(
            [i] * frozen.shape[1],
            frozen.loc[model],
            color=theme.PROXY_COLOR,
            marker=theme.PROXY_MARKER,
            s=14 if paper else 28,
            alpha=0.75,
            zorder=3,
            label="frozen proxy" if keyed else None,
        )
        ax.scatter(
            [i] * finetune.shape[1],
            finetune.loc[model],
            color=theme.REFERENCE_COLOR,
            marker=theme.REFERENCE_MARKER,
            s=14 if paper else 28,
            alpha=0.75,
            zorder=3,
            label="finetune reference" if keyed else None,
        )
    labels = [m if m in trainable else f"{m} *" for m in order]
    ax.set_xticks(range(len(order)), labels, rotation=45, ha="right")
    ax.set_ylabel("test r²")
    if paper:
        ax.legend(loc="lower left", ncol=2, markerscale=1.6)
    theme.title(
        ax,
        f"r² across {frozen.shape[1]} seeds — frozen proxy (tan) vs finetune reference "
        "(burgundy); * = model2vec frozen fallback",
    )
    _save(fig, out)


def plot_rank_stability(fz_corr: pd.DataFrame, ft_corr: pd.DataFrame, out: Path) -> None:
    import numpy as np

    plt, sns = _setup()
    paper = theme.is_paper()
    # Appendix figure: full \textwidth, which also reclaims the 0.75x it was included at.
    fig, axes = plt.subplots(
        1, 2, figsize=theme.size((11, 4.4), (theme.TEXT_W_IN, 2.8)), constrained_layout=paper
    )
    vmin = min(0.0, float(fz_corr.min().min()), float(ft_corr.min().min()))
    for ax, corr, label in ((axes[0], fz_corr, "frozen"), (axes[1], ft_corr, "finetune")):
        sns.heatmap(
            corr.astype(float),
            annot=True,
            fmt=".2f",
            cmap=theme.get_cmap_r2(),
            vmin=vmin,
            vmax=1.0,
            cbar=False,
            square=True,
            ax=ax,
            annot_kws=theme.annot_kws(),
        )
        mean_off = float(np.mean(_offdiag(corr)))
        # The per-panel title is the only way to tell the two matrices apart, so unlike every
        # other title in the paper preset it has to stay — it is a label, not a description.
        ax.set_title(
            f"{label} (mean {mean_off:.2f})"
            if paper
            else f"{label} ranking, seed x seed rho (mean {mean_off:.2f})"
        )
    if not paper:
        fig.suptitle("Ranking stability across seeds (Spearman rho)")
    _save(fig, out)


def plot_regret_vs_null(table: pd.DataFrame, out: Path) -> None:
    plt, _ = _setup()
    paper = theme.is_paper()
    # Body figure (page-limited): hold the 3.34 x 2.00 in footprint it already occupies, and
    # buy the room for 9 pt text by dropping the title and thinning the tick set instead.
    fig, ax = plt.subplots(
        figsize=theme.size((8, 4.8), (theme.COLUMN_W_IN, 2.0)),
        constrained_layout=paper,
    )
    budgets = table["budget"]
    null_color = theme.STATUS_COLORS["ok"]
    ax.fill_between(
        budgets, table["null_p05"], table["null_p95"], color=null_color, alpha=0.20, lw=0
    )
    ax.plot(
        budgets,
        table["null_mean"],
        color=null_color,
        ls="--",
        # The caption already spells out "mean, 5-95% band", so paper mode drops it from the
        # legend — at 9 pt those parentheticals cost a third of the plot area.
        label="random-ranking null" if paper else "random-ranking null (mean, 5-95%)",
    )
    ax.fill_between(
        budgets, table["obs_min"], table["obs_max"], color=theme.REFERENCE_COLOR, alpha=0.25, lw=0
    )
    ax.plot(
        budgets,
        table["obs_mean"],
        color=theme.REFERENCE_COLOR,
        marker="o",
        ms=2.5 if paper else 4,
        label="frozen proxy" if paper else "frozen proxy (mean, min-max over seeds)",
    )
    ax.axhline(0.0, color=theme.GRID, lw=1)
    ax.set_xlabel("budget B (models fine-tuned)")
    ax.set_ylabel("regret@B (r²)")
    # 16 tick labels at 9 pt need ~2.9 in — more than the axes has. Every 3rd budget keeps the
    # axis readable and still lands on both endpoints (set() drops the duplicate when the
    # stride already hits the last budget).
    ticks = list(budgets)
    ax.set_xticks(sorted({*ticks[::3], ticks[-1]}) if paper else ticks)
    theme.title(ax, "Observed proxy regret vs the random-ranking null")
    ax.legend()
    _save(fig, out)


# --------------------------------------------------------------------------- report


def top1_counts(matrix: pd.DataFrame) -> str:
    counts = matrix.idxmax().value_counts()
    return ", ".join(f"{model} {n}/{matrix.shape[1]}" for model, n in counts.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("directory", type=Path, nargs="?", default=DEFAULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="default <directory>/analysis")
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0, help="Monte-Carlo RNG seed")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="print preset: 9 pt text, no in-plot titles, exact column/text width, vector PDF",
    )
    args = parser.parse_args()
    global PAPER
    PAPER = args.paper
    out_dir = args.out_dir or args.directory / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = load_seed_triples(args.directory)
    frozen = r2_matrix(triples, "frozen")
    finetune = r2_matrix(triples, "finetune")
    trainable = trainable_models(triples)
    n_diverged = int((finetune.loc[trainable] < 0).sum().sum())

    stats = noise_stats_table(frozen, finetune, trainable)
    per_model = per_model_table(frozen, finetune, trainable)
    stability, fz_corr, ft_corr = rank_stability_tables(frozen, finetune, trainable, triples)
    observed = observed_regret(triples)
    null = null_regret_samples(finetune, args.n_permutations, args.seed)
    regret_table = regret_null_table(observed, null)
    observed_tr = observed_regret_trainable(triples, trainable)
    null_tr = null_regret_samples(finetune.loc[trainable], args.n_permutations, args.seed)
    regret_table_tr = regret_null_table(observed_tr, null_tr)

    write_table(stats, out_dir / "noise_stats")
    write_table(per_model, out_dir / "per_model_seed_stats")
    write_table(stability, out_dir / "rank_stability")
    write_table(fz_corr.rename_axis("seed").reset_index(), out_dir / "rank_stability_frozen")
    write_table(ft_corr.rename_axis("seed").reset_index(), out_dir / "rank_stability_finetune")
    write_table(regret_table, out_dir / "regret_null")
    write_table(regret_table_tr, out_dir / "regret_null_trainable")

    plot_seed_variability(frozen, finetune, trainable, out_dir / "seed_variability.png")
    plot_rank_stability(fz_corr, ft_corr, out_dir / "rank_stability.png")
    plot_regret_vs_null(regret_table, out_dir / "regret_vs_null.png")

    at1 = regret_table.iloc[0]
    at1_tr = regret_table_tr.iloc[0]
    verdict = "\n".join(
        [
            f"- seeds: {frozen.shape[1]}; pool: {len(frozen)} models "
            f"({len(trainable)} trainable); diverged finetunes (r² < 0): {n_diverged}",
            f"- frozen top-1 across seeds: {top1_counts(frozen)}",
            f"- finetune top-1 across seeds: {top1_counts(finetune)}",
            f"- regret@1: observed mean {at1['obs_mean']:.4f} "
            f"[{at1['obs_min']:.4f}, {at1['obs_max']:.4f}] vs "
            f"null {at1['null_mean']:.4f} [{at1['null_p05']:.4f}, {at1['null_p95']:.4f}]",
            f"- regret@1, trainable pool only: observed mean {at1_tr['obs_mean']:.4f} "
            f"[{at1_tr['obs_min']:.4f}, {at1_tr['obs_max']:.4f}] vs "
            f"null {at1_tr['null_mean']:.4f} "
            f"[{at1_tr['null_p05']:.4f}, {at1_tr['null_p95']:.4f}]",
        ]
    )

    sections = [
        f"# Noise-floor report ({args.directory})\n",
        "## Headline\n\n" + verdict + "\n",
        "## Noise statistics (METHOD.md 1-3)\n\n" + df_to_markdown(stats),
        "## Ranking stability (METHOD.md 4)\n\n" + df_to_markdown(stability),
        "## Regret vs random-ranking null (METHOD.md 5)\n\n" + df_to_markdown(regret_table),
        "## Regret vs null, non-finetunable candidates removed\n\n"
        + df_to_markdown(regret_table_tr),
        "## Per-model spread across seeds\n\n" + df_to_markdown(per_model),
    ]
    (out_dir / "NOISE_SUMMARY.md").write_text("\n".join(sections))

    print("\n\n".join(sections))
    print(f"wrote tables + figures to {out_dir}")


if __name__ == "__main__":
    main()

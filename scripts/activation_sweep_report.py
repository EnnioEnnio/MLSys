"""Compare frozen-only CSVs from an activation-function sweep (see docs/head-activation.md).

`mlsys analyze` needs a frozen+finetune pair per run-id to compute regret; a same-head,
frozen-only sweep across activations (relu/gelu/tanh/silu at one fixed hidden width) has no
such pair, so this is a standalone comparison instead of a pipeline feature:

- per activation: mean/std r2, mean spearman, proxy top-1 model
- pairwise Spearman correlation between activations' per-model r2 rankings, to answer the
  question that actually matters (see docs/head-activation.md "Why"): does the frozen-proxy
  ranking move when the activation changes, or does relu stay the default?

Each input CSV is expected to be one activation's frozen-pass results, one row per model,
with at least "model", "r2", "spearman" columns (i.e. a `*_frozen.csv` downloaded from W&B).
The activation is recovered from the filename: exactly one of --activations must appear as an
underscore-delimited token in the stem (e.g. "..._relu.csv" or "relu_..._frozen.csv" both work;
"relugelu.csv" or a name matching two activations does not).

Usage:
    uv run python scripts/activation_sweep_report.py results/activation_sweep
    uv run python scripts/activation_sweep_report.py results/activation_sweep \
        --activations relu,gelu,tanh,silu
    uv run python scripts/activation_sweep_report.py results/activation_sweep --update-doc
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_ACTIVATIONS = ("relu", "gelu", "tanh", "silu")
DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "head-activation.md"


def _match_activation(stem: str, activations: tuple[str, ...]) -> str:
    tokens = set(stem.split("_"))
    hits = [a for a in activations if a in tokens]
    if len(hits) != 1:
        raise ValueError(
            f"could not uniquely identify activation in filename stem {stem!r}: expected "
            f"exactly one of {activations} as an underscore-delimited token, found {hits}"
        )
    return hits[0]


def load_csvs(directory: Path, activations: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    import pandas as pd

    frames: dict[str, pd.DataFrame] = {}
    csvs = sorted(directory.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no CSVs found in {directory}")
    for csv in csvs:
        activation = _match_activation(csv.stem, activations)
        if activation in frames:
            raise ValueError(
                f"activation {activation!r} matched more than one CSV in {directory} "
                f"(already have one from a previous file)"
            )
        frames[activation] = pd.read_csv(csv)
    missing = [a for a in activations if a not in frames]
    if missing:
        raise FileNotFoundError(f"no CSV found for activation(s) {missing} in {directory}")
    return frames


def summary_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for activation, df in frames.items():
        top = df.loc[df["r2"].idxmax()]
        rows.append(
            {
                "activation": activation,
                "mean_r2": df["r2"].mean(),
                "std_r2": df["r2"].std(),
                "mean_spearman": df["spearman"].mean(),
                "proxy_top1": top["model"],
            }
        )
    return pd.DataFrame(rows).set_index("activation")


def per_model_ranking(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Full model x activation table: r2 per activation + rank per activation (1 = best).

    Sorted by mean rank ascending, so the top row is the model that ranks highest on
    average across activations.
    """
    import pandas as pd

    r2_by_model = pd.DataFrame({a: df.set_index("model")["r2"] for a, df in frames.items()})
    ranks = r2_by_model.rank(ascending=False, method="min").astype(int)
    ranks.columns = [f"{a}_rank" for a in ranks.columns]
    table = pd.concat([r2_by_model.add_suffix("_r2"), ranks], axis=1)
    table["mean_rank"] = ranks.mean(axis=1)
    table = table.sort_values("mean_rank")
    interleaved = [c for a in r2_by_model.columns for c in (f"{a}_r2", f"{a}_rank")]
    return table[[*interleaved, "mean_rank"]]


def rank_agreement(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise Spearman correlation between activations' per-model r2 rankings."""
    import pandas as pd
    from scipy.stats import spearmanr

    r2_by_model = pd.DataFrame({a: df.set_index("model")["r2"] for a, df in frames.items()})
    activations = list(r2_by_model.columns)
    corr = pd.DataFrame(index=activations, columns=activations, dtype=float)
    for a in activations:
        for b in activations:
            corr.loc[a, b] = spearmanr(r2_by_model[a], r2_by_model[b]).correlation
    return corr


def _ranking_move_note(summary: pd.DataFrame, corr: pd.DataFrame) -> str:
    activations = list(corr.columns)
    off_diag = [corr.loc[a, b] for a in activations for b in activations if a != b]
    min_rho, max_rho = min(off_diag), max(off_diag)
    top1s = summary["proxy_top1"]
    top1_stable = top1s.nunique() == 1
    if top1_stable:
        top1_note = f"proxy top-1 is **{top1s.iloc[0]}** for every activation."
    else:
        pairs = ", ".join(f"{a}={m}" for a, m in top1s.items())
        top1_note = f"proxy top-1 **changes** across activations: {pairs}."
    return (
        f"pairwise Spearman rho over per-model r2 ranks ranges "
        f"[{min_rho:.3f}, {max_rho:.3f}]; {top1_note}"
    )


def _to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if formatted[col].dtype.kind in "fc":
            formatted[col] = formatted[col].map(float_format.format)
    index_name = str(formatted.index.name) if formatted.index.name else ""
    columns = [str(c) for c in formatted.columns]
    header = "| " + " | ".join([index_name, *columns]) + " |"
    sep = "| " + " | ".join(["---"] * (len(formatted.columns) + 1)) + " |"
    lines = [header, sep]
    for idx, row in formatted.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def _patch_doc(summary: pd.DataFrame, corr: pd.DataFrame, note: str) -> None:
    text = DOC_PATH.read_text()

    table_re = re.compile(
        r"(\| activation \| mean r² \| std \| mean spearman \| proxy top-1 \|\n"
        r"\| --- \| --- \| --- \| --- \| --- \|\n)"
        r"(?:\|.*\|\n)+"
    )
    new_rows = "".join(
        f"| {activation} | {row.mean_r2:.4f} | {row.std_r2:.4f} | "
        f"{row.mean_spearman:.4f} | {row.proxy_top1} |\n"
        for activation, row in summary.iterrows()
    )
    text, n = table_re.subn(lambda m: m.group(1) + new_rows, text)
    if n != 1:
        raise ValueError("could not find the sweep table in docs/head-activation.md to patch")

    move_re = re.compile(r"\*\*Does the ranking move\?\*\*.*?(?=\n\n)", re.DOTALL)
    text, n = move_re.subn(f"**Does the ranking move?** {note}", text)
    if n != 1:
        raise ValueError("could not find the 'Does the ranking move?' line to patch")

    DOC_PATH.write_text(text)


def _ranks_markdown(frames: dict[str, pd.DataFrame]) -> str:
    ranking = per_model_ranking(frames)
    formatted = ranking.copy()
    for col in formatted.columns:
        if col.endswith("_r2") or col == "mean_rank":
            formatted[col] = formatted[col].map("{:.4f}".format)
    return _to_markdown(formatted, float_format="{}")


def _render_report(
    directory: Path,
    frames: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    corr: pd.DataFrame,
    note: str,
    *,
    include_ranks: bool,
) -> str:
    parts = [f"# Activation sweep report ({directory})\n", "## Summary\n", _to_markdown(summary)]
    if include_ranks:
        parts += [
            "\n## Per-model ranking (1 = best r2 within that activation)\n",
            _ranks_markdown(frames),
        ]
    parts += [
        "\n## Pairwise Spearman (model r2 rank agreement)\n",
        _to_markdown(corr, float_format="{:.3f}"),
        f"\n{note}\n",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="folder of per-activation frozen CSVs")
    parser.add_argument(
        "--activations",
        default=",".join(DEFAULT_ACTIVATIONS),
        help=(
            "comma-separated activation names to look for "
            f"(default: {','.join(DEFAULT_ACTIVATIONS)})"
        ),
    )
    parser.add_argument(
        "--update-doc",
        action="store_true",
        help="also patch the sweep table + ranking-move line in docs/head-activation.md",
    )
    parser.add_argument(
        "--ranks",
        action="store_true",
        help="also print the full model x activation r2/rank table",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="write the full report (summary + per-model ranking + rank agreement) as markdown "
        "to this path, e.g. results/activation_sweep/report.md",
    )
    args = parser.parse_args()

    activations = tuple(a.strip() for a in args.activations.split(","))
    frames = load_csvs(args.directory, activations)
    summary = summary_table(frames)
    corr = rank_agreement(frames)
    note = _ranking_move_note(summary, corr)

    print("## Summary\n")
    print(_to_markdown(summary))

    if args.ranks:
        print("\n## Per-model ranking (1 = best r2 within that activation)\n")
        print(_ranks_markdown(frames))
    print("\n## Pairwise Spearman (model r2 rank agreement)\n")
    print(_to_markdown(corr, float_format="{:.3f}"))
    print(f"\n{note}\n")

    if args.update_doc:
        _patch_doc(summary, corr, note)
        print(f"updated {DOC_PATH.relative_to(DOC_PATH.parent.parent)}")

    if args.out:
        report = _render_report(args.directory, frames, summary, corr, note, include_ranks=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

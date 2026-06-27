"""Table builders + a tabulate-free markdown writer.

Two families of artifact:

* **per-triple** — one head config: a per-model quality+cost table and a one-row summary.
* **cross-run** — over all surviving heads: headxmodel r² matrices and a per-head summary.

Every builder returns a pandas DataFrame; :func:`write_table` dumps it as both ``.csv`` and a
GitHub-flavoured ``.md`` (hand-rolled so the only runtime dep stays pandas — no ``tabulate``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mlsys.analysis.regret_recompute import recompute_regret

if TYPE_CHECKING:
    import pandas as pd

    from mlsys.analysis.loader import Triple


@dataclass(frozen=True)
class TripleSummary:
    """Headline numbers for one head config (drives the per-head comparison table + section 6)."""

    head: str
    n_models: int
    best_frozen_model: str
    best_frozen_r2: float
    best_finetune_model: str
    best_finetune_r2: float
    regret_at_1: float
    normalized_regret_at_1: float
    budget_to_zero: int
    regret_auc: float
    rank_spearman: float
    n_diverged: int
    n_finetune_skipped: int


def _fmt(value: object) -> str:
    """Render a cell: floats to 4 sig-ish figures, bools as ✓/·, everything else as str."""
    if isinstance(value, bool):
        return "✓" if value else "·"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.4f}"
    return "" if value is None else str(value)


def df_to_markdown(df: pd.DataFrame) -> str:
    """A minimal GitHub-flavoured markdown table (avoids the ``tabulate`` optional dep)."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows]) + "\n"


def write_table(df: pd.DataFrame, stem: str | Path) -> tuple[Path, Path]:
    """Write ``<stem>.csv`` and ``<stem>.md``; return both paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    md_path.write_text(df_to_markdown(df))
    return csv_path, md_path


# --------------------------------------------------------------------------- per-triple


def per_triple_table(triple: Triple) -> pd.DataFrame:
    """Per-model frozen/finetune quality (r²,mse,mae,spearman) + Δ, flags, and cost columns.

    Cost columns are labelled by pass: in the **frozen** pass cost splits across
    ``inference_s`` (encode) and ``train_head_s`` (head fit); in the **finetune** pass
    inference is fused into the joint loop so ``finetune_train_head_s`` is the end-to-end
    finetune cost (``inference_s == 0``). See RQ2 framing in DATAPLAN / CLAUDE.md.
    """
    import pandas as pd

    fz = triple.frozen.set_index("model")
    ft = triple.finetune.set_index("model")
    rows = []
    for model in triple.models:
        fzr = fz.loc[model]
        ftr = ft.loc[model] if model in ft.index else None
        ft_r2 = float(ftr["r2"]) if ftr is not None else float("nan")
        rows.append(
            {
                "model": model,
                "frozen_r2": float(fzr["r2"]),
                "finetune_r2": ft_r2,
                "delta_r2": ft_r2 - float(fzr["r2"]),
                "frozen_mse": float(fzr["mse"]),
                "finetune_mse": float(ftr["mse"]) if ftr is not None else float("nan"),
                "frozen_mae": float(fzr["mae"]),
                "finetune_mae": float(ftr["mae"]) if ftr is not None else float("nan"),
                "frozen_spearman": float(fzr["spearman"]),
                "finetune_spearman": float(ftr["spearman"]) if ftr is not None else float("nan"),
                "finetune_skipped": triple.finetune_skipped.get(model, False),
                "diverged": triple.diverged.get(model, False),
                "finetune_epochs": int(ftr["epochs_run"]) if ftr is not None else 0,
                "frozen_inference_s": float(fzr.get("inference_s", float("nan"))),
                "frozen_train_head_s": float(fzr.get("train_head_s", float("nan"))),
                "finetune_train_head_s": (
                    float(ftr["train_head_s"]) if ftr is not None else float("nan")
                ),
                "frozen_peak_gpu_mem_mb": float(fzr.get("peak_gpu_mem_mb", float("nan"))),
                "finetune_peak_gpu_mem_mb": (
                    float(ftr["peak_gpu_mem_mb"]) if ftr is not None else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _triple_regret_df(triple: Triple) -> pd.DataFrame:
    """The triple's regret curve — the committed one if present, else recomputed."""
    if triple.regret is not None:
        return triple.regret
    return recompute_regret(triple.frozen, triple.finetune)


def rank_spearman(triple: Triple) -> float:
    """Spearman(frozen r², finetune r²) across models — does the cheap proxy rank like truth?"""
    from scipy.stats import spearmanr

    fz = triple.frozen.set_index("model")["r2"]
    ft = triple.finetune.set_index("model")["r2"]
    common = [m for m in triple.models if m in ft.index]
    if len(common) < 2:
        return float("nan")
    rho, _ = spearmanr([fz[m] for m in common], [ft[m] for m in common])
    return float(rho)


def budget_to_zero(regret_df: pd.DataFrame) -> int:
    """Smallest budget B at which regret hits 0. Always defined (regret=0 at B=|M|)."""
    zero = regret_df.loc[regret_df["regret"] <= 0, "budget"]
    return int(zero.min()) if not zero.empty else int(regret_df["budget"].max())


def per_triple_summary(triple: Triple) -> TripleSummary:
    """Headline numbers for one head: best models, regret@1, budget-to-zero, proxy rank-rho."""
    fz = triple.frozen.set_index("model")["r2"]
    ft = triple.finetune.set_index("model")["r2"]
    regret_df = _triple_regret_df(triple)
    first = regret_df.iloc[0]
    return TripleSummary(
        head=triple.head,
        n_models=len(triple.models),
        best_frozen_model=str(fz.idxmax()),
        best_frozen_r2=float(fz.max()),
        best_finetune_model=str(ft.idxmax()),
        best_finetune_r2=float(ft.max()),
        regret_at_1=float(first["regret"]),
        normalized_regret_at_1=float(first["normalized_regret"]),
        budget_to_zero=budget_to_zero(regret_df),
        regret_auc=float(regret_df["regret"].mean()),
        rank_spearman=rank_spearman(triple),
        n_diverged=int(sum(triple.diverged.values())),
        n_finetune_skipped=int(sum(triple.finetune_skipped.values())),
    )


# --------------------------------------------------------------------------- cross-run


def head_model_r2_matrix(triples: list[Triple], kind: str) -> pd.DataFrame:
    """model x head r² matrix for ``kind`` in {"frozen","finetune"}; index = model."""
    import pandas as pd

    series = {}
    for t in triples:
        frame = t.frozen if kind == "frozen" else t.finetune
        series[t.head] = frame.set_index("model")["r2"]
    matrix = pd.DataFrame(series)
    matrix.index.name = "model"
    return matrix.reset_index()


def divergence_matrix(triples: list[Triple]) -> pd.DataFrame:
    """model x head boolean matrix of ``diverged`` (finetune r² < 0); index = model."""
    import pandas as pd

    series = {t.head: pd.Series(t.diverged) for t in triples}
    matrix = pd.DataFrame(series)
    matrix.index.name = "model"
    return matrix.reset_index()


def per_head_summary_table(triples: list[Triple]) -> pd.DataFrame:
    """One summary row per head (capacity order) — the cross-head comparison table."""
    import pandas as pd

    return pd.DataFrame([asdict(per_triple_summary(t)) for t in triples])


def diverged_models_table(triples: list[Triple]) -> pd.DataFrame:
    """Every model that diverged in *any* head: frozen→finetune r² + finetune rho, per head.

    Pins the "rank preserved (high Spearman), scale broken (negative r²)" story into a
    templated table so the narrative is written over given numbers, not re-derived from plots.
    """
    import pandas as pd

    diverged_any = sorted({m for t in triples for m, d in t.diverged.items() if d})
    rows = []
    for model in diverged_any:
        row: dict[str, object] = {"model": model}
        for t in triples:
            fz = t.frozen.set_index("model")["r2"]
            ftf = t.finetune.set_index("model")
            row[f"{t.head}_frozen_r2"] = float(fz[model]) if model in fz.index else float("nan")
            if model in ftf.index:
                row[f"{t.head}_finetune_r2"] = float(ftf.loc[model, "r2"])
                row[f"{t.head}_finetune_spearman"] = float(ftf.loc[model, "spearman"])
            else:
                row[f"{t.head}_finetune_r2"] = float("nan")
                row[f"{t.head}_finetune_spearman"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def cost_table(triples: list[Triple]) -> pd.DataFrame:
    """Per-(head,model) RQ2 cost: frozen total, finetune total, peak mem, epochs.

    Frozen total = ``inference_s + train_head_s`` (encode + head fit); finetune total =
    ``train_head_s`` (inference fused in). Surfaces the ~10x model2vec-vs-transformer spread.
    """
    import pandas as pd

    rows = []
    for t in triples:
        fz = t.frozen.set_index("model")
        ft = t.finetune.set_index("model")
        for model in t.models:
            fzr = fz.loc[model]
            ftr = ft.loc[model] if model in ft.index else None
            frozen_total = float(fzr.get("inference_s", 0.0)) + float(fzr.get("train_head_s", 0.0))
            rows.append(
                {
                    "head": t.head,
                    "model": model,
                    "frozen_total_s": frozen_total,
                    "finetune_total_s": (
                        float(ftr["train_head_s"]) if ftr is not None else float("nan")
                    ),
                    "frozen_peak_gpu_mem_mb": float(fzr.get("peak_gpu_mem_mb", float("nan"))),
                    "finetune_peak_gpu_mem_mb": (
                        float(ftr["peak_gpu_mem_mb"]) if ftr is not None else float("nan")
                    ),
                    "finetune_epochs": int(ftr["epochs_run"]) if ftr is not None else 0,
                    "finetune_skipped": t.finetune_skipped.get(model, False),
                }
            )
    return pd.DataFrame(rows)

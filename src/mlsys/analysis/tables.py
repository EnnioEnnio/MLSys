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

from mlsys.analysis.regret_recompute import triple_regret_df

if TYPE_CHECKING:
    import pandas as pd

    from mlsys.analysis.loader import Triple


@dataclass(frozen=True)
class TripleSummary:
    """Headline numbers for one head config (drives the per-head comparison table + section 6)."""

    head: str
    n_models: int
    best_proxy_model: str
    best_proxy_r2: float
    best_gt_model: str
    best_gt_r2: float
    regret_at_1: float
    normalized_regret_at_1: float
    budget_to_zero: int
    regret_auc: float
    rank_spearman: float
    n_diverged: int
    n_gt_skipped: int


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
    """Per-model proxy/gt quality (r²,mse,mae,spearman) + Δ, flags, and cost columns.

    Cost columns are labelled by pass: in the **proxy** pass cost splits across
    ``inference_s`` (encode) and ``train_head_s`` (head fit); in the **gt** pass
    inference is fused into the joint loop so ``gt_train_head_s`` is the end-to-end
    cost (``inference_s == 0``). See RQ2 framing in DATAPLAN / CLAUDE.md.
    """
    import pandas as pd

    fz = triple.proxy.set_index("model")
    ft = triple.gt.set_index("model")
    rows = []
    for model in triple.models:
        fzr = fz.loc[model]
        ftr = ft.loc[model] if model in ft.index else None
        ft_r2 = float(ftr["r2"]) if ftr is not None else float("nan")
        rows.append(
            {
                "model": model,
                "proxy_r2": float(fzr["r2"]),
                "gt_r2": ft_r2,
                "delta_r2": ft_r2 - float(fzr["r2"]),
                "proxy_mse": float(fzr["mse"]),
                "gt_mse": float(ftr["mse"]) if ftr is not None else float("nan"),
                "proxy_mae": float(fzr["mae"]),
                "gt_mae": float(ftr["mae"]) if ftr is not None else float("nan"),
                "proxy_spearman": float(fzr["spearman"]),
                "gt_spearman": float(ftr["spearman"]) if ftr is not None else float("nan"),
                "gt_skipped": triple.gt_skipped.get(model, False),
                "diverged": triple.diverged.get(model, False),
                "proxy_epochs": int(fzr["epochs_run"]) if "epochs_run" in fzr.index else 0,
                "gt_epochs": int(ftr["epochs_run"]) if ftr is not None else 0,
                "proxy_inference_s": float(fzr.get("inference_s", float("nan"))),
                "proxy_train_head_s": float(fzr.get("train_head_s", float("nan"))),
                "gt_train_head_s": (
                    float(ftr["train_head_s"]) if ftr is not None else float("nan")
                ),
                "proxy_peak_gpu_mem_mb": float(fzr.get("peak_gpu_mem_mb", float("nan"))),
                "gt_peak_gpu_mem_mb": (
                    float(ftr["peak_gpu_mem_mb"]) if ftr is not None else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_spearman(triple: Triple) -> float:
    """Spearman(proxy r², gt r²) across models — does the cheap proxy rank like truth?"""
    from scipy.stats import spearmanr

    fz = triple.proxy.set_index("model")["r2"]
    ft = triple.gt.set_index("model")["r2"]
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
    fz = triple.proxy.set_index("model")["r2"]
    ft = triple.gt.set_index("model")["r2"]
    regret_df = triple_regret_df(triple)
    first = regret_df.iloc[0]
    return TripleSummary(
        head=triple.head,
        n_models=len(triple.models),
        best_proxy_model=str(fz.idxmax()),
        best_proxy_r2=float(fz.max()),
        best_gt_model=str(ft.idxmax()),
        best_gt_r2=float(ft.max()),
        regret_at_1=float(first["regret"]),
        normalized_regret_at_1=float(first["normalized_regret"]),
        budget_to_zero=budget_to_zero(regret_df),
        regret_auc=float(regret_df["regret"].mean()),
        rank_spearman=rank_spearman(triple),
        n_diverged=int(sum(triple.diverged.values())),
        n_gt_skipped=int(sum(triple.gt_skipped.values())),
    )


# --------------------------------------------------------------------------- cross-run


def head_model_r2_matrix(triples: list[Triple], kind: str) -> pd.DataFrame:
    """model x head r² matrix for ``kind`` in {"proxy","gt"}; index = model."""
    import pandas as pd

    series = {}
    for t in triples:
        frame = t.proxy if kind == "proxy" else t.gt
        series[t.head] = frame.set_index("model")["r2"]
    matrix = pd.DataFrame(series)
    matrix.index.name = "model"
    return matrix.reset_index()


def divergence_matrix(triples: list[Triple]) -> pd.DataFrame:
    """model x head boolean matrix of ``diverged`` (gt r² < 0); index = model."""
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
    """Every model that diverged in *any* head: proxy→gt r² + gt rho, per head.

    Pins the "rank preserved (high Spearman), scale broken (negative r²)" story into a
    templated table so the narrative is written over given numbers, not re-derived from plots.
    """
    import pandas as pd

    diverged_any = sorted({m for t in triples for m, d in t.diverged.items() if d})
    rows = []
    for model in diverged_any:
        row: dict[str, object] = {"model": model}
        for t in triples:
            fz = t.proxy.set_index("model")["r2"]
            ftf = t.gt.set_index("model")
            row[f"{t.head}_proxy_r2"] = float(fz[model]) if model in fz.index else float("nan")
            if model in ftf.index:
                row[f"{t.head}_gt_r2"] = float(ftf.loc[model, "r2"])
                row[f"{t.head}_gt_spearman"] = float(ftf.loc[model, "spearman"])
            else:
                row[f"{t.head}_gt_r2"] = float("nan")
                row[f"{t.head}_gt_spearman"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def frozen_distribution_table(triples: list[Triple]) -> pd.DataFrame:
    """Per-head proxy r² spread: mean/std/min/max and n_negative (#1 — spread-collapse story).

    Uses population std (ddof=0) to match the deck's reported 0.31→0.07 collapse.
    """
    import pandas as pd

    rows = []
    for t in triples:
        r2 = t.proxy["r2"].astype(float)
        rows.append(
            {
                "head": t.head,
                "mean_proxy_r2": float(r2.mean()),
                "std_proxy_r2": float(r2.std(ddof=0)),
                "min_proxy_r2": float(r2.min()),
                "max_proxy_r2": float(r2.max()),
                "n_negative": int((r2 < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def head_gain_table(triples: list[Triple]) -> pd.DataFrame:
    """Per-model Δ proxy r² from narrowest to widest head (#2 — biggest gainers).

    Reuses the proxy-r² data already available in each triple; models sorted by gain desc.
    Single-head experiments: narrowest == widest, gain = 0 for every model.
    """
    import pandas as pd

    if not triples:
        return pd.DataFrame(columns=["model", "narrow_r2", "wide_r2", "gain"])
    narrowest = triples[0].proxy.set_index("model")["r2"].astype(float)
    widest = triples[-1].proxy.set_index("model")["r2"].astype(float)
    models = [m for m in narrowest.index if m in widest.index]
    rows = [
        {
            "model": m,
            "narrow_r2": float(narrowest[m]),
            "wide_r2": float(widest[m]),
            "gain": float(widest[m]) - float(narrowest[m]),
        }
        for m in models
    ]
    df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    return df


def epochs_table(triples: list[Triple]) -> pd.DataFrame:
    """Per-head early-stopping summary (#3): mean proxy epochs, n at cap, cap, mean gt.

    ``proxy_cap`` = global max of ``epochs_run`` across all proxy passes.
    ``n_proxy_at_cap`` = number of models that hit the cap (early stopping did not trigger).
    Skips gracefully if ``epochs_run`` is missing from the proxy CSV.
    """
    import pandas as pd

    if not triples:
        return pd.DataFrame(
            columns=[
                "head",
                "mean_proxy_epochs",
                "n_proxy_at_cap",
                "proxy_cap",
                "mean_gt_epochs",
            ]
        )

    # Global cap = max epochs_run across all proxy passes (the configured patience limit).
    all_caps: list[float] = []
    for t in triples:
        if "epochs_run" in t.proxy.columns:
            all_caps.extend(t.proxy["epochs_run"].dropna().tolist())
    proxy_cap = int(max(all_caps)) if all_caps else None

    rows = []
    for t in triples:
        if "epochs_run" not in t.proxy.columns:
            rows.append(
                {
                    "head": t.head,
                    "mean_proxy_epochs": float("nan"),
                    "n_proxy_at_cap": float("nan"),
                    "proxy_cap": proxy_cap,
                    "mean_gt_epochs": float("nan"),
                }
            )
            continue
        proxy_epochs = t.proxy["epochs_run"].dropna().astype(float)
        gt_epochs = (
            t.gt["epochs_run"].dropna().astype(float) if "epochs_run" in t.gt.columns else None
        )
        rows.append(
            {
                "head": t.head,
                "mean_proxy_epochs": float(proxy_epochs.mean()),
                "n_proxy_at_cap": (
                    int((proxy_epochs == proxy_cap).sum())
                    if proxy_cap is not None
                    else float("nan")
                ),
                "proxy_cap": proxy_cap,
                "mean_gt_epochs": (
                    float(gt_epochs.mean()) if gt_epochs is not None else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def head_rank_agreement_matrix(triples: list[Triple]) -> pd.DataFrame:
    """Head x head Spearman rho over proxy r2 across the common model set (#4).

    Each cell = Spearman(proxy r2 for head_i, proxy r2 for head_j) over models present
    in *both* heads. Index and columns = head labels. 1x1 for single-head experiments.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    heads = [t.head for t in triples]
    r2_by_head = {t.head: t.proxy.set_index("model")["r2"].astype(float) for t in triples}
    matrix: dict[str, list[float]] = {h: [] for h in heads}
    for hi in heads:
        for hj in heads:
            si, sj = r2_by_head[hi], r2_by_head[hj]
            common = [m for m in si.index if m in sj.index]
            if len(common) < 2:
                matrix[hi].append(float("nan"))
            else:
                rho, _ = spearmanr([si[m] for m in common], [sj[m] for m in common])
                matrix[hi].append(float(rho))
    df = pd.DataFrame(matrix, index=heads)
    df.index.name = "head"
    return df.reset_index()


def frozen_timing_share_table(triples: list[Triple]) -> pd.DataFrame:
    """Per-head % share of each timing substep summed over the model pool (#5).

    ``feature_extraction_pct`` = prepare_model + prepare_data + inference (backbone-bound cost).
    Rows sum to 100 (modulo floating-point rounding).
    """
    import pandas as pd

    from mlsys.analysis.theme import SUBSTEP_KEYS

    rows = []
    for t in triples:
        fz = t.proxy
        totals = {k: float(fz[k].sum()) if k in fz.columns else 0.0 for k in SUBSTEP_KEYS}
        grand = sum(totals.values())
        if grand == 0.0:
            pct = {k: 0.0 for k in SUBSTEP_KEYS}
        else:
            pct = {k: v / grand * 100.0 for k, v in totals.items()}
        rows.append(
            {
                "head": t.head,
                "prepare_model_pct": pct["prepare_model_s"],
                "prepare_data_pct": pct["prepare_data_s"],
                "inference_pct": pct["inference_s"],
                "train_head_pct": pct["train_head_s"],
                "eval_pct": pct["eval_s"],
                "feature_extraction_pct": (
                    pct["prepare_model_s"] + pct["prepare_data_s"] + pct["inference_s"]
                ),
            }
        )
    return pd.DataFrame(rows)


def value_frontier_table(triples: list[Triple]) -> pd.DataFrame:
    """Inference-cost vs r² at the widest head (#6 — inference value-frontier).

    Uses the widest head (``triples[-1]``) because ``inference_s`` is backbone-bound /
    head-independent, and the widest head gives the most informative r² cut. Writer must
    state explicitly that the frontier uses the widest head.
    Sorted by ``proxy_inference_s`` ascending (cheapest first).
    """
    import pandas as pd

    if not triples:
        return pd.DataFrame(
            columns=[
                "model",
                "proxy_inference_s",
                "proxy_r2",
                "gt_r2",
                "proxy_peak_gpu_mem_mb",
            ]
        )
    widest = triples[-1]
    fz = widest.proxy.set_index("model")
    ft = widest.gt.set_index("model")
    rows = []
    for model in widest.models:
        fzr = fz.loc[model]
        rows.append(
            {
                "model": model,
                "proxy_inference_s": float(fzr.get("inference_s", float("nan"))),
                "proxy_r2": float(fzr["r2"]),
                "gt_r2": float(ft.loc[model, "r2"]) if model in ft.index else float("nan"),
                "proxy_peak_gpu_mem_mb": float(fzr.get("peak_gpu_mem_mb", float("nan"))),
            }
        )
    df = pd.DataFrame(rows).sort_values("proxy_inference_s").reset_index(drop=True)
    return df


def cost_table(triples: list[Triple]) -> pd.DataFrame:
    """Per-(head,model) RQ2 cost: proxy total, gt total, peak mem, epochs.

    Proxy total = ``inference_s + train_head_s`` (encode + head fit); gt total =
    ``train_head_s`` (inference fused in). Surfaces the ~10x model2vec-vs-transformer spread.
    """
    import pandas as pd

    rows = []
    for t in triples:
        fz = t.proxy.set_index("model")
        ft = t.gt.set_index("model")
        for model in t.models:
            fzr = fz.loc[model]
            ftr = ft.loc[model] if model in ft.index else None
            proxy_total = float(fzr.get("inference_s", 0.0)) + float(fzr.get("train_head_s", 0.0))
            rows.append(
                {
                    "head": t.head,
                    "model": model,
                    "proxy_total_s": proxy_total,
                    "gt_total_s": (float(ftr["train_head_s"]) if ftr is not None else float("nan")),
                    "proxy_peak_gpu_mem_mb": float(fzr.get("peak_gpu_mem_mb", float("nan"))),
                    "gt_peak_gpu_mem_mb": (
                        float(ftr["peak_gpu_mem_mb"]) if ftr is not None else float("nan")
                    ),
                    "gt_epochs": int(ftr["epochs_run"]) if ftr is not None else 0,
                    "gt_skipped": t.gt_skipped.get(model, False),
                }
            )
    return pd.DataFrame(rows)

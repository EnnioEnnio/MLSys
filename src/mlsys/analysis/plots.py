"""seaborn / matplotlib plot functions, one fixed PNG slug each.

Every function ``savefig``s to ``<out_dir>/<slug>.png`` and returns that path. The slugs are
contractual: ``report.py`` and ``analysis.md`` reference them verbatim, so do not rename.
All heavy imports are lazy and matplotlib is forced onto the non-interactive ``Agg`` backend
so the module works headless (CI / cluster / tmp smoke tests).

RQ2 note encoded in the timing plots: in the **frozen** pass cost splits across
``inference_s`` (encode) + ``train_head_s`` (head fit); in the **finetune** pass inference is
fused into the joint loop (``inference_s == 0``) so ``train_head_s`` is the end-to-end cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlsys.analysis.regret_recompute import recompute_regret
from mlsys.analysis.tables import budget_to_zero, rank_spearman

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from mlsys.analysis.loader import Triple


def _setup():
    """Lazy import + consistent report theme. Returns the (plt, sns) pair."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    return plt, sns


def _save(fig: Figure, out_dir: str | Path, slug: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _regret_df(triple: Triple):
    return (
        triple.regret
        if triple.regret is not None
        else recompute_regret(triple.frozen, triple.finetune)
    )


# --------------------------------------------------------------------- per-triple quality


def plot_r2_frozen_vs_finetune(triple: Triple, out_dir: str | Path) -> Path:
    """(1) Grouped bars: frozen vs finetune r² per model. → ``r2_frozen_vs_finetune.png``."""
    import pandas as pd

    plt, sns = _setup()
    fz = triple.frozen.set_index("model")["r2"]
    ft = triple.finetune.set_index("model")["r2"]
    long = pd.DataFrame(
        [{"model": m, "pass": "frozen", "r2": float(fz[m])} for m in triple.models]
        + [
            {"model": m, "pass": "finetune", "r2": float(ft[m])}
            for m in triple.models
            if m in ft.index
        ]
    )
    fig, ax = plt.subplots(figsize=(max(8, len(triple.models)), 5))
    sns.barplot(long, x="model", y="r2", hue="pass", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Frozen vs finetune r² — head {triple.head}")
    ax.tick_params(axis="x", rotation=75)
    return _save(fig, out_dir, "r2_frozen_vs_finetune")


def plot_proxy_scatter(triple: Triple, out_dir: str | Path) -> Path:
    """(2) Scatter frozen-r² (proxy) vs finetune-r² with y=x; flag skipped/diverged.

    → ``proxy_scatter.png``.
    """
    plt, _ = _setup()
    fz = triple.frozen.set_index("model")["r2"]
    ft = triple.finetune.set_index("model")["r2"]
    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = 1e9, -1e9
    for m in triple.models:
        if m not in ft.index:
            continue
        x, y = float(fz[m]), float(ft[m])
        lo, hi = min(lo, x, y), max(hi, x, y)
        if triple.finetune_skipped.get(m):
            color, marker = "tab:gray", "s"
        elif triple.diverged.get(m):
            color, marker = "tab:red", "X"
        else:
            color, marker = "tab:blue", "o"
        ax.scatter(x, y, c=color, marker=marker, s=60, zorder=3)
        ax.annotate(m, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    pad = 0.05 * (hi - lo or 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", label="y = x")
    ax.set_xlabel("frozen r² (cheap proxy)")
    ax.set_ylabel("finetune r² (ground truth)")
    ax.set_title(f"Proxy quality — head {triple.head}\nblue=ok  red=diverged  gray=skipped")
    ax.legend()
    return _save(fig, out_dir, "proxy_scatter")


def plot_r2_delta(triple: Triple, out_dir: str | Path) -> Path:
    """(3) Diverging bars of Δ = finetune - frozen r² per model. → ``r2_delta.png``."""
    import pandas as pd

    plt, _ = _setup()
    fz = triple.frozen.set_index("model")["r2"]
    ft = triple.finetune.set_index("model")["r2"]
    deltas = [
        {"model": m, "delta_r2": float(ft[m]) - float(fz[m])}
        for m in triple.models
        if m in ft.index
    ]
    df = pd.DataFrame(deltas).sort_values("delta_r2")
    colors = ["tab:red" if d < 0 else "tab:green" for d in df["delta_r2"]]
    fig, ax = plt.subplots(figsize=(max(8, len(df)), 5))
    ax.bar(df["model"], df["delta_r2"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ r² (finetune - frozen)")
    ax.set_title(f"Finetune lift over frozen — head {triple.head}")
    ax.tick_params(axis="x", rotation=75)
    return _save(fig, out_dir, "r2_delta")


def plot_regret_curve(triple: Triple, out_dir: str | Path) -> Path:
    """(4) Regret + normalized_regret vs budget B. → ``regret_curve.png``."""
    plt, _ = _setup()
    df = _regret_df(triple)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["budget"], df["regret"], marker="o", label="regret")
    ax.plot(df["budget"], df["normalized_regret"], marker="s", linestyle="--", label="normalized")
    b0 = budget_to_zero(df)
    ax.axvline(b0, color="gray", linestyle=":", label=f"budget-to-zero = {b0}")
    ax.set_xlabel("budget B (top-B of frozen-r² ranking)")
    ax.set_ylabel("regret (r²)")
    ax.set_title(f"Regret vs budget — head {triple.head}")
    ax.legend()
    return _save(fig, out_dir, "regret_curve")


def plot_finetune_spearman_vs_r2(triple: Triple, out_dir: str | Path) -> Path:
    """(5) Finetune Spearman vs r² — the "rank kept, scale broken" story.

    → ``finetune_spearman_vs_r2.png``.
    """
    plt, _ = _setup()
    ft = triple.finetune
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, r2, spearman in zip(ft["model"], ft["r2"], ft["spearman"], strict=True):
        color = "tab:red" if triple.diverged.get(str(model), False) else "tab:blue"
        ax.scatter(r2, spearman, c=color, s=60, zorder=3)
        ax.annotate(
            str(model), (r2, spearman), fontsize=7, xytext=(3, 3), textcoords="offset points"
        )
    ax.axvline(0, color="gray", linestyle="--", label="r² = 0 (scale broken left of here)")
    ax.set_xlabel("finetune r²")
    ax.set_ylabel("finetune Spearman")
    ax.set_title(
        f"Rank preserved vs scale broken — head {triple.head}\n"
        "diverged models (red) keep high Spearman despite negative r²"
    )
    ax.legend()
    return _save(fig, out_dir, "finetune_spearman_vs_r2")


# ------------------------------------------------------------------------ per-triple RQ2


def plot_timing_stacked(triple: Triple, out_dir: str | Path) -> Path:
    """(R1) Stacked substep timing per model, frozen vs finetune side-by-side.

    → ``timing_stacked.png``. Frozen stack = inference_s + train_head_s + others; finetune
    stack = train_head_s (inference fused → 0).
    """
    import numpy as np

    plt, _ = _setup()
    keys = ["prepare_model_s", "prepare_data_s", "inference_s", "train_head_s", "eval_s"]
    fz = triple.frozen.set_index("model")
    ft = triple.finetune.set_index("model")
    models = triple.models
    x = np.arange(len(models))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.1), 5.5))
    cmap = plt.get_cmap("tab10")
    for offset, frame, label in ((-width / 2, fz, "frozen"), (width / 2, ft, "finetune")):
        bottom = np.zeros(len(models))
        for ki, key in enumerate(keys):
            vals = np.array([float(frame.loc[m, key]) if m in frame.index else 0.0 for m in models])
            ax.bar(
                x + offset,
                vals,
                width,
                bottom=bottom,
                color=cmap(ki),
                label=f"{key}" if label == "frozen" else None,
            )
            bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=75)
    ax.set_ylabel("seconds (left bar=frozen, right=finetune)")
    ax.set_title(f"Timing breakdown frozen vs finetune — head {triple.head}")
    ax.legend(title="substep", fontsize=8)
    return _save(fig, out_dir, "timing_stacked")


def plot_peak_gpu_mem(triple: Triple, out_dir: str | Path) -> Path:
    """(R2) peak_gpu_mem_mb per model, frozen vs finetune. → ``peak_gpu_mem.png``."""
    import pandas as pd

    plt, sns = _setup()
    fz = triple.frozen.set_index("model")["peak_gpu_mem_mb"]
    ft = triple.finetune.set_index("model")["peak_gpu_mem_mb"]
    long = pd.DataFrame(
        [{"model": m, "pass": "frozen", "peak_gpu_mem_mb": float(fz[m])} for m in triple.models]
        + [
            {"model": m, "pass": "finetune", "peak_gpu_mem_mb": float(ft[m])}
            for m in triple.models
            if m in ft.index
        ]
    )
    fig, ax = plt.subplots(figsize=(max(8, len(triple.models)), 5))
    sns.barplot(long, x="model", y="peak_gpu_mem_mb", hue="pass", ax=ax)
    ax.set_title(f"Peak GPU memory frozen vs finetune — head {triple.head}")
    ax.tick_params(axis="x", rotation=75)
    return _save(fig, out_dir, "peak_gpu_mem")


def plot_frozen_time_breakdown(triple: Triple, out_dir: str | Path) -> Path:
    """(R3) Frozen pass: inference_s vs train_head_s share per model.

    → ``frozen_time_breakdown.png``.
    """
    import numpy as np

    plt, _ = _setup()
    fz = triple.frozen.set_index("model")
    models = triple.models
    inf = np.array([float(fz.loc[m, "inference_s"]) for m in models])
    head = np.array([float(fz.loc[m, "train_head_s"]) for m in models])
    fig, ax = plt.subplots(figsize=(max(8, len(models)), 5))
    ax.bar(models, inf, label="inference_s (encode)", color="tab:purple")
    ax.bar(models, head, bottom=inf, label="train_head_s (head fit)", color="tab:orange")
    ax.set_ylabel("seconds")
    ax.set_title(f"Frozen pass: where the time goes — head {triple.head}")
    ax.tick_params(axis="x", rotation=75)
    ax.legend()
    return _save(fig, out_dir, "frozen_time_breakdown")


# --------------------------------------------------------------------------- comparison


def plot_regret_curves_by_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(6) Overlaid regret curves, one line per head. → ``regret_curves_by_head.png``."""
    plt, _ = _setup()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for t in triples:
        df = _regret_df(t)
        ax.plot(df["budget"], df["regret"], marker="o", label=t.head)
    ax.set_xlabel("budget B")
    ax.set_ylabel("regret (r²)")
    ax.set_title("Regret vs budget by head width")
    ax.legend(title="head")
    return _save(fig, out_dir, "regret_curves_by_head")


def plot_regret_at1_vs_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(7) regret@1 / AUC / budget-to-zero vs head. → ``regret_at1_vs_head.png``."""
    plt, _ = _setup()
    heads = [t.head for t in triples]
    at1, auc, b0 = [], [], []
    for t in triples:
        df = _regret_df(t)
        at1.append(float(df.iloc[0]["regret"]))
        auc.append(float(df["regret"].mean()))
        b0.append(budget_to_zero(df))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(heads, at1, marker="o", color="tab:blue", label="regret@1")
    ax1.plot(heads, auc, marker="s", color="tab:green", label="mean regret (AUC)")
    ax1.set_ylabel("regret (r²)")
    ax2 = ax1.twinx()
    ax2.plot(heads, b0, marker="^", color="tab:red", label="budget-to-zero")
    ax2.set_ylabel("budget-to-zero")
    ax1.set_title("Proxy shortlist quality vs head width")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    return _save(fig, out_dir, "regret_at1_vs_head")


def plot_best_r2_vs_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(8) Best frozen & best finetune r² vs head width. → ``best_r2_vs_head.png``."""
    plt, _ = _setup()
    heads = [t.head for t in triples]
    best_fz = [float(t.frozen["r2"].max()) for t in triples]
    best_ft = [float(t.finetune["r2"].max()) for t in triples]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(heads, best_fz, marker="o", label="best frozen r²")
    ax.plot(heads, best_ft, marker="s", label="best finetune r²")
    ax.set_ylabel("r²")
    ax.set_title("Best achievable r² vs head width")
    ax.legend()
    return _save(fig, out_dir, "best_r2_vs_head")


def _heatmap(triples: list[Triple], kind: str, out_dir: str | Path, slug: str, title: str) -> Path:
    import pandas as pd

    plt, sns = _setup()
    series = {}
    for t in triples:
        frame = t.frozen if kind == "frozen" else t.finetune
        series[t.head] = frame.set_index("model")["r2"]
    matrix = pd.DataFrame(series)
    fig, ax = plt.subplots(figsize=(max(6, len(triples) * 1.5), max(6, len(matrix) * 0.5)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={"label": "r²"})
    ax.set_title(title)
    return _save(fig, out_dir, slug)


def plot_heatmap_frozen_r2(triples: list[Triple], out_dir: str | Path) -> Path:
    """(9a) model x head frozen-r² heatmap. → ``heatmap_frozen_r2.png``."""
    return _heatmap(triples, "frozen", out_dir, "heatmap_frozen_r2", "Frozen r² (model x head)")


def plot_heatmap_finetune_r2(triples: list[Triple], out_dir: str | Path) -> Path:
    """(9b) model x head finetune-r² heatmap. → ``heatmap_finetune_r2.png``."""
    return _heatmap(
        triples, "finetune", out_dir, "heatmap_finetune_r2", "Finetune r² (model x head)"
    )


def plot_divergence_map(triples: list[Triple], out_dir: str | Path) -> Path:
    """(10) Binary model x head map of ``diverged`` (finetune r² < 0). → ``divergence_map.png``.

    Reads the deberta/electra/roberta blow-up as a *structural* failure mode across heads.
    """
    import pandas as pd

    plt, sns = _setup()
    matrix = pd.DataFrame({t.head: pd.Series(t.diverged) for t in triples}).astype(float)
    fig, ax = plt.subplots(figsize=(max(6, len(triples) * 1.5), max(6, len(matrix) * 0.5)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap=sns.color_palette(["#2ca02c", "#d62728"]),
        cbar=False,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Divergence map — 1 = finetune r² < 0 (red)")
    return _save(fig, out_dir, "divergence_map")


def plot_proxy_rank_spearman_vs_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(11) Spearman(frozen proxy rank, finetune rank) per head, one bar each.

    → ``proxy_rank_spearman_vs_head.png``. Does a wider frozen head make a *better proxy*?
    """
    plt, sns = _setup()
    heads = [t.head for t in triples]
    rhos = [rank_spearman(t) for t in triples]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=heads, y=rhos, ax=ax, color="tab:blue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Spearman(frozen rank, finetune rank)")
    ax.set_title("Proxy rank fidelity vs head width")
    for i, r in enumerate(rhos):
        ax.text(i, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
    return _save(fig, out_dir, "proxy_rank_spearman_vs_head")


def plot_cost_vs_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(R4) Cross-run mean train_head_s & peak_gpu_mem_mb vs head width. → ``cost_vs_head.png``."""
    plt, _ = _setup()
    heads = [t.head for t in triples]
    mean_head_s = [float(t.finetune["train_head_s"].mean()) for t in triples]
    mean_mem = [float(t.finetune["peak_gpu_mem_mb"].mean()) for t in triples]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(heads, mean_head_s, marker="o", color="tab:orange", label="mean finetune train_head_s")
    ax1.set_ylabel("mean train_head_s")
    ax2 = ax1.twinx()
    ax2.plot(heads, mean_mem, marker="s", color="tab:purple", label="mean peak_gpu_mem_mb")
    ax2.set_ylabel("mean peak_gpu_mem_mb")
    ax1.set_title("Finetune cost vs head width")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    return _save(fig, out_dir, "cost_vs_head")

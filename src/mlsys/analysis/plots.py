"""seaborn / matplotlib plot functions, one fixed PNG slug each.

Every function ``savefig``s to ``<out_dir>/<slug>.png`` and returns that path. The slugs are
contractual: ``report.py`` and ``analysis.md`` reference them verbatim, so do not rename.
All heavy imports are lazy and matplotlib is forced onto the non-interactive ``Agg`` backend
so the module works headless (CI / cluster / tmp smoke tests).

RQ2 note encoded in the timing plots: in the **frozen** pass cost splits across
``inference_s`` (encode) + ``train_head_s`` (head fit); in the **finetune** pass inference is
fused into the joint loop (``inference_s == 0``) so ``train_head_s`` is the end-to-end cost.

Colours, markers, and colormaps come from ``theme.py`` — one consistent palette across every
plot so frozen is always tan, finetune always burgundy, and each substep keeps one
colour+hatch in every stacked-bar chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlsys.analysis import theme
from mlsys.analysis.regret_recompute import triple_regret_df
from mlsys.analysis.tables import budget_to_zero, rank_spearman

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from mlsys.analysis.loader import Triple


def _setup():
    """Lazy import + themed report style. Returns the (plt, sns) pair."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    theme.apply_theme(plt, sns)
    return plt, sns


def _save(fig: Figure, out_dir: str | Path, slug: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _pass_labels(triples: list[Triple]) -> tuple[str, str]:
    """Return (proxy_label, reference_label) from the first triple, with safe defaults."""
    if triples:
        return triples[0].proxy_label, triples[0].reference_label
    return "proxy", "reference"


# --------------------------------------------------------------------- per-triple quality


def plot_r2_proxy_vs_reference(triple: Triple, out_dir: str | Path) -> Path:
    """(1) Grouped bars: proxy vs reference r² per model. → ``r2_proxy_vs_reference.png``."""
    import pandas as pd

    plt, sns = _setup()
    pl, tl = triple.proxy_label, triple.reference_label
    fz = triple.proxy.set_index("model")["r2"]
    ft = triple.reference.set_index("model")["r2"]
    long = pd.DataFrame(
        [{"model": m, "pass": pl, "r2": float(fz[m])} for m in triple.models]
        + [{"model": m, "pass": tl, "r2": float(ft[m])} for m in triple.models if m in ft.index]
    )
    palette = {pl: theme.PROXY_COLOR, tl: theme.REFERENCE_COLOR}
    fig, ax = plt.subplots(figsize=(max(8, len(triple.models)), 5))
    sns.barplot(long, x="model", y="r2", hue="pass", palette=palette, ax=ax)
    ax.axhline(0, color=theme.INK, linewidth=0.8)
    ax.set_title(f"{pl} vs {tl} r² — head {triple.head}")
    ax.tick_params(axis="x", rotation=75)
    return _save(fig, out_dir, "r2_proxy_vs_reference")


def plot_proxy_scatter(triple: Triple, out_dir: str | Path) -> Path:
    """(2) Scatter proxy-r² vs reference-r² with y=x; flag skipped/diverged.

    → ``proxy_scatter.png``.
    """
    plt, _ = _setup()
    fz = triple.proxy.set_index("model")["r2"]
    ft = triple.reference.set_index("model")["r2"]
    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = 1e9, -1e9
    for m in triple.models:
        if m not in ft.index:
            continue
        x, y = float(fz[m]), float(ft[m])
        lo, hi = min(lo, x, y), max(hi, x, y)
        if triple.ref_skipped.get(m):
            color = theme.STATUS_COLORS["skipped"]
            marker = theme.STATUS_MARKERS["skipped"]
        elif triple.diverged.get(m):
            color = theme.STATUS_COLORS["diverged"]
            marker = theme.STATUS_MARKERS["diverged"]
        else:
            color = theme.STATUS_COLORS["ok"]
            marker = theme.STATUS_MARKERS["ok"]
        ax.scatter(x, y, c=color, marker=marker, s=60, zorder=3)
        ax.annotate(m, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    pad = 0.05 * (hi - lo or 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=theme.GRID, label="y = x")
    ax.set_xlabel(f"{triple.proxy_label} r² (cheap proxy)")
    ref_qualifier = " (ground truth)" if triple.reference_label == "finetune" else ""
    ax.set_ylabel(f"{triple.reference_label} r²{ref_qualifier}")
    ax.set_title(f"Proxy quality — head {triple.head}\nslate=ok  brick-red=diverged  grey=skipped")
    ax.legend()
    return _save(fig, out_dir, "proxy_scatter")


def plot_r2_delta(triple: Triple, out_dir: str | Path) -> Path:
    """(3) Diverging bars of Δ = reference - proxy r² per model. → ``r2_delta.png``."""
    import pandas as pd

    plt, _ = _setup()
    fz = triple.proxy.set_index("model")["r2"]
    ft = triple.reference.set_index("model")["r2"]
    deltas = [
        {"model": m, "delta_r2": float(ft[m]) - float(fz[m])}
        for m in triple.models
        if m in ft.index
    ]
    df = pd.DataFrame(deltas).sort_values("delta_r2")
    colors = [
        theme.DELTA_COLORS["neg"] if d < 0 else theme.DELTA_COLORS["pos"] for d in df["delta_r2"]
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(df)), 5))
    ax.bar(df["model"], df["delta_r2"], color=colors)
    ax.axhline(0, color=theme.INK, linewidth=0.8)
    ax.set_ylabel(f"Δ r² ({triple.reference_label} - {triple.proxy_label})")
    ax.set_title(f"{triple.reference_label} lift over {triple.proxy_label} — head {triple.head}")
    ax.tick_params(axis="x", rotation=75)
    return _save(fig, out_dir, "r2_delta")


def plot_regret_curve(triple: Triple, out_dir: str | Path) -> Path:
    """(4) Regret + normalized_regret vs budget B. → ``regret_curve.png``."""
    plt, _ = _setup()
    df = triple_regret_df(triple)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["budget"],
        df["regret"],
        marker=theme.PROXY_MARKER,
        color=theme.PROXY_COLOR,
        label="regret",
    )
    ax.plot(
        df["budget"],
        df["normalized_regret"],
        marker=theme.REFERENCE_MARKER,
        linestyle="--",
        color=theme.REFERENCE_COLOR,
        label="normalized",
    )
    b0 = budget_to_zero(df)
    ax.axvline(b0, color=theme.GRID, linestyle=":", label=f"budget-to-zero = {b0}")
    ax.set_xlabel(f"budget B (top-B of {triple.proxy_label}-r² ranking)")
    ax.set_ylabel("regret (r²)")
    ax.set_title(f"Regret vs budget — head {triple.head}")
    ax.legend()
    return _save(fig, out_dir, "regret_curve")


def plot_ref_spearman_vs_r2(triple: Triple, out_dir: str | Path) -> Path:
    """(5) Reference-pass Spearman vs r² — the "rank kept, scale broken" story.

    → ``ref_spearman_vs_r2.png``.
    """
    plt, _ = _setup()
    ft = triple.reference
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, r2, spearman in zip(ft["model"], ft["r2"], ft["spearman"], strict=True):
        if triple.diverged.get(str(model), False):
            color = theme.STATUS_COLORS["diverged"]
            marker = theme.STATUS_MARKERS["diverged"]
        else:
            color = theme.STATUS_COLORS["ok"]
            marker = theme.STATUS_MARKERS["ok"]
        ax.scatter(r2, spearman, c=color, marker=marker, s=60, zorder=3)
        ax.annotate(
            str(model), (r2, spearman), fontsize=7, xytext=(3, 3), textcoords="offset points"
        )
    ax.axvline(0, color=theme.GRID, linestyle="--", label="r² = 0 (scale broken left of here)")
    ax.set_xlabel(f"{triple.reference_label} r²")
    ax.set_ylabel(f"{triple.reference_label} Spearman")
    ax.set_title(
        f"Rank preserved vs scale broken — head {triple.head}\n"
        "diverged models (brick-red) keep high Spearman despite negative r²"
    )
    ax.legend()
    return _save(fig, out_dir, "ref_spearman_vs_r2")


# ------------------------------------------------------------------------ per-triple RQ2


def plot_timing_stacked(triple: Triple, out_dir: str | Path) -> Path:
    """(R1) Stacked substep timing per model, proxy vs reference side-by-side.

    → ``timing_stacked.png``. Proxy stack = inference_s + train_head_s + others; reference
    stack = train_head_s (inference fused → 0 for finetune).
    """
    import numpy as np

    from mlsys.analysis.theme import SUBSTEP_COLORS, SUBSTEP_HATCHES, SUBSTEP_KEYS

    plt, _ = _setup()
    fz = triple.proxy.set_index("model")
    ft = triple.reference.set_index("model")
    models = triple.models
    x = np.arange(len(models))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.1), 5.5))
    pl, tl = triple.proxy_label, triple.reference_label
    for offset, frame, label in ((-width / 2, fz, pl), (width / 2, ft, tl)):
        bottom = np.zeros(len(models))
        for ki, key in enumerate(SUBSTEP_KEYS):
            vals = np.array([float(frame.loc[m, key]) if m in frame.index else 0.0 for m in models])
            ax.bar(
                x + offset,
                vals,
                width,
                bottom=bottom,
                color=SUBSTEP_COLORS[ki],
                hatch=SUBSTEP_HATCHES[ki],
                label=f"{key}" if label == pl else None,
            )
            bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=75)
    ax.set_ylabel(f"seconds (left bar={pl}, right={tl})")
    ax.set_title(f"Timing breakdown {pl} vs {tl} — head {triple.head}")
    ax.legend(title="substep", fontsize=8)
    return _save(fig, out_dir, "timing_stacked")


def plot_peak_gpu_mem(triple: Triple, out_dir: str | Path) -> Path:
    """(R2) peak_gpu_mem_mb per model, proxy vs reference. → ``peak_gpu_mem.png``."""
    import pandas as pd

    plt, sns = _setup()
    pl, tl = triple.proxy_label, triple.reference_label
    fz = triple.proxy.set_index("model")["peak_gpu_mem_mb"]
    ft = triple.reference.set_index("model")["peak_gpu_mem_mb"]
    long = pd.DataFrame(
        [{"model": m, "pass": pl, "peak_gpu_mem_mb": float(fz[m])} for m in triple.models]
        + [
            {"model": m, "pass": tl, "peak_gpu_mem_mb": float(ft[m])}
            for m in triple.models
            if m in ft.index
        ]
    )
    palette = {pl: theme.PROXY_COLOR, tl: theme.REFERENCE_COLOR}
    fig, ax = plt.subplots(figsize=(max(8, len(triple.models)), 5))
    sns.barplot(long, x="model", y="peak_gpu_mem_mb", hue="pass", palette=palette, ax=ax)
    ax.set_title(f"Peak GPU memory {pl} vs {tl} — head {triple.head}")
    ax.tick_params(axis="x", rotation=75)
    return _save(fig, out_dir, "peak_gpu_mem")


def plot_proxy_time_breakdown(triple: Triple, out_dir: str | Path) -> Path:
    """(R3) Proxy pass: inference_s vs train_head_s share per model.

    → ``proxy_time_breakdown.png``.
    """
    import numpy as np

    from mlsys.analysis.theme import SUBSTEP_COLORS, SUBSTEP_HATCHES, SUBSTEP_KEYS

    plt, _ = _setup()
    fz = triple.proxy.set_index("model")
    models = triple.models
    inf_idx = SUBSTEP_KEYS.index("inference_s")
    head_idx = SUBSTEP_KEYS.index("train_head_s")
    inf = np.array([float(fz.loc[m, "inference_s"]) for m in models])
    head = np.array([float(fz.loc[m, "train_head_s"]) for m in models])
    fig, ax = plt.subplots(figsize=(max(8, len(models)), 5))
    ax.bar(
        models,
        inf,
        label="inference_s (encode)",
        color=SUBSTEP_COLORS[inf_idx],
        hatch=SUBSTEP_HATCHES[inf_idx],
    )
    ax.bar(
        models,
        head,
        bottom=inf,
        label="train_head_s (head fit)",
        color=SUBSTEP_COLORS[head_idx],
        hatch=SUBSTEP_HATCHES[head_idx],
    )
    ax.set_ylabel("seconds")
    ax.set_title(f"{triple.proxy_label} pass: where the time goes — head {triple.head}")
    ax.tick_params(axis="x", rotation=75)
    ax.legend()
    return _save(fig, out_dir, "proxy_time_breakdown")


# --------------------------------------------------------------------------- comparison


def plot_regret_curves_by_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(6) Overlaid regret curves, one line per head. → ``regret_curves_by_head.png``."""
    plt, _ = _setup()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for t in triples:
        df = triple_regret_df(t)
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
        df = triple_regret_df(t)
        at1.append(float(df.iloc[0]["regret"]))
        auc.append(float(df["regret"].mean()))
        b0.append(budget_to_zero(df))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(heads, at1, marker=theme.PROXY_MARKER, color=theme.PROXY_COLOR, label="regret@1")
    ax1.plot(
        heads,
        auc,
        marker=theme.REFERENCE_MARKER,
        color=theme.REFERENCE_COLOR,
        label="mean regret (AUC)",
    )
    ax1.set_ylabel("regret (r²)")
    ax2 = ax1.twinx()
    ax2.plot(heads, b0, marker="^", color=theme.STATUS_COLORS["diverged"], label="budget-to-zero")
    ax2.set_ylabel("budget-to-zero")
    ax1.set_title("Proxy shortlist quality vs head width")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    return _save(fig, out_dir, "regret_at1_vs_head")


def plot_best_r2_vs_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(8) Best proxy & best reference r² vs head width. → ``best_r2_vs_head.png``."""
    pl, tl = _pass_labels(triples)
    plt, _ = _setup()
    heads = [t.head for t in triples]
    best_fz = [float(t.proxy["r2"].max()) for t in triples]
    best_ft = [float(t.reference["r2"].max()) for t in triples]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        heads, best_fz, marker=theme.PROXY_MARKER, color=theme.PROXY_COLOR, label=f"best {pl} r²"
    )
    ax.plot(
        heads,
        best_ft,
        marker=theme.REFERENCE_MARKER,
        color=theme.REFERENCE_COLOR,
        label=f"best {tl} r²",
    )
    ax.set_ylabel("r²")
    ax.set_title("Best achievable r² vs head width")
    ax.legend()
    return _save(fig, out_dir, "best_r2_vs_head")


def _heatmap(triples: list[Triple], kind: str, out_dir: str | Path, slug: str, title: str) -> Path:
    import pandas as pd

    plt, sns = _setup()
    series = {}
    for t in triples:
        frame = t.proxy if kind == "proxy" else t.reference
        series[t.head] = frame.set_index("model")["r2"]
    matrix = pd.DataFrame(series)
    fig, ax = plt.subplots(figsize=(max(6, len(triples) * 1.5), max(6, len(matrix) * 0.5)))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap=theme.get_cmap_r2(), ax=ax, cbar_kws={"label": "r²"}
    )
    ax.set_title(title)
    return _save(fig, out_dir, slug)


def plot_heatmap_proxy_r2(triples: list[Triple], out_dir: str | Path) -> Path:
    """(9a) model x head proxy-r² heatmap. → ``heatmap_proxy_r2.png``."""
    pl = _pass_labels(triples)[0]
    return _heatmap(triples, "proxy", out_dir, "heatmap_proxy_r2", f"{pl} r² (model x head)")


def plot_heatmap_ref_r2(triples: list[Triple], out_dir: str | Path) -> Path:
    """(9b) model x head reference-r² heatmap. → ``heatmap_ref_r2.png``."""
    tl = _pass_labels(triples)[1]
    return _heatmap(triples, "reference", out_dir, "heatmap_ref_r2", f"{tl} r² (model x head)")


def plot_divergence_map(triples: list[Triple], out_dir: str | Path) -> Path:
    """(10) Binary model x head map of ``diverged`` (finetune r² < 0). → ``divergence_map.png``.

    Reads the deberta/electra/roberta blow-up as a *structural* failure mode across heads.
    """
    import pandas as pd

    plt, sns = _setup()
    matrix = pd.DataFrame({t.head: pd.Series(t.diverged) for t in triples}).astype(float)
    fig, ax = plt.subplots(figsize=(max(6, len(triples) * 1.5), max(6, len(matrix) * 0.5)))
    if matrix.empty:
        ax.text(0.5, 0.5, "no diverged models", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, out_dir, "divergence_map")
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap=theme.get_binary_cmap(),
        cbar=False,
        ax=ax,
        linewidths=0.5,
    )
    tl = _pass_labels(triples)[1]
    ax.set_title(f"Divergence map — 1 = {tl} r² < 0 (brick-red)")
    return _save(fig, out_dir, "divergence_map")


def plot_proxy_rank_spearman_vs_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(11) Spearman(proxy rank, reference rank) per head, one bar each.

    → ``proxy_rank_spearman_vs_head.png``. Does a wider proxy head make a *better proxy*?
    """
    pl, tl = _pass_labels(triples)
    plt, sns = _setup()
    heads = [t.head for t in triples]
    rhos = [rank_spearman(t) for t in triples]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=heads, y=rhos, ax=ax, color=theme.PROXY_COLOR)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Spearman({pl} rank, {tl} rank)")
    ax.set_title("Proxy rank fidelity vs head width")
    for i, r in enumerate(rhos):
        ax.text(i, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
    return _save(fig, out_dir, "proxy_rank_spearman_vs_head")


def plot_cost_vs_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(R4) Cross-run mean train_head_s & peak_gpu_mem_mb vs head width. → ``cost_vs_head.png``."""
    tl = _pass_labels(triples)[1]
    plt, _ = _setup()
    heads = [t.head for t in triples]
    mean_head_s = [float(t.reference["train_head_s"].mean()) for t in triples]
    mean_mem = [float(t.reference["peak_gpu_mem_mb"].mean()) for t in triples]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        heads,
        mean_head_s,
        marker=theme.REFERENCE_MARKER,
        color=theme.REFERENCE_COLOR,
        label=f"mean {tl} train_head_s",
    )
    ax1.set_ylabel("mean train_head_s")
    ax2 = ax1.twinx()
    ax2.plot(
        heads,
        mean_mem,
        marker=theme.PROXY_MARKER,
        color=theme.PROXY_COLOR,
        label="mean peak_gpu_mem_mb",
    )
    ax2.set_ylabel("mean peak_gpu_mem_mb")
    ax1.set_title(f"{tl} cost vs head width")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    return _save(fig, out_dir, "cost_vs_head")


# ----------------------------------------------------------- new deck-slide plots (#3-#6)


def plot_epochs_vs_head(triples: list[Triple], out_dir: str | Path) -> Path:
    """(#3) Grouped bar: mean proxy vs reference epochs per head; annotate proxy cap.

    → ``epochs_vs_head.png``. Shows where early-stopping kicks in (proxy) vs where the
    full budget is consumed (reference/finetune).
    """
    from mlsys.analysis.tables import epochs_table

    plt, _ = _setup()
    et = epochs_table(triples)
    if et.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5, "epochs_run not available", ha="center", va="center", transform=ax.transAxes
        )
        return _save(fig, out_dir, "epochs_vs_head")

    heads = et["head"].tolist()
    x = range(len(heads))
    mean_fz = et["mean_proxy_epochs"].tolist()
    mean_ft = et["mean_ref_epochs"].tolist()
    cap = et["proxy_cap"].iloc[0] if "proxy_cap" in et.columns else None

    fig, ax = plt.subplots(figsize=(max(6, len(heads) * 1.5), 5))
    width = 0.35
    pl, tl = _pass_labels(triples)
    ax.bar(
        [i - width / 2 for i in x],
        mean_fz,
        width,
        label=f"mean {pl} epochs",
        color=theme.PROXY_COLOR,
    )
    ax.bar(
        [i + width / 2 for i in x],
        mean_ft,
        width,
        label=f"mean {tl} epochs",
        color=theme.REFERENCE_COLOR,
    )
    if cap is not None and not (isinstance(cap, float) and cap != cap):
        ax.axhline(
            cap,
            color=theme.STATUS_COLORS["diverged"],
            linestyle="--",
            label=f"proxy cap = {int(cap)}",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(heads)
    ax.set_ylabel("mean epochs")
    ax.set_title(f"Early-stopping epochs: {pl} vs {tl} per head")
    ax.legend()
    return _save(fig, out_dir, "epochs_vs_head")


def plot_head_rank_agreement(triples: list[Triple], out_dir: str | Path) -> Path:
    """(#4) Heatmap of the head x head Spearman rho matrix over proxy r2.

    -> ``head_rank_agreement.png``. Low off-diagonal rho means head choice changes the proxy
    ranking substantially.
    """
    from mlsys.analysis.tables import head_rank_agreement_matrix

    plt, sns = _setup()
    df = head_rank_agreement_matrix(triples)
    matrix = df.set_index("head") if "head" in df.columns else df

    fig, ax = plt.subplots(figsize=(max(4, len(matrix) * 1.5), max(4, len(matrix) * 1.2)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=theme.get_cmap_r2(),
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Spearman rho"},
    )
    ax.set_title("Head x head proxy-rank agreement (Spearman rho over proxy r2)")
    return _save(fig, out_dir, "head_rank_agreement")


def plot_proxy_timing_share(triples: list[Triple], out_dir: str | Path) -> Path:
    """(#5) Horizontal 100%-stacked bar: proxy pass timing substep % share per head.

    -> ``proxy_timing_share.png``. The deck's exact viz -- shows inference dominates 58-65%.
    """
    from mlsys.analysis.tables import frozen_timing_share_table
    from mlsys.analysis.theme import SUBSTEP_COLORS, SUBSTEP_HATCHES

    plt, _ = _setup()
    ft_table = frozen_timing_share_table(triples)
    if ft_table.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no timing data", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, out_dir, "proxy_timing_share")

    pct_cols = [
        "prepare_model_pct",
        "prepare_data_pct",
        "inference_pct",
        "train_head_pct",
        "eval_pct",
    ]
    labels = ["prepare_model_s", "prepare_data_s", "inference_s", "train_head_s", "eval_s"]
    heads = ft_table["head"].tolist()
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(7, len(heads) * 2), 4))
    left = np.zeros(len(heads))
    for i, (col, label) in enumerate(zip(pct_cols, labels, strict=True)):
        vals = ft_table[col].to_numpy(dtype=float)
        ax.barh(
            heads,
            vals,
            left=left,
            color=SUBSTEP_COLORS[i],
            hatch=SUBSTEP_HATCHES[i],
            label=label,
        )
        for j, (v, offset) in enumerate(zip(vals, left, strict=True)):
            if v > 4:
                ax.text(offset + v / 2, j, f"{v:.0f}%", ha="center", va="center", fontsize=8)
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of total proxy wall-time")
    ax.set_title("Proxy timing substep share per head")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out_dir, "proxy_timing_share")


def plot_value_frontier(triples: list[Triple], out_dir: str | Path) -> Path:
    """(#6) Scatter: proxy inference_s (x) vs proxy r² (y) at the widest head.

    → ``value_frontier.png``. Models annotated with their name. Shows the inference-cost vs
    quality trade-off for the backbone pool. Uses the widest head — inference_s is
    backbone-bound (head-independent) and the widest head gives the best r² cut.
    """
    from mlsys.analysis.tables import value_frontier_table

    plt, _ = _setup()
    vf = value_frontier_table(triples)
    if vf.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return _save(fig, out_dir, "value_frontier")

    widest_head = triples[-1].head if triples else "widest"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        vf["proxy_inference_s"],
        vf["proxy_r2"],
        color=theme.PROXY_COLOR,
        marker=theme.PROXY_MARKER,
        s=70,
        zorder=3,
    )
    for _, row in vf.iterrows():
        ax.annotate(
            str(row["model"]),
            (row["proxy_inference_s"], row["proxy_r2"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("proxy inference_s (backbone encode cost, head-independent)")
    ax.set_ylabel("proxy r²")
    ax.set_title(f"Inference value-frontier — widest head ({widest_head})")
    return _save(fig, out_dir, "value_frontier")

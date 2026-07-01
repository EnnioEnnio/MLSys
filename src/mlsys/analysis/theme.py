"""Centralised visual theme — one source of truth for colours, markers, and colormaps.

Every semantic entity (frozen/finetune pass, ok/diverged/skipped status, timing substep,
delta direction) gets one fixed colour + marker used across *all* plots. Palette sampled
from the intermediate deck: cream background #F5EFE6, deep burgundy #6B1F2A.

All constants are module-level (no lazy-import needed — only stdlib types here).
Heavy matplotlib/seaborn objects (LinearSegmentedColormap) are created inside
``get_cmap_r2`` / ``get_binary_cmap`` so the module stays importable without matplotlib.
``apply_theme(plt, sns)`` is called from ``_setup`` in plots.py.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- palette roots
BG = "#F5EFE6"  # cream background
GRID = "#D8CCBB"  # taupe grid lines
INK = "#3A2A2E"  # body text / axis labels

# --------------------------------------------------------------------------- pass colours
PASS_COLORS: dict[str, str] = {
    "frozen": "#C9A66B",  # warm tan
    "finetune": "#6B1F2A",  # deep burgundy
}
PASS_MARKERS: dict[str, str] = {
    "frozen": "o",
    "finetune": "s",
}

# Role-based aliases: proxy = cheap ranking pass, truth = expensive ground-truth pass.
# Use these instead of PASS_COLORS["frozen"] / PASS_COLORS["finetune"] in plot code so
# that the colour mapping stays correct for non-frozen/finetune comparison modes (e.g. r1/r3).
PROXY_COLOR: str = PASS_COLORS["frozen"]
TRUTH_COLOR: str = PASS_COLORS["finetune"]
PROXY_MARKER: str = PASS_MARKERS["frozen"]
TRUTH_MARKER: str = PASS_MARKERS["finetune"]

# --------------------------------------------------------------------------- status colours
STATUS_COLORS: dict[str, str] = {
    "ok": "#4A5A6A",  # slate
    "diverged": "#B5402F",  # brick-red
    "skipped": "#9B9B9B",  # grey
}
STATUS_MARKERS: dict[str, str] = {
    "ok": "o",
    "diverged": "X",
    "skipped": "s",
}

# ---------------------------------------------------------------------- substep colours / hatches
# Cream → burgundy ramp for the five timing substeps (same order as the CSV columns).
SUBSTEP_KEYS = ["prepare_model_s", "prepare_data_s", "inference_s", "train_head_s", "eval_s"]
SUBSTEP_COLORS: list[str] = ["#D9C7A3", "#C99B5E", "#B5604E", "#8C3340", "#6B1F2A"]
SUBSTEP_HATCHES: list[str] = ["", "//", "..", "xx", "\\\\"]

# --------------------------------------------------------------------------- delta colours
DELTA_COLORS: dict[str, str] = {
    "pos": "#5C7A4A",  # muted green
    "neg": "#B5402F",  # brick-red
}


def get_cmap_r2():
    """Cream → burgundy sequential colormap for r² heatmaps (replaces viridis)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("cmap_r2", [BG, "#6B1F2A"])


def get_binary_cmap():
    """Two-colour map [ok-slate, diverged-brick-red] for the divergence-map heatmap."""
    import seaborn as sns

    return sns.color_palette([STATUS_COLORS["ok"], STATUS_COLORS["diverged"]])


def apply_theme(plt, sns) -> None:  # type: ignore[no-untyped-def]
    """Set cream background, taupe grid, and the default seaborn palette.

    Call this from ``_setup`` in plots.py so the look is uniform without per-plot rcParams.
    """
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "axes.edgecolor": GRID,
            "grid.color": GRID,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        },
    )
    sns.set_palette([PASS_COLORS["frozen"], PASS_COLORS["finetune"]])

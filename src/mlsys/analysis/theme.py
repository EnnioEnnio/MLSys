"""Centralised visual theme — one source of truth for colours, markers, and colormaps.

Every semantic entity (frozen/finetune pass, ok/diverged/skipped status, timing substep,
delta direction, head width) gets one fixed colour + marker used across *all* plots. Palette
rooted in the intermediate deck's warm tan/burgundy pair (#C9A66B / #6B1F2A) on a plain white
background.

All constants are module-level (no lazy-import needed — only stdlib types here).
Heavy matplotlib/seaborn objects (LinearSegmentedColormap) are created inside
``get_cmap_r2`` / ``get_binary_cmap`` so the module stays importable without matplotlib.
``apply_theme(plt, sns)`` is called from ``_setup`` in plots.py (and the side-scripts in
``scripts/`` that render their own figures).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- palette roots
BG = "#FFFFFF"  # white background
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

# Role-based aliases: proxy = cheap ranking pass, reference = second pass (ground truth only
# for frozen/finetune pairs). Use these instead of PASS_COLORS["frozen"] /
# PASS_COLORS["finetune"] in plot code so the colour mapping stays correct for
# non-frozen/finetune comparison modes (e.g. r1/r3).
PROXY_COLOR: str = PASS_COLORS["frozen"]
REFERENCE_COLOR: str = PASS_COLORS["finetune"]
PROXY_MARKER: str = PASS_MARKERS["frozen"]
REFERENCE_MARKER: str = PASS_MARKERS["finetune"]

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


# Okabe & Ito (2008) categorical palette — the standard colorblind-safe choice for scientific
# plots with several *unordered* categories (used for head-indexed lines/bars). A single-hue
# lightness ramp (the old tan->burgundy interpolation) reads fine as a legend swatch but
# collapses into near-identical greys/browns once several lines overlap on the same axes, so
# head-indexed plots use this instead of PASS_COLORS. Ordered by on-white contrast; yellow is
# last because it's the one Okabe-Ito colour that's hard to see on a white background.
HEAD_PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
]
HEAD_MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]


def get_head_colors(n: int) -> list[str]:
    """First ``n`` colours of the Okabe-Ito palette (cycles if ``n`` exceeds its length)."""
    return [HEAD_PALETTE[i % len(HEAD_PALETTE)] for i in range(n)]


def get_head_markers(n: int) -> list[str]:
    """First ``n`` markers paired 1:1 with :func:`get_head_colors` (colour + shape encoding)."""
    return [HEAD_MARKERS[i % len(HEAD_MARKERS)] for i in range(n)]


def get_binary_cmap():
    """Two-colour map [ok-slate, diverged-brick-red] for the divergence-map heatmap."""
    import seaborn as sns

    return sns.color_palette([STATUS_COLORS["ok"], STATUS_COLORS["diverged"]])


# ------------------------------------------------------------------ paper (print) preset
# Geometry measured from the rendered report PDF (`pdfimages -list`: px / rendered ppi), not
# assumed from the class file. Authoring a figure at exactly its final printed size and
# including it with `width=\columnwidth` / `width=\textwidth` means scale == 1.0, so a 9 pt
# matplotlib font is 9 pt on the page — the supervisor's "same size as the LaTeX caption".
COLUMN_W_IN = 3.335  # \columnwidth  = 240 pt
TEXT_W_IN = 7.0  # \textwidth    = 504 pt (two-column span)
PAPER_FONT_PT = 9.0  # acmart sigconf body / caption size

# Module-level toggle rather than a threaded parameter: `apply_theme` already works by
# mutating global rcParams, and every one of the 26 plot functions calls it. Flipping one
# flag here beats adding a `paper=` argument to all of them.
_PAPER = False


def is_paper() -> bool:
    """True when the paper (print) preset is active — see :func:`apply_theme`."""
    return _PAPER


def title(ax, text: str) -> None:  # type: ignore[no-untyped-def]
    """Set an axes title, suppressed in paper mode.

    In the report the LaTeX ``\\caption`` carries the description, so an in-plot title is
    duplicated text that also steals vertical space from a page-limited body. Outside paper
    mode the title stays, because ``SUMMARY.md`` needs to be readable standalone.
    """
    if not _PAPER:
        ax.set_title(text)


def size(default: tuple[float, float], paper: tuple[float, float]) -> tuple[float, float]:
    """Pick a figsize: ``default`` for SUMMARY.md, ``paper`` for print.

    Paper sizes must be the *final printed* size (``COLUMN_W_IN`` or ``TEXT_W_IN`` wide) so
    LaTeX never rescales them.
    """
    return paper if _PAPER else default


def annot_kws() -> dict[str, float]:
    """``annot_kws`` for ``sns.heatmap`` — sizes in-cell numbers, which rcParams misses."""
    return {"size": PAPER_FONT_PT} if _PAPER else {}


def fig_ext() -> str:
    """Vector PDF for the paper, PNG for SUMMARY.md."""
    return "pdf" if _PAPER else "png"


def apply_theme(plt, sns, paper: bool = False) -> None:  # type: ignore[no-untyped-def]
    """Set cream background, taupe grid, and the default seaborn palette.

    Call this from ``_setup`` in plots.py so the look is uniform without per-plot rcParams.
    ``paper=True`` additionally switches typography to the print preset: every text element
    at :data:`PAPER_FONT_PT`, thinner rules, and Type 42 font embedding.
    """
    global _PAPER
    _PAPER = paper

    # `object` values: the colour keys are str but the paper block adds ints and floats.
    rc: dict[str, object] = {
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "axes.edgecolor": GRID,
        "grid.color": GRID,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
    if paper:
        pt = PAPER_FONT_PT
        rc |= {
            # One size for everything — the supervisor's guideline is caption parity, and a
            # figure with three different text sizes reads as three different figures.
            "font.size": pt,
            "axes.titlesize": pt,
            "axes.labelsize": pt,
            "xtick.labelsize": pt,
            "ytick.labelsize": pt,
            "legend.fontsize": pt,
            "legend.title_fontsize": pt,
            "figure.titlesize": pt,
            # matplotlib's PDF backend defaults to Type 3 fonts, which ACM/VLDB/IEEE PDF
            # checkers reject; the compiled paper is PDF/A-2b, which additionally requires
            # every font embedded. Type 42 (TrueType) subsets satisfy both.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Rules scaled for a 3.3 in figure rather than an 8 in one.
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.borderaxespad": 0.3,
            "legend.handlelength": 1.4,
            "legend.columnspacing": 1.0,
            "legend.labelspacing": 0.3,
        }
    sns.set_theme(style="whitegrid", context="notebook", rc=rc)
    sns.set_palette([PASS_COLORS["frozen"], PASS_COLORS["finetune"]])

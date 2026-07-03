"""Orchestrate the whole analysis: per-head folders, a comparison folder, and ``SUMMARY.md``.

``analyze_experiment`` is the entrypoint behind ``mlsys analyze``. It is crash-tolerant by
design (this command exists for recovery):

* missing ``*_regret.csv`` for a head → recompute it and write it back into the folder;
* a whole head config missing → skip that head, warn, still produce everything the surviving
  heads support;
* a frozen/finetune CSV missing for a head → skip just that head's folder, warn.

``SUMMARY.md`` is written in a **fixed section order** (0 metadata → 1 frozen → 2 finetune →
3 frozen-vs-finetune → 4 regret → 5 RQ2 bottlenecks → 6 RQ1/RQ2 synthesis stubs) regardless
of the order artifacts were generated, because Claude reads it top-down to write the report
prose. Section 6 is templated: every number is filled in as ``**metric:** value  <!-- prose:
-->`` so Claude writes narrative over given values rather than re-deriving them from plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mlsys.analysis import loader, plots, tables
from mlsys.analysis.regret_recompute import recompute_regret

if TYPE_CHECKING:
    import pandas as pd

    from mlsys.analysis.loader import Triple
    from mlsys.analysis.tables import TripleSummary

log = logging.getLogger(__name__)


@dataclass
class PerHead:
    """Everything one head config contributes to the report (typed so ty is happy)."""

    table: pd.DataFrame
    summary: TripleSummary
    png: dict[str, Path]
    dir: Path


@dataclass
class Comparison:
    """Cross-head tables + plots, plus their on-disk PNG paths."""

    dir: Path
    png: dict[str, Path]
    proxy_matrix: pd.DataFrame
    gt_matrix: pd.DataFrame
    per_head: pd.DataFrame
    diverged: pd.DataFrame
    cost: pd.DataFrame
    # § 7 tables
    proxy_distribution: pd.DataFrame
    head_gain: pd.DataFrame
    epochs: pd.DataFrame
    head_rank_agreement: pd.DataFrame
    proxy_timing_share: pd.DataFrame
    value_frontier: pd.DataFrame


def _ensure_regret_csv(tf: loader._TripleFiles, triple: Triple) -> None:
    """Crash recovery: if the triple had no ``*_regret.csv``, recompute + write it back."""
    if "regret" in tf.paths:
        return
    role = loader.resolve_role_pair(tf.paths)
    if role is None:
        return
    proxy_kind = role[0]
    proxy_path = tf.paths[proxy_kind]
    regret_name = proxy_path.name.replace(f"_{proxy_kind}.csv", "_regret.csv")
    regret_path = proxy_path.with_name(regret_name)
    curve = recompute_regret(triple.proxy, triple.gt)
    curve.to_csv(regret_path, index=False)
    triple.regret = curve
    tf.paths["regret"] = regret_path
    log.warning("recomputed missing regret curve → %s", regret_path)


def _rel(png: Path, base: Path) -> str:
    return png.relative_to(base).as_posix()


def _load_surviving(experiment_dir: Path) -> tuple[list[Triple], list[str]]:
    """Load every head that has a recognised proxy/truth pair; recompute missing regret.

    Returns ``(triples, skipped)`` where ``skipped`` are human-readable skip reasons for the
    metadata section.  Accepts any pair registered in ``loader._ROLE_PAIRS`` (e.g. frozen +
    finetune, or r1 + r3).
    """
    triples: list[Triple] = []
    skipped: list[str] = []
    for tf in loader.discover_triples(experiment_dir):
        if loader.resolve_role_pair(tf.paths) is None:
            msg = (
                f"{tf.head} (run {tf.run_id}): no recognised proxy/truth pair "
                f"in kinds {sorted(tf.paths)}"
            )
            log.warning("skipping head — %s", msg)
            skipped.append(msg)
            continue
        triple = loader.load_triple(tf)
        _ensure_regret_csv(tf, triple)
        triples.append(triple)
    return triples, skipped


def analyze_experiment(experiment_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    """Build every artifact + ``SUMMARY.md`` for an experiment folder. Returns the SUMMARY path."""
    experiment_dir = Path(experiment_dir)
    out_dir = Path(out_dir) if out_dir else experiment_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    triples, skipped = _load_surviving(experiment_dir)
    if not triples:
        raise RuntimeError(
            f"no analysable head found in {experiment_dir} (need a frozen+finetune CSV pair)"
        )

    per_head = {t.head: _build_per_head(t, out_dir) for t in triples}
    comparison = _build_comparison(triples, out_dir)

    summary = _assemble_summary(experiment_dir, triples, skipped, per_head, comparison, out_dir)
    summary_path = out_dir / "SUMMARY.md"
    summary_path.write_text(summary)
    log.info("wrote %s", summary_path)
    return summary_path


def _build_per_head(triple: Triple, out_dir: Path) -> PerHead:
    """Per-triple tables + plots into ``<out_dir>/<head>/``. Returns paths + summary."""
    head_dir = out_dir / triple.head
    head_dir.mkdir(parents=True, exist_ok=True)
    table = tables.per_triple_table(triple)
    tables.write_table(table, head_dir / "tables")
    summary = tables.per_triple_summary(triple)

    png = {
        "r2_proxy_vs_gt": plots.plot_r2_proxy_vs_gt(triple, head_dir),
        "proxy_scatter": plots.plot_proxy_scatter(triple, head_dir),
        "r2_delta": plots.plot_r2_delta(triple, head_dir),
        "regret_curve": plots.plot_regret_curve(triple, head_dir),
        "gt_spearman_vs_r2": plots.plot_gt_spearman_vs_r2(triple, head_dir),
        "timing_stacked": plots.plot_timing_stacked(triple, head_dir),
        "peak_gpu_mem": plots.plot_peak_gpu_mem(triple, head_dir),
        "proxy_time_breakdown": plots.plot_proxy_time_breakdown(triple, head_dir),
    }
    return PerHead(table=table, summary=summary, png=png, dir=head_dir)


def _build_comparison(triples: list[Triple], out_dir: Path) -> Comparison:
    """Cross-head tables + plots into ``<out_dir>/comparison/``."""
    comp_dir = out_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    proxy_matrix = tables.head_model_r2_matrix(triples, "proxy")
    gt_matrix = tables.head_model_r2_matrix(triples, "gt")
    per_head = tables.per_head_summary_table(triples)
    diverged = tables.diverged_models_table(triples)
    cost = tables.cost_table(triples)
    tables.write_table(proxy_matrix, comp_dir / "proxy_r2_matrix")
    tables.write_table(gt_matrix, comp_dir / "gt_r2_matrix")
    tables.write_table(per_head, comp_dir / "per_head_summary")
    tables.write_table(diverged, comp_dir / "diverged_models")
    tables.write_table(cost, comp_dir / "cost_table")

    # §7 tables
    proxy_distribution = tables.frozen_distribution_table(triples)
    head_gain = tables.head_gain_table(triples)
    epochs = tables.epochs_table(triples)
    head_rank_agreement = tables.head_rank_agreement_matrix(triples)
    proxy_timing_share = tables.frozen_timing_share_table(triples)
    value_frontier = tables.value_frontier_table(triples)
    tables.write_table(proxy_distribution, comp_dir / "proxy_distribution")
    tables.write_table(head_gain, comp_dir / "head_gain")
    tables.write_table(epochs, comp_dir / "epochs_table")
    tables.write_table(head_rank_agreement, comp_dir / "head_rank_agreement")
    tables.write_table(proxy_timing_share, comp_dir / "proxy_timing_share")
    tables.write_table(value_frontier, comp_dir / "value_frontier")

    png = {
        "regret_curves_by_head": plots.plot_regret_curves_by_head(triples, comp_dir),
        "regret_at1_vs_head": plots.plot_regret_at1_vs_head(triples, comp_dir),
        "best_r2_vs_head": plots.plot_best_r2_vs_head(triples, comp_dir),
        "heatmap_proxy_r2": plots.plot_heatmap_proxy_r2(triples, comp_dir),
        "heatmap_gt_r2": plots.plot_heatmap_gt_r2(triples, comp_dir),
        "divergence_map": plots.plot_divergence_map(triples, comp_dir),
        "proxy_rank_spearman_vs_head": plots.plot_proxy_rank_spearman_vs_head(triples, comp_dir),
        "cost_vs_head": plots.plot_cost_vs_head(triples, comp_dir),
        # §7 plots
        "epochs_vs_head": plots.plot_epochs_vs_head(triples, comp_dir),
        "head_rank_agreement": plots.plot_head_rank_agreement(triples, comp_dir),
        "proxy_timing_share": plots.plot_proxy_timing_share(triples, comp_dir),
        "value_frontier": plots.plot_value_frontier(triples, comp_dir),
    }
    return Comparison(
        dir=comp_dir,
        png=png,
        proxy_matrix=proxy_matrix,
        gt_matrix=gt_matrix,
        per_head=per_head,
        diverged=diverged,
        cost=cost,
        proxy_distribution=proxy_distribution,
        head_gain=head_gain,
        epochs=epochs,
        head_rank_agreement=head_rank_agreement,
        proxy_timing_share=proxy_timing_share,
        value_frontier=value_frontier,
    )


def _img(png: Path, base: Path, caption: str = "") -> str:
    return f"![{caption}]({_rel(png, base)})\n"


def _assemble_summary(
    experiment_dir: Path,
    triples: list[Triple],
    skipped: list[str],
    per_head: dict[str, PerHead],
    comparison: Comparison,
    out_dir: Path,
) -> str:
    md = tables.df_to_markdown
    heads = [t.head for t in triples]
    dataset = str(triples[0].proxy["dataset"].iloc[0]) if "dataset" in triples[0].proxy else "?"
    pool_size = len(triples[0].models)

    proxy_label = triples[0].proxy_label if triples else "proxy"
    truth_label = triples[0].truth_label if triples else "truth"

    parts: list[str] = [f"# Analysis — {experiment_dir.name}\n"]

    # 0. metadata
    parts.append("## 0. Experiment metadata\n")
    parts.append(f"- **dataset:** {dataset}\n")
    parts.append(f"- **pool size:** {pool_size} models\n")
    parts.append(f"- **heads found:** {', '.join(heads)}\n")
    parts.append(f"- **heads skipped:** {'; '.join(skipped) if skipped else 'none'}\n")
    parts.append(f"- **proxy pass:** {proxy_label}\n")
    parts.append(f"- **truth pass:** {truth_label}\n")

    # 1. proxy pass results
    parts.append(f"\n## 1. {proxy_label} results (cheap proxy)\n")
    for t in triples:
        ph = per_head[t.head]
        cols = ["model", "proxy_r2", "proxy_mse", "proxy_mae", "proxy_spearman"]
        parts.append(f"### Head {t.head}\n")
        parts.append(md(ph.table[cols]))
        parts.append(_img(ph.png["r2_proxy_vs_gt"], out_dir))
        parts.append(_img(ph.png["proxy_scatter"], out_dir))

    # 2. truth pass results (with divergence / spearman-vs-r2 story)
    n_diverged_any = sum(1 for t in triples for d in t.diverged.values() if d)
    diverged_guardrail = (
        f"> **Caveat:** {n_diverged_any} model/head combinations have {truth_label} r² < 0 "
        f"(training instability — backbone diverged). Their {truth_label} ground truth and the "
        "regret derived from it are unreliable; caveat accordingly when interpreting regret "
        "numbers that involve these models.\n"
        if n_diverged_any > 0
        else ""
    )
    parts.append(f"\n## 2. {truth_label} results (ground truth)\n")
    if diverged_guardrail:
        parts.append(diverged_guardrail)
    for t in triples:
        ph = per_head[t.head]
        cols = [
            "model",
            "gt_r2",
            "gt_spearman",
            "diverged",
            "gt_skipped",
            "gt_epochs",
        ]
        parts.append(f"### Head {t.head}\n")
        parts.append(md(ph.table[cols]))
        parts.append(_img(ph.png["gt_spearman_vs_r2"], out_dir))

    # 3. proxy vs truth comparison
    parts.append(f"\n## 3. {proxy_label} vs {truth_label} comparison\n")
    parts.append(f"### {proxy_label} r² (model x head)\n")
    parts.append(md(comparison.proxy_matrix))
    parts.append(f"### {truth_label} r² (model x head)\n")
    parts.append(md(comparison.gt_matrix))
    cpng = comparison.png
    parts.append(_img(cpng["heatmap_proxy_r2"], out_dir))
    parts.append(_img(cpng["heatmap_gt_r2"], out_dir))
    parts.append(_img(cpng["divergence_map"], out_dir))
    parts.append(_img(cpng["best_r2_vs_head"], out_dir))
    for t in triples:
        parts.append(_img(per_head[t.head].png["r2_delta"], out_dir))

    # 4. regret
    parts.append("\n## 4. Regret\n")
    parts.append(md(comparison.per_head))
    parts.append(_img(cpng["regret_curves_by_head"], out_dir))
    parts.append(_img(cpng["regret_at1_vs_head"], out_dir))
    parts.append(_img(cpng["proxy_rank_spearman_vs_head"], out_dir))
    for t in triples:
        parts.append(_img(per_head[t.head].png["regret_curve"], out_dir))

    # 5. RQ2 bottlenecks
    parts.append("\n## 5. RQ2 — bottlenecks (timing + GPU memory)\n")
    if proxy_label == "frozen" and truth_label == "finetune":
        parts.append(
            "Frozen cost splits across `inference_s` (encode) + `train_head_s` (head fit); "
            "finetune fuses inference into the joint loop so `inference_s = 0` and "
            "`train_head_s` is the end-to-end finetune cost.\n"
        )
    else:
        parts.append(
            f"Both passes ({proxy_label} and {truth_label}) split cost across the same "
            "substeps: `inference_s` (backbone encode) + `train_head_s` (head fit).\n"
        )
    parts.append(_img(cpng["cost_vs_head"], out_dir))
    for t in triples:
        ph = per_head[t.head]
        parts.append(f"### Head {t.head}\n")
        parts.append(_img(ph.png["timing_stacked"], out_dir))
        parts.append(_img(ph.png["peak_gpu_mem"], out_dir))
        parts.append(_img(ph.png["proxy_time_breakdown"], out_dir))

    # 6. synthesis stubs (templated numbers; prose for Claude)
    parts.append(_synthesis_section(triples, per_head, comparison, proxy_label, truth_label))

    # 7. distribution, ranking stability & cost (new gaps from the deck)
    parts.append(_section7(triples, comparison, out_dir))

    # Every part already carries its own trailing newline (and section markers their own leading
    # blank line), so join with "" — "\n".join would double every blank line.
    return "".join(parts)


def _section7(
    triples: list[Triple],
    comparison: Comparison,
    out_dir: Path,
) -> str:
    """§7: distribution, ranking stability & cost — six sub-sections, one per deck gap."""
    md = tables.df_to_markdown
    cpng = comparison.png
    lines: list[str] = ["\n## 7. Distribution, ranking stability & cost\n"]

    # 7.1 Spread-collapse (#1)
    lines.append("### 7.1 Proxy r² spread per head\n")
    lines.append(md(comparison.proxy_distribution))
    dist = comparison.proxy_distribution
    if not dist.empty:
        std_vals = dist["std_proxy_r2"].tolist()
        std_str = " → ".join(f"{v:.2f}" for v in std_vals)
        n_neg_total = int(dist["n_negative"].sum())
        lines.append(
            f"- **std_proxy_r2 across heads:** {std_str}  <!-- prose: spread-collapse story -->"
        )
        lines.append(
            f"- **total n_negative frozen r² (all heads):** {n_neg_total}  <!-- prose: -->\n"
        )

    # 7.2 Per-model gain narrowest→widest (#2)
    lines.append("### 7.2 Per-model Δ frozen r² (narrowest → widest head)\n")
    lines.append(md(comparison.head_gain))
    gain = comparison.head_gain
    if not gain.empty:
        top = gain.iloc[0]
        lines.append(
            f"- **biggest gainer:** {top['model']} (+{top['gain']:.4f})  <!-- prose: -->\n"
        )

    # 7.3 Early-stopping epochs (#3)
    lines.append("### 7.3 Early-stopping epochs\n")
    lines.append(md(comparison.epochs))
    lines.append(_img(cpng["epochs_vs_head"], out_dir))
    ep = comparison.epochs
    if not ep.empty and "n_proxy_at_cap" in ep.columns:
        at_cap_str = ", ".join(
            f"{row['head']}={int(row['n_proxy_at_cap'])}/{len(triples[0].models)}"
            for _, row in ep.iterrows()
        )
        lines.append(
            f"- **n_proxy_at_cap per head:** {at_cap_str}  <!-- prose: early-stop story -->\n"
        )

    # 7.4 Head x head rank agreement (#4)
    lines.append("### 7.4 Head x head frozen-r2 rank agreement (Spearman rho)\n")
    lines.append(md(comparison.head_rank_agreement))
    lines.append(_img(cpng["head_rank_agreement"], out_dir))
    hrm = comparison.head_rank_agreement
    if len(hrm) > 1:
        # off-diagonal minimum (ignoring the index/head column)
        numeric = hrm.drop(columns=["head"], errors="ignore").to_numpy(dtype=float)
        import numpy as np

        mask = ~np.eye(len(numeric), dtype=bool)
        min_rho = float(numeric[mask].min()) if mask.any() else float("nan")
        lines.append(
            f"- **min off-diagonal rho:** {min_rho:.2f}  <!-- prose: ranking stability -->\n"
        )

    # 7.5 Proxy timing substep share (#5)
    lines.append("### 7.5 Proxy timing substep share\n")
    lines.append(md(comparison.proxy_timing_share))
    lines.append(_img(cpng["proxy_timing_share"], out_dir))
    fts = comparison.proxy_timing_share
    if not fts.empty and "inference_pct" in fts.columns:
        inf_min = float(fts["inference_pct"].min())
        inf_max = float(fts["inference_pct"].max())
        lines.append(
            f"- **inference_pct range:** {inf_min:.0f}-{inf_max:.0f}%"
            "  <!-- prose: backbone dominates -->\n"
        )

    # 7.6 Inference value-frontier (#6)
    widest_head = triples[-1].head if triples else "?"
    lines.append(
        f"### 7.6 Inference value-frontier (widest head: {widest_head})\n"
        f"> Note: frontier uses the widest head ({widest_head}) — "
        "`inference_s` is backbone-bound / head-independent so the cost column is comparable "
        "across any head; widest head gives the best r² cut.\n"
    )
    lines.append(md(comparison.value_frontier))
    lines.append(_img(cpng["value_frontier"], out_dir))

    return "\n".join(lines)


def _synthesis_section(
    triples: list[Triple],
    per_head: dict[str, PerHead],
    comparison: Comparison,
    proxy_label: str = "frozen",
    truth_label: str = "finetune",
) -> str:
    lines: list[str] = ["\n## 6. Synthesis (numbers filled in; prose for the writer)\n"]

    # --- RQ1 ---
    lines.append("### RQ1 — adapting model search to regression\n")
    for t in triples:
        s = per_head[t.head].summary
        lines.append(f"#### Head {t.head}\n")
        lines.append(f"- **regret@1:** {s.regret_at_1:.4f}  <!-- prose: -->")
        lines.append(f"- **normalized regret@1:** {s.normalized_regret_at_1:.4f}  <!-- prose: -->")
        lines.append(f"- **budget-to-zero:** {s.budget_to_zero}  <!-- prose: -->")
        lines.append(
            f"- **best {proxy_label} r²:** {s.best_proxy_r2:.4f} "
            f"({s.best_proxy_model})  <!-- prose: -->"
        )
        lines.append(
            f"- **best {truth_label} r²:** {s.best_gt_r2:.4f} "
            f"({s.best_gt_model})  <!-- prose: -->"
        )
        lines.append(f"- **diverged models:** {s.n_diverged}  <!-- prose: -->")
        lines.append(f"- **proxy rank Spearman:** {s.rank_spearman:.4f}  <!-- prose: -->\n")

    at1_by_head = ", ".join(f"{t.head}={per_head[t.head].summary.regret_at_1:.4f}" for t in triples)
    first = per_head[triples[0].head].summary.regret_at_1
    last = per_head[triples[-1].head].summary.regret_at_1
    trend = "decreasing" if last < first else "increasing" if last > first else "flat"
    lines.append(
        f"- **regret@1 vs head width:** {at1_by_head} ({trend} with width)  <!-- prose: -->\n"
    )

    # --- diverged-model story table ---
    diverged = comparison.diverged
    lines.append(f"#### Diverged-model story ({truth_label} r² < 0)\n")
    if len(diverged):
        lines.append(tables.df_to_markdown(diverged))
    else:
        lines.append("_No model diverged in any head._\n")

    # --- RQ2 ---
    lines.append("### RQ2 — where do the bottlenecks shift?\n")
    widest = triples[-1]
    ph_table = per_head[widest.head].table
    best_model = per_head[widest.head].summary.best_gt_model
    row = ph_table[ph_table["model"] == best_model].iloc[0]
    fz_cost = float(row["proxy_inference_s"]) + float(row["proxy_train_head_s"])
    ft_cost = float(row["gt_train_head_s"])
    mem_fz = float(row["proxy_peak_gpu_mem_mb"])
    mem_ft = float(row["gt_peak_gpu_mem_mb"])
    cost_ratio = f"{ft_cost / fz_cost:.1f}x" if fz_cost > 0 else "n/a"
    mem_ratio = f"{mem_ft / mem_fz:.1f}x" if mem_fz > 0 else "n/a"
    lines.append(f"For the best model (**{best_model}**, head {widest.head}):\n")
    lines.append(
        f"- **{truth_label}/{proxy_label} train cost ratio:** {cost_ratio} "
        f"({ft_cost:.0f}s vs {fz_cost:.0f}s)  <!-- prose: -->"
    )
    lines.append(
        f"- **{truth_label}/{proxy_label} peak GPU mem ratio:** {mem_ratio} "
        f"({mem_ft:.0f}MB vs {mem_fz:.0f}MB)  <!-- prose: -->\n"
    )

    # Encode (inference_s) is the *backbone-specific* cost — head fitting is the same head
    # everywhere — so the backbone spread reads off it, not off total_s (which dilutes).
    encode = ph_table[["model", "proxy_inference_s"]]
    cheapest = encode.loc[encode["proxy_inference_s"].idxmin()]
    priciest = encode.loc[encode["proxy_inference_s"].idxmax()]
    cheap_s = float(cheapest["proxy_inference_s"])
    pricey_s = float(priciest["proxy_inference_s"])
    spread_str = f"{pricey_s / cheap_s:.1f}x" if cheap_s > 0 else "n/a"
    lines.append(
        f"- **backbone encode-cost spread (head {widest.head}, {proxy_label} inference_s):** "
        f"{spread_str} — cheapest {cheapest['model']} {cheap_s:.0f}s vs "
        f"priciest {priciest['model']} {pricey_s:.0f}s  <!-- prose: -->\n"
    )

    return "\n".join(lines)

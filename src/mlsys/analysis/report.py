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
    frozen_matrix: pd.DataFrame
    finetune_matrix: pd.DataFrame
    per_head: pd.DataFrame
    diverged: pd.DataFrame
    cost: pd.DataFrame


def _ensure_regret_csv(tf: loader._TripleFiles, triple: Triple) -> None:
    """Crash recovery: if the triple had no ``*_regret.csv``, recompute + write it back."""
    if "regret" in tf.paths:
        return
    frozen_path = tf.paths["frozen"]
    regret_name = frozen_path.name.replace("_frozen.csv", "_regret.csv")
    regret_path = frozen_path.with_name(regret_name)
    curve = recompute_regret(triple.frozen, triple.finetune)
    curve.to_csv(regret_path, index=False)
    triple.regret = curve
    tf.paths["regret"] = regret_path
    log.warning("recomputed missing regret curve → %s", regret_path)


def _rel(png: Path, base: Path) -> str:
    return png.relative_to(base).as_posix()


def _load_surviving(experiment_dir: Path) -> tuple[list[Triple], list[str]]:
    """Load every head that has both frozen+finetune; recompute missing regret. Warn + skip rest.

    Returns ``(triples, skipped)`` where ``skipped`` are human-readable skip reasons for the
    metadata section.
    """
    triples: list[Triple] = []
    skipped: list[str] = []
    for tf in loader.discover_triples(experiment_dir):
        if "frozen" not in tf.paths or "finetune" not in tf.paths:
            missing = [k for k in ("frozen", "finetune") if k not in tf.paths]
            msg = f"{tf.head} (run {tf.run_id}): missing {missing}"
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
        "r2_frozen_vs_finetune": plots.plot_r2_frozen_vs_finetune(triple, head_dir),
        "proxy_scatter": plots.plot_proxy_scatter(triple, head_dir),
        "r2_delta": plots.plot_r2_delta(triple, head_dir),
        "regret_curve": plots.plot_regret_curve(triple, head_dir),
        "finetune_spearman_vs_r2": plots.plot_finetune_spearman_vs_r2(triple, head_dir),
        "timing_stacked": plots.plot_timing_stacked(triple, head_dir),
        "peak_gpu_mem": plots.plot_peak_gpu_mem(triple, head_dir),
        "frozen_time_breakdown": plots.plot_frozen_time_breakdown(triple, head_dir),
    }
    return PerHead(table=table, summary=summary, png=png, dir=head_dir)


def _build_comparison(triples: list[Triple], out_dir: Path) -> Comparison:
    """Cross-head tables + plots into ``<out_dir>/comparison/``."""
    comp_dir = out_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    frozen_matrix = tables.head_model_r2_matrix(triples, "frozen")
    finetune_matrix = tables.head_model_r2_matrix(triples, "finetune")
    per_head = tables.per_head_summary_table(triples)
    diverged = tables.diverged_models_table(triples)
    cost = tables.cost_table(triples)
    tables.write_table(frozen_matrix, comp_dir / "frozen_r2_matrix")
    tables.write_table(finetune_matrix, comp_dir / "finetune_r2_matrix")
    tables.write_table(per_head, comp_dir / "per_head_summary")
    tables.write_table(diverged, comp_dir / "diverged_models")
    tables.write_table(cost, comp_dir / "cost_table")

    png = {
        "regret_curves_by_head": plots.plot_regret_curves_by_head(triples, comp_dir),
        "regret_at1_vs_head": plots.plot_regret_at1_vs_head(triples, comp_dir),
        "best_r2_vs_head": plots.plot_best_r2_vs_head(triples, comp_dir),
        "heatmap_frozen_r2": plots.plot_heatmap_frozen_r2(triples, comp_dir),
        "heatmap_finetune_r2": plots.plot_heatmap_finetune_r2(triples, comp_dir),
        "divergence_map": plots.plot_divergence_map(triples, comp_dir),
        "proxy_rank_spearman_vs_head": plots.plot_proxy_rank_spearman_vs_head(triples, comp_dir),
        "cost_vs_head": plots.plot_cost_vs_head(triples, comp_dir),
    }
    return Comparison(
        dir=comp_dir,
        png=png,
        frozen_matrix=frozen_matrix,
        finetune_matrix=finetune_matrix,
        per_head=per_head,
        diverged=diverged,
        cost=cost,
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
    dataset = str(triples[0].frozen["dataset"].iloc[0]) if "dataset" in triples[0].frozen else "?"
    pool_size = len(triples[0].models)

    parts: list[str] = [f"# Analysis — {experiment_dir.name}\n"]

    # 0. metadata
    parts.append("## 0. Experiment metadata\n")
    parts.append(f"- **dataset:** {dataset}\n")
    parts.append(f"- **pool size:** {pool_size} models\n")
    parts.append(f"- **heads found:** {', '.join(heads)}\n")
    parts.append(f"- **heads skipped:** {'; '.join(skipped) if skipped else 'none'}\n")

    # 1. frozen
    parts.append("\n## 1. Frozen results (cheap proxy)\n")
    for t in triples:
        ph = per_head[t.head]
        cols = ["model", "frozen_r2", "frozen_mse", "frozen_mae", "frozen_spearman"]
        parts.append(f"### Head {t.head}\n")
        parts.append(md(ph.table[cols]))
        parts.append(_img(ph.png["r2_frozen_vs_finetune"], out_dir))
        parts.append(_img(ph.png["proxy_scatter"], out_dir))

    # 2. finetune (with divergence / spearman-vs-r2 story)
    parts.append("\n## 2. Finetune results (ground truth)\n")
    for t in triples:
        ph = per_head[t.head]
        cols = [
            "model",
            "finetune_r2",
            "finetune_spearman",
            "diverged",
            "finetune_skipped",
            "finetune_epochs",
        ]
        parts.append(f"### Head {t.head}\n")
        parts.append(md(ph.table[cols]))
        parts.append(_img(ph.png["finetune_spearman_vs_r2"], out_dir))

    # 3. frozen-vs-finetune comparison
    parts.append("\n## 3. Frozen vs finetune comparison\n")
    parts.append("### Frozen r² (model x head)\n")
    parts.append(md(comparison.frozen_matrix))
    parts.append("### Finetune r² (model x head)\n")
    parts.append(md(comparison.finetune_matrix))
    cpng = comparison.png
    parts.append(_img(cpng["heatmap_frozen_r2"], out_dir))
    parts.append(_img(cpng["heatmap_finetune_r2"], out_dir))
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
    parts.append(
        "Frozen cost splits across `inference_s` (encode) + `train_head_s` (head fit); "
        "finetune fuses inference into the joint loop so `inference_s = 0` and "
        "`train_head_s` is the end-to-end finetune cost.\n"
    )
    parts.append(_img(cpng["cost_vs_head"], out_dir))
    for t in triples:
        ph = per_head[t.head]
        parts.append(f"### Head {t.head}\n")
        parts.append(_img(ph.png["timing_stacked"], out_dir))
        parts.append(_img(ph.png["peak_gpu_mem"], out_dir))
        parts.append(_img(ph.png["frozen_time_breakdown"], out_dir))

    # 6. synthesis stubs (templated numbers; prose for Claude)
    parts.append(_synthesis_section(triples, per_head, comparison))

    return "\n".join(parts)


def _synthesis_section(
    triples: list[Triple],
    per_head: dict[str, PerHead],
    comparison: Comparison,
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
            f"- **best frozen r²:** {s.best_frozen_r2:.4f} ({s.best_frozen_model})  <!-- prose: -->"
        )
        lines.append(
            f"- **best finetune r²:** {s.best_finetune_r2:.4f} "
            f"({s.best_finetune_model})  <!-- prose: -->"
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
    lines.append("#### Diverged-model story (rank kept, scale broken)\n")
    if len(diverged):
        lines.append(tables.df_to_markdown(diverged))
    else:
        lines.append("_No model diverged in any head._\n")

    # --- RQ2 ---
    lines.append("### RQ2 — where do the bottlenecks shift?\n")
    widest = triples[-1]
    ph_table = per_head[widest.head].table
    best_model = per_head[widest.head].summary.best_finetune_model
    row = ph_table[ph_table["model"] == best_model].iloc[0]
    fz_cost = float(row["frozen_inference_s"]) + float(row["frozen_train_head_s"])
    ft_cost = float(row["finetune_train_head_s"])
    mem_fz = float(row["frozen_peak_gpu_mem_mb"])
    mem_ft = float(row["finetune_peak_gpu_mem_mb"])
    lines.append(f"For the best model (**{best_model}**, head {widest.head}):\n")
    lines.append(
        f"- **finetune/frozen train cost ratio:** {ft_cost / fz_cost:.1f}x "
        f"({ft_cost:.0f}s vs {fz_cost:.0f}s)  <!-- prose: -->"
    )
    lines.append(
        f"- **finetune/frozen peak GPU mem ratio:** {mem_ft / mem_fz:.1f}x "
        f"({mem_ft:.0f}MB vs {mem_fz:.0f}MB)  <!-- prose: -->\n"
    )

    # Encode (inference_s) is the *backbone-specific* cost — head fitting is the same head
    # everywhere — so the model2vec-vs-transformer spread (the ~10x story) reads off it, not
    # off frozen_total_s (which head training dilutes).
    encode = ph_table[["model", "frozen_inference_s"]]
    cheapest = encode.loc[encode["frozen_inference_s"].idxmin()]
    priciest = encode.loc[encode["frozen_inference_s"].idxmax()]
    cheap_s = float(cheapest["frozen_inference_s"])
    pricey_s = float(priciest["frozen_inference_s"])
    # Guard the ratio: a 0s cheapest encode (a hypothetical zero-cost backbone) would divide
    # by zero — fall back to "n/a" but still report the absolute spread.
    spread_str = f"{pricey_s / cheap_s:.1f}x" if cheap_s > 0 else "n/a"
    lines.append(
        f"- **backbone encode-cost spread (head {widest.head}, frozen inference_s):** "
        f"{spread_str} — cheapest {cheapest['model']} {cheap_s:.0f}s vs "
        f"priciest {priciest['model']} {pricey_s:.0f}s  <!-- prose: -->\n"
    )

    return "\n".join(lines)

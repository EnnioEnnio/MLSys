"""Merge SLURM job-array fragments into one ``full_eval``-shaped run directory.

An array run writes one fragment per task (``runs/<id>/<id>_task_<n>/results.jsonl``,
one model each). :func:`consolidate_run` merges them into ``runs/<id>/results.jsonl``,
recomputes ``regret.json`` over the merged pool exactly as a single-node ``full_eval``
would have, and exports analysis-ready CSVs in the filename grammar ``mlsys analyze``
expects.

Cluster-safe by construction: stdlib ``json``/``csv``/``pathlib`` only — the core
install has no pandas (do not reuse ``analysis/regret_recompute.py`` here).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsys.models.registry import load_specs
from mlsys.search.regret import RegretSummary, summarize_regret, write_regret_json

log = logging.getLogger(__name__)

_TASK_DIR_RE = re.compile(r"_task_(\d+)$")

# Row keys consumed positionally by flatten_row (or dropped: the per-epoch curves);
# everything else passes through as extras.
_CORE_KEYS = frozenset(
    {
        "model",
        "dataset",
        "strategy",
        "metrics",
        "timing",
        "head_train_curve",
        "head_val_curve",
        "grad_norm_curve",
        "grad_norm_max_curve",
    }
)


@dataclass(frozen=True)
class ConsolidationResult:
    """What :func:`consolidate_run` produced (paths are ``None`` when skipped)."""

    dataset: str
    run_name: str
    rows: list[dict[str, Any]]  # frozen block then finetune block, registry order
    results_path: Path
    regret_path: Path | None  # None with allow_partial on an incomplete pool
    summary: RegretSummary | None
    csv_paths: list[Path]


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten a results.jsonl row into the W&B results-table layout.

    Dict-based equivalent of ``full_eval._result_row``: ``metrics``/``timing`` are
    inlined, the per-epoch curves dropped, and any extras (e.g. ``finetune_skipped``)
    pass through — so a locally exported CSV is byte-compatible with the
    W&B-table-download path.
    """
    out: dict[str, Any] = {
        "model": row["model"],
        "dataset": row["dataset"],
        "strategy": row["strategy"],
        **row.get("metrics", {}),
        **row.get("timing", {}),
    }
    for key, value in row.items():
        if key not in _CORE_KEYS:
            out[key] = value
    return out


def consolidate_run(
    run_dir: str | Path,
    *,
    hidden: int | None = None,
    cleanup: bool = False,
    allow_partial: bool = False,
) -> ConsolidationResult:
    """Merge ``<run_dir>/*_task_*/results.jsonl`` fragments into ``<run_dir>/``.

    Rows pass through verbatim (only ``model``/``strategy``/``dataset``/``metrics.r2``
    are read), deduped keep-last on ``(model, strategy)``, ordered frozen-block-then-
    finetune-block by ``load_specs()`` index — reproducing a single-node ``full_eval``'s
    row order and proxy-ranking tie-break exactly. ``hidden`` only shapes the exported
    run name's head token (``FCH`` / ``MLP_<w>``).

    With no fragments, falls back to an already-consolidated ``<run_dir>/results.jsonl``
    (idempotent after ``cleanup``); raises if neither exists. An incomplete pool
    (frozen/finetune model sets differ) raises unless ``allow_partial``, which merges
    but skips regret. ``cleanup`` removes the task dirs after all writes succeed.
    """
    run_dir = Path(run_dir)
    rows = _load_fragment_rows(run_dir)
    dataset = _single_dataset(rows)
    rows = _dedupe_and_order(rows)

    frozen = [r for r in rows if r["strategy"] == "frozen"]
    finetune = [r for r in rows if r["strategy"] == "finetune"]
    missing = _missing_pairs(frozen, finetune)
    if missing and not allow_partial:
        raise ValueError(
            f"incomplete pool in {run_dir}: missing rows for {missing}; "
            "retry the failed tasks or pass allow_partial to merge without regret"
        )

    results_path = _write_results_jsonl(run_dir, rows)

    summary: RegretSummary | None = None
    regret_path: Path | None = None
    if not missing:
        summary = summarize_regret(
            dataset,
            {r["model"]: r["metrics"]["r2"] for r in frozen},
            {r["model"]: r["metrics"]["r2"] for r in finetune},
        )
        regret_path = write_regret_json(run_dir, summary)
        log.info("wrote %s (proxy ranking: %s)", regret_path, ", ".join(summary.proxy_ranking))
    else:
        log.warning("skipping regret.json: missing rows for %s", missing)

    # The run name mirrors a single-node full_eval run so the exported CSVs drop
    # straight into a results/<experiment>/ folder with zero renaming.
    from mlsys.cli.main import _wandb_run_name

    n_models = len({r["model"] for r in rows})
    run_name = _wandb_run_name(run_dir.name, dataset, "full_eval", n_models, hidden)

    csv_paths = [
        _write_pass_csv(run_dir / f"{run_name}_frozen.csv", frozen),
        _write_pass_csv(run_dir / f"{run_name}_finetune.csv", finetune),
    ]
    if summary is not None:
        csv_paths.append(_write_regret_csv(run_dir / f"{run_name}_regret.csv", summary))

    if cleanup:
        for task_dir, _ in _task_dirs(run_dir):
            shutil.rmtree(task_dir)
            log.info("removed fragment dir %s", task_dir)

    return ConsolidationResult(
        dataset=dataset,
        run_name=run_name,
        rows=rows,
        results_path=results_path,
        regret_path=regret_path,
        summary=summary,
        csv_paths=csv_paths,
    )


def _task_dirs(run_dir: Path) -> list[tuple[Path, int]]:
    """Fragment dirs with their numeric task index, sorted numerically (not lexically)."""
    found = []
    for path in run_dir.glob("*_task_*"):
        m = _TASK_DIR_RE.search(path.name)
        if path.is_dir() and m:
            found.append((path, int(m.group(1))))
    return sorted(found, key=lambda pair: pair[1])


def _load_fragment_rows(run_dir: Path) -> list[dict[str, Any]]:
    fragments = [d / "results.jsonl" for d, _ in _task_dirs(run_dir)]
    fragments = [f for f in fragments if f.is_file()]
    if not fragments:
        merged = run_dir / "results.jsonl"
        if merged.is_file():
            log.info("no fragments in %s; reusing consolidated %s", run_dir, merged.name)
            fragments = [merged]
        else:
            raise FileNotFoundError(
                f"no *_task_*/results.jsonl fragments and no consolidated results.jsonl "
                f"in {run_dir}"
            )
    rows: list[dict[str, Any]] = []
    for path in fragments:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"fragments in {run_dir} contain no result rows")
    return rows


def _single_dataset(rows: list[dict[str, Any]]) -> str:
    datasets = {r["dataset"] for r in rows}
    if len(datasets) != 1:
        raise ValueError(f"fragments disagree on dataset: {sorted(datasets)}")
    return next(iter(datasets))


def _dedupe_and_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe keep-last on (model, strategy); frozen block then finetune, registry order.

    Registry (models.yaml) order matters beyond cosmetics: the frozen block's insertion
    order is the proxy-ranking tie-break (stable sort), so this reproduces a single-node
    run's regret.json exactly. Unknown models sort last (warned), keeping their fragment
    order.
    """
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        deduped[(row["model"], row["strategy"])] = row

    registry_index = {name: i for i, name in enumerate(load_specs())}
    unknown = sorted({m for m, _ in deduped if m not in registry_index})
    if unknown:
        log.warning("models not in models.yaml ordered last: %s", ", ".join(unknown))

    def order(strategy: str) -> list[dict[str, Any]]:
        block = [r for (_, s), r in deduped.items() if s == strategy]
        # Stable sort → unknown models keep their fragment order at the tail.
        return sorted(block, key=lambda r: registry_index.get(r["model"], len(registry_index)))

    other = sorted({s for _, s in deduped if s not in ("frozen", "finetune")})
    if other:
        raise ValueError(f"unexpected strategy rows: {other}")
    return order("frozen") + order("finetune")


def _missing_pairs(frozen: list[dict[str, Any]], finetune: list[dict[str, Any]]) -> list[str]:
    """Model names missing their counterpart row (or everything, if a pass is empty)."""
    frozen_models = {r["model"] for r in frozen}
    finetune_models = {r["model"] for r in finetune}
    if not frozen_models or not finetune_models:
        return sorted(frozen_models | finetune_models) or ["<no rows>"]
    return sorted(frozen_models ^ finetune_models)


def _write_results_jsonl(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Atomic whole-file write (never append) — re-running consolidation is idempotent."""
    path = run_dir / "results.jsonl"
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=".results-", suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def _write_pass_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """One pass's rows as CSV, columns from the first row (the W&B table layout)."""
    flat = [flatten_row(r) for r in rows]
    columns = list(flat[0].keys()) if flat else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(flat)
    return path


def _write_regret_csv(path: Path, summary: RegretSummary) -> Path:
    """The regret curve in the same shape as ``mlsys regret --out``."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["budget", "regret", "normalized_regret"])
        for p in summary.curve:
            writer.writerow([p.budget, p.regret, p.normalized_regret])
    return path

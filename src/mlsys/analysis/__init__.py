"""Result-analysis + report generation for ``full_eval`` experiments.

Turns a folder of ``*_frozen.csv`` / ``*_finetune.csv`` / ``*_regret.csv`` dumps into a
``SUMMARY.md`` (tables + plots) ready to hand to Claude for prose. All heavy deps
(``pandas``/``matplotlib``/``seaborn``) are in the optional ``analysis`` group and imported
lazily inside functions — importing this package does **not** pull them in.

Entrypoints used by the CLI:

* :func:`analyze_experiment` — behind ``mlsys analyze <experiment_dir>``.
* :func:`recompute_to_files` — behind ``mlsys regret --frozen … --finetune …``.
"""

from __future__ import annotations

from mlsys.analysis.regret_recompute import recompute_to_files
from mlsys.analysis.report import analyze_experiment

__all__ = ["analyze_experiment", "recompute_to_files"]

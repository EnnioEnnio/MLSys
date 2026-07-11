"""CLI: ``python -m mlsys {search,consolidate,list-models,list-datasets,analyze,regret}``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from mlsys.datasets import load_dataset
from mlsys.datasets.registry import load_specs as load_dataset_specs
from mlsys.finetune import FinetuneConfig
from mlsys.head import ACTIVATIONS, HeadTrainConfig
from mlsys.models.registry import load_specs as load_model_specs
from mlsys.search.full_eval import STRATEGIES, run_strategy


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _setup_logging(verbose: bool) -> None:
    """Send progress logs to stderr. ``-v`` drops to DEBUG to also show per-substep timing.

    The root logger stays at WARNING so noisy third-party libraries (httpx,
    sentence_transformers, urllib3, …) don't flood the output with request logs;
    only the loggers we care about (our own ``mlsys.*`` plus ``wandb``) are
    lowered to INFO/DEBUG.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("mlsys").setLevel(logging.DEBUG if verbose else logging.INFO)
    # Keep W&B's run banner / sync messages (logged at INFO) visible.
    logging.getLogger("wandb").setLevel(logging.INFO)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlsys", description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging: show each candidate's per-substep timing as it runs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Run a model search over a dataset.")
    search.add_argument("--dataset", required=True, help="dataset name from config/datasets.yaml")
    search.add_argument(
        "--models",
        default=None,
        help="comma-separated model names; default = all from config/models.yaml",
    )
    search.add_argument(
        "--strategy",
        default="frozen",
        choices=STRATEGIES,
        help=(
            "frozen: train an FC head on the frozen backbone (cheap proxy ranking). "
            "finetune: unfreeze + train backbone+head jointly (ground truth). "
            "full_eval: run both over the pool and compute the regret-vs-budget curve."
        ),
    )
    search.add_argument(
        "--output-dir",
        default=None,
        help="run directory; default runs/<unix-ts>",
    )
    search.add_argument("--epochs", type=int, default=HeadTrainConfig.epochs)
    search.add_argument("--batch-size", type=int, default=64)
    search.add_argument(
        "--hidden",
        type=int,
        default=HeadTrainConfig.hidden,
        metavar="WIDTH",
        help=(
            "hidden-layer width of the head. Omit or 0 -> linear probe "
            "(in_dim -> 1). A positive value builds a 2-layer MLP "
            "(in_dim -> WIDTH -> ACT -> 1); WIDTH is the size of that "
            "intermediate layer (more units = more capacity)."
        ),
    )
    search.add_argument(
        "--activation",
        default=HeadTrainConfig.activation,
        choices=sorted(ACTIVATIONS),
        help="activation between the two layers of the MLP head "
        "(ignored for the linear probe, i.e. --hidden 0/omitted; default: %(default)s)",
    )
    search.add_argument("--device", default=None, help="cpu|cuda; default auto-detect")
    search.add_argument(
        "--head-repeats",
        type=int,
        default=3,
        metavar="N",
        help="train the linear head N times and average predictions to reduce ranking variance"
        " (default: 3)",
    )
    search.add_argument(
        "--finetune-epochs",
        type=int,
        default=FinetuneConfig.epochs,
        help="epochs for the finetune/full_eval joint loop (default: %(default)s)",
    )
    search.add_argument(
        "--warmup-epochs",
        type=int,
        default=FinetuneConfig.warmup_epochs,
        help="head-only warmup epochs with the backbone frozen before the finetune/full_eval "
        "joint loop (LP-FT; 0 = off, default: %(default)s)",
    )
    search.add_argument(
        "--finetune-lr",
        type=float,
        default=FinetuneConfig.backbone_lr,
        help="backbone learning rate for finetune/full_eval (default: %(default)s)",
    )
    search.add_argument(
        "--finetune-batch-size",
        type=int,
        default=FinetuneConfig.batch_size,
        help="batch size for the finetune/full_eval joint loop (default: %(default)s)",
    )
    search.add_argument(
        "--grad-clipping",
        type=float,
        default=FinetuneConfig.grad_clipping,
        metavar="MAX_NORM",
        help="clip the finetune/full_eval joint loop's gradients to this global L2 norm "
        "(0 = off; the pre-clip norm is measured and logged either way, "
        "default: %(default)s)",
    )
    search.add_argument("--wandb", action="store_true", help="opt-in W&B logging")
    search.add_argument(
        "--cache-embeddings",
        action="store_true",
        help="(v2) cache extracted embeddings on disk; currently stubbed",
    )

    list_models = sub.add_parser("list-models", help="Dump models.yaml entries.")
    list_models.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help="print only the bare model name at registry position N (for SLURM array tasks)",
    )
    list_models.add_argument(
        "--count",
        action="store_true",
        help="print only the pool size (single source of truth for the --array bound)",
    )
    sub.add_parser("list-datasets", help="Dump datasets.yaml entries.")

    consolidate = sub.add_parser(
        "consolidate",
        help="Merge a job array's *_task_*/results.jsonl fragments and recompute regret.json.",
    )
    consolidate.add_argument(
        "run_dir",
        help="experiment dir holding the *_task_* fragment dirs (runs/<ARRAY_JOB_ID>)",
    )
    consolidate.add_argument(
        "--hidden",
        type=int,
        default=HeadTrainConfig.hidden,
        metavar="WIDTH",
        help="head hidden width the tasks ran with; only shapes the exported "
        "run name's head token (FCH / MLP_<WIDTH>)",
    )
    consolidate.add_argument(
        "--cleanup",
        action="store_true",
        help="remove the *_task_* fragment dirs after a successful merge",
    )
    consolidate.add_argument(
        "--allow-partial",
        action="store_true",
        help="merge even if frozen/finetune rows are missing (skips regret.json)",
    )
    consolidate.add_argument(
        "--wandb",
        action="store_true",
        help="push one consolidated W&B run (results_frozen/results_finetune tables + regret)",
    )

    analyze = sub.add_parser(
        "analyze",
        help="Build tables + plots + SUMMARY.md from a full_eval experiment folder.",
    )
    analyze.add_argument(
        "experiment_dir",
        help="folder of *_frozen/*_finetune/*_regret CSVs (one full_eval experiment)",
    )
    analyze.add_argument(
        "--out-dir",
        default=None,
        help="where to write artifacts; default <experiment_dir>/analysis",
    )

    regret = sub.add_parser(
        "regret",
        help="Recompute a regret curve from a frozen + finetune CSV (crash recovery).",
    )
    regret.add_argument("--frozen", required=True, help="path to the *_frozen.csv")
    regret.add_argument("--finetune", required=True, help="path to the *_finetune.csv")
    regret.add_argument(
        "--out",
        default=None,
        help="write the budget,regret,normalized_regret CSV here (default: stdout)",
    )
    regret.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="also write a regret.json-shaped payload here",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    # Load a local .env (e.g. WANDB_API_KEY) before any --wandb branch. Guarded so
    # the tool still runs if python-dotenv isn't installed; on the cluster the key
    # comes from the exported shell env, not .env.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    args = _build_parser().parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    if args.command == "list-models":
        specs = list(load_model_specs().values())
        if args.count:
            print(len(specs))
            return 0
        if args.index is not None:
            # A bad SLURM --array bound must fail loudly so afterok blocks consolidation.
            if not 0 <= args.index < len(specs):
                print(
                    f"model index {args.index} out of range (pool size {len(specs)})",
                    file=sys.stderr,
                )
                return 1
            print(specs[args.index].name)
            return 0
        for spec in specs:
            print(
                f"{spec.name:30s}  loader={spec.loader:24s}  dim={spec.embedding_dim:5d}  "
                f"max_len={spec.max_length}  repo={spec.hf_repo}"
            )
        return 0

    if args.command == "list-datasets":
        for spec in load_dataset_specs().values():
            print(
                f"{spec.name:20s}  target={spec.target_column} ({spec.target_type})  "
                f"repo={spec.hf_repo}  splits={sorted(spec.splits)}"
            )
        return 0

    if args.command == "search":
        return _run_search(args)

    if args.command == "consolidate":
        return _run_consolidate(args)

    if args.command == "analyze":
        return _run_analyze(args)

    if args.command == "regret":
        return _run_regret(args)

    raise SystemExit(f"unknown command: {args.command}")


def _run_analyze(args: argparse.Namespace) -> int:
    # Heavy analysis deps (pandas/matplotlib/seaborn) are imported lazily inside the
    # package so config-only commands stay fast (project convention).
    from mlsys.analysis import analyze_experiment

    summary = analyze_experiment(args.experiment_dir, out_dir=args.out_dir)
    print(f"[mlsys] wrote analysis to {summary.parent} (see {summary})")
    return 0


def _run_regret(args: argparse.Namespace) -> int:
    from mlsys.analysis import recompute_to_files

    curve = recompute_to_files(
        args.frozen,
        args.finetune,
        out_csv=args.out,
        json_path=args.json_path,
    )
    if args.out:
        print(f"[mlsys] wrote regret curve to {args.out}")
    else:
        print(curve.to_csv(index=False), end="")
    if args.json_path:
        print(f"[mlsys] wrote regret json to {args.json_path}", file=sys.stderr)
    return 0


def _run_consolidate(args: argparse.Namespace) -> int:
    # Lazy import: consolidation is stdlib-only but pulls in the search package.
    from mlsys.search.consolidate import consolidate_run, flatten_row

    result = consolidate_run(
        args.run_dir,
        hidden=args.hidden,
        cleanup=args.cleanup,
        allow_partial=args.allow_partial,
    )
    print(f"[mlsys] wrote {len(result.rows)} rows to {result.results_path}")
    if result.regret_path is not None:
        print(f"[mlsys] wrote regret curve to {result.regret_path}")
    else:
        print("[mlsys] partial pool — regret.json skipped", file=sys.stderr)
    for path in result.csv_paths:
        print(f"[mlsys] wrote {path}")

    if args.wandb:
        import wandb  # type: ignore[import-not-found]

        from mlsys.search.full_eval import _log_regret_to_wandb

        run = wandb.init(
            entity="HPI_MLSys",
            project="mlsys-model-search",
            name=result.run_name,
            config={
                "dataset": result.dataset,
                "strategy": "full_eval",
                "hidden": args.hidden,
                "consolidated_from": args.run_dir,
            },
        )
        # Two tables, named exactly as a single-node full_eval's passes, so the
        # analysis CSV-download workflow keeps working unchanged.
        tables = (("frozen", "results_frozen"), ("finetune", "results_finetune"))
        for strategy, table_name in tables:
            flat = [flatten_row(r) for r in result.rows if r["strategy"] == strategy]
            if not flat:
                continue
            columns = list(flat[0].keys())
            data = [[row.get(c) for c in columns] for row in flat]
            wandb.log({table_name: wandb.Table(columns=columns, data=data)})
        if result.summary is not None:
            _log_regret_to_wandb(result.summary.curve)
        run.finish()
    return 0


def _wandb_run_name(
    run_id: str, dataset: str, strategy: str, num_models: int, hidden: int | None
) -> str:
    """Descriptive W&B run name mirroring the analysis filename grammar.

    ``<runid>_<dataset>_<strategy>_<num>_model_<HEAD>[_<width>]`` — e.g.
    ``2296342_wine_reviews_fulleval_16_model_MLP_512``. A linear probe stays bare
    (``..._model_FCH``, no width); an MLP carries its hidden width (``MLP_512``).
    The strategy's underscore is stripped (``full_eval`` -> ``fulleval``) so the
    token count is stable.
    """
    head = f"MLP_{hidden}" if hidden and hidden > 0 else "FCH"
    return f"{run_id}_{dataset}_{strategy.replace('_', '')}_{num_models}_model_{head}"


def _run_search(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"runs/{int(time.time())}")
    device = args.device or _default_device()

    if args.cache_embeddings:
        print(
            "[mlsys] --cache-embeddings is reserved for v2; the embedding cache store "
            "is not yet wired. Continuing without on-disk caching.",
            file=sys.stderr,
        )

    log = logging.getLogger("mlsys.cli")
    log.info(
        "search: dataset=%s strategy=%s device=%s output_dir=%s",
        args.dataset,
        args.strategy,
        device,
        output_dir,
    )

    dataset = load_dataset(args.dataset)
    model_names = [m.strip() for m in args.models.split(",")] if args.models else None
    head_cfg = HeadTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        activation=args.activation,
    )
    finetune_cfg = FinetuneConfig(
        epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        backbone_lr=args.finetune_lr,
        warmup_epochs=args.warmup_epochs,
        grad_clipping=args.grad_clipping,
    )

    wandb_run = None
    if args.wandb:
        import wandb  # type: ignore[import-not-found]

        num_models = len(model_names) if model_names else len(load_model_specs())
        wandb_run = wandb.init(
            entity="HPI_MLSys",
            project="mlsys-model-search",
            name=_wandb_run_name(
                output_dir.name, args.dataset, args.strategy, num_models, head_cfg.hidden
            ),
            config={
                "dataset": args.dataset,
                "strategy": args.strategy,
                "models": model_names,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": device,
                "hidden": head_cfg.hidden,
                "activation": head_cfg.activation,
                "finetune_epochs": finetune_cfg.epochs,
                "finetune_batch_size": finetune_cfg.batch_size,
                "finetune_backbone_lr": finetune_cfg.backbone_lr,
                "finetune_warmup_epochs": finetune_cfg.warmup_epochs,
                "finetune_grad_clipping": finetune_cfg.grad_clipping,
            },
        )

    records = run_strategy(
        args.strategy,
        dataset,
        output_dir=output_dir,
        model_names=model_names,
        device=device,
        batch_size=args.batch_size,
        head_config=head_cfg,
        finetune_config=finetune_cfg,
        head_repeats=args.head_repeats,
        wandb_run=wandb_run,
    )
    print(f"[mlsys] wrote {len(records)} rows to {output_dir / 'results.jsonl'}")
    if args.strategy == "full_eval":
        print(f"[mlsys] wrote regret curve to {output_dir / 'regret.json'}")
    if wandb_run is not None:
        wandb_run.finish()
    return 0

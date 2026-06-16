"""CLI: ``python -m mlsys {search,list-models,list-datasets}``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mlsys.datasets import load_dataset
from mlsys.datasets.registry import load_specs as load_dataset_specs
from mlsys.head import HeadTrainConfig
from mlsys.models.registry import load_specs as load_model_specs
from mlsys.search.full_eval import full_eval


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlsys", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Run a model search over a dataset.")
    search.add_argument("--dataset", required=True, help="dataset name from config/datasets.yaml")
    search.add_argument(
        "--models",
        default=None,
        help="comma-separated model names; default = all from config/models.yaml",
    )
    search.add_argument("--strategy", default="full_eval", choices=("full_eval",))
    search.add_argument(
        "--output-dir",
        default=None,
        help="run directory; default runs/<unix-ts>",
    )
    search.add_argument("--epochs", type=int, default=HeadTrainConfig.epochs)
    search.add_argument("--batch-size", type=int, default=64)
    search.add_argument("--device", default=None, help="cpu|cuda; default auto-detect")
    search.add_argument("--wandb", action="store_true", help="opt-in W&B logging")
    search.add_argument(
        "--cache-embeddings",
        action="store_true",
        help="(v2) cache extracted embeddings on disk; currently stubbed",
    )

    sub.add_parser("list-models", help="Dump models.yaml entries.")
    sub.add_parser("list-datasets", help="Dump datasets.yaml entries.")

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

    if args.command == "list-models":
        for spec in load_model_specs().values():
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

    raise SystemExit(f"unknown command: {args.command}")


def _run_search(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"runs/{int(time.time())}")
    device = args.device or _default_device()

    if args.cache_embeddings:
        print(
            "[mlsys] --cache-embeddings is reserved for v2; the embedding cache store "
            "is not yet wired. Continuing without on-disk caching.",
            file=sys.stderr,
        )

    dataset = load_dataset(args.dataset)
    model_names = [m.strip() for m in args.models.split(",")] if args.models else None
    head_cfg = HeadTrainConfig(epochs=args.epochs, batch_size=args.batch_size)

    wandb_run = None
    if args.wandb:
        import wandb  # type: ignore[import-not-found]

        wandb_run = wandb.init(
            entity="HPI_MLSys",
            project="mlsys-model-search",
            name=output_dir.name,
            config={
                "dataset": args.dataset,
                "strategy": args.strategy,
                "models": model_names,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": device,
            },
        )

    if args.strategy != "full_eval":
        raise SystemExit(f"unknown strategy {args.strategy!r}")

    records = full_eval(
        dataset,
        output_dir=output_dir,
        model_names=model_names,
        device=device,
        batch_size=args.batch_size,
        head_config=head_cfg,
        wandb_run=wandb_run,
    )
    print(f"[mlsys] wrote {len(records)} rows to {output_dir / 'results.jsonl'}")
    if wandb_run is not None:
        wandb_run.finish()
    return 0

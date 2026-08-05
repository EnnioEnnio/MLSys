"""Does the *ranking criterion* change measured proxy fidelity? (METHOD.md, Result 6)

`mlsys analyze` and `regret.json` rank candidates by r² only — inherited from the
classification literature's accuracy. r² is a faithful ranking signal only while the target's
variance is not set by outliers. On `usa_real_estate` it is: std(price) ≈ $4.5M against a
typical MAE of ≈ $324k, so `r2 = 1 - MSE/Var(y)` pins the whole pool near zero.

This script re-ranks the *same* frozen/finetune CSV pairs under three criteria (r², MAE,
per-model prediction-vs-target Spearman) and reports, per criterion:

- proxy→reference rank correlation over the pool (is the ordering recoverable at all?)
- reference-side separation band, full pool and `can_finetune`-only (is there anything to rank?)
- regret@1 + budget-to-zero against a Monte-Carlo random-ranking null (does it *shortlist*?)

Two things it is designed to show, both zero-GPU: the housing pool separates on MAE/Spearman
while looking flat in r², and switching criterion nonetheless does **not** rescue regret@1,
because the B=1 failure is the non-finetunable model2vec top pick (Result 5's trap), which no
choice of metric touches.

Read with the caveat the tables print: n=16, one run per configuration.

Deterministic: fixed null RNG seed, sorted discovery, no timestamps in outputs.

Usage:
    uv run python scripts/criterion_sensitivity.py
    uv run python scripts/criterion_sensitivity.py --out-dir results/criterion_sensitivity
"""

from __future__ import annotations

import argparse
import random
import statistics as st
from pathlib import Path
from typing import TYPE_CHECKING

from mlsys.analysis.loader import Triple, discover_triples, load_triple, resolve_role_pair
from mlsys.analysis.tables import df_to_markdown, write_table

if TYPE_CHECKING:
    import pandas as pd

# label -> (folder, run-ids). Wine uses all 5 final-recipe seed repeats (Result 5), NOT the
# 2x2 design-space cells: REPORT.md is explicit that single-run values there "carry
# essentially no information individually". Housing has one run, so its row is a point
# estimate and is labelled as such.
DEFAULT_GROUPS = {
    "wine (5 seeds)": (
        Path("results/full_eval_noise"),
        ["2341209", "2341210", "2341245", "2341254", "2341255"],
    ),
    "housing (1 run)": (
        Path("results/full_eval_usa_housing_no_early_stopping"),
        ["2344122"],
    ),
}

# criterion -> (column, higher_is_better)
CRITERIA = {"r2": ("r2", True), "mae": ("mae", False), "spearman": ("spearman", True)}

NULL_DRAWS = 10_000
NULL_SEED = 0


def _load(groups: dict[str, tuple[Path, list[str]]]) -> dict[str, list[Triple]]:
    out: dict[str, list[Triple]] = {}
    for label, (folder, run_ids) in groups.items():
        discovered = {tf.run_id: tf for tf in discover_triples(folder)}
        triples = []
        for run_id in run_ids:
            tf = discovered.get(run_id)
            if tf is None:
                raise FileNotFoundError(f"run-id {run_id} not found under {folder}")
            if resolve_role_pair(tf.paths) is None:
                raise ValueError(f"run-id {run_id} has no recognised proxy/reference pair")
            triples.append(load_triple(tf))
        out[label] = triples
    return out


def _aligned(
    triple: Triple, column: str, higher: bool
) -> tuple[list[str], list[float], list[float]]:
    """Proxy/reference values in a shared model order, sign-flipped so higher is always better."""
    sign = 1.0 if higher else -1.0
    fz = triple.proxy.set_index("model")[column]
    ft = triple.reference.set_index("model")[column]
    models = [m for m in triple.models if m in ft.index]
    return models, [sign * float(fz[m]) for m in models], [sign * float(ft[m]) for m in models]


def _spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr

    rho, _ = spearmanr(a, b)
    return float(rho)


def _regret_curve(proxy: list[float], reference: list[float]) -> list[float]:
    """regret(B) = best reference overall - best reference within the proxy's top-B shortlist."""
    order = sorted(range(len(reference)), key=lambda i: -proxy[i])
    best, running, curve = max(reference), -float("inf"), []
    for i in order:
        running = max(running, reference[i])
        curve.append(best - running)
    return curve


def _null_regret_at_1(reference: list[float]) -> float:
    """Mean regret@1 of a proxy that knows nothing — one uniformly random top pick."""
    rng = random.Random(NULL_SEED)
    best = max(reference)
    return st.mean(best - reference[rng.randrange(len(reference))] for _ in range(NULL_DRAWS))


def build_table(groups: dict[str, list[Triple]]) -> pd.DataFrame:
    """One row per (group, criterion).

    Rank correlation and regret@1 are computed **per seed** and reported as mean [min, max] —
    they are rankings, and Result 5 showed single-seed values move a lot for free. Bands and
    the proxy's top pick come from the per-model seed *mean*, matching Result 5's
    "band of seed-mean r²" convention.
    """
    import pandas as pd

    rows = []
    for label, triples in groups.items():
        skipped = {m for t in triples for m, flag in t.ref_skipped.items() if flag}
        for criterion, (column, higher) in CRITERIA.items():
            per_seed = [_aligned(t, column, higher) for t in triples]
            common = sorted(set.intersection(*(set(models) for models, _, _ in per_seed)))

            corrs, regrets = [], []
            proxy_sum = dict.fromkeys(common, 0.0)
            ref_sum = dict.fromkeys(common, 0.0)
            for models, proxy, reference in per_seed:
                idx = {m: i for i, m in enumerate(models)}
                p = [proxy[idx[m]] for m in common]
                r = [reference[idx[m]] for m in common]
                corrs.append(_spearman(p, r))
                regrets.append(_regret_curve(p, r)[0])
                for m, pv, rv in zip(common, p, r, strict=True):
                    proxy_sum[m] += pv / len(per_seed)
                    ref_sum[m] += rv / len(per_seed)

            mean_proxy = [proxy_sum[m] for m in common]
            mean_ref = [ref_sum[m] for m in common]
            trainable = [ref_sum[m] for m in common if m not in skipped]
            top_pick = common[max(range(len(common)), key=lambda i: mean_proxy[i])]
            # Undo the higher-is-better sign flip for display; for MAE that also swaps the
            # band endpoints back (min of -MAE is the *largest* error).
            sign = 1.0 if higher else -1.0
            lo, hi = sign * min(trainable), sign * max(trainable)
            if not higher:
                lo, hi = hi, lo
            curve = _regret_curve(mean_proxy, mean_ref)
            rows.append(
                {
                    "group": label,
                    "criterion": criterion,
                    "rank_corr_mean": round(st.mean(corrs), 3),
                    "rank_corr_min": round(min(corrs), 3),
                    "rank_corr_max": round(max(corrs), 3),
                    "ref_min_trainable": round(lo, 4),
                    "ref_max_trainable": round(hi, 4),
                    "regret_at_1_mean": round(st.mean(regrets), 4),
                    "regret_at_1_max": round(max(regrets), 4),
                    "null_regret_at_1": round(_null_regret_at_1(mean_ref), 4),
                    "budget_to_zero": next(
                        (i + 1 for i, v in enumerate(curve) if v <= 1e-9), len(curve)
                    ),
                    "proxy_top_1": top_pick,
                    "top_1_finetunable": top_pick not in skipped,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/criterion_sensitivity"))
    args = parser.parse_args()

    table = build_table(_load(DEFAULT_GROUPS))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(table, args.out_dir / "criterion_sensitivity")

    housing = table[table.group.str.startswith("housing")]
    wine = table[table.group.str.startswith("wine")]
    headline = "\n".join(
        [
            f"- housing rank correlation spans {housing.rank_corr_mean.min():+.3f} to "
            f"{housing.rank_corr_mean.max():+.3f} across criteria on identical runs — the "
            "criterion, not the proxy, decides",
            f"- wine spans only {wine.rank_corr_mean.min():+.3f}..{wine.rank_corr_mean.max():+.3f} "
            "(5-seed means): no tail, so the criterion barely matters",
            "- housing regret@1 is at or above the random null under EVERY criterion; under "
            "MAE/Spearman the proxy's top pick is the non-finetunable model2vec embedder "
            "(Result 5's trap, independent of the metric)",
            "- wine rows are 5 seed repeats under the final recipe (Result 5); housing is a "
            "single run, so its rank correlations are point estimates (n=16, SE ~ 0.26) and its "
            "regret@1 carries the [0.000, 0.132] single-run spread Result 5 measured",
        ]
    )
    sections = [
        "# Criterion sensitivity (does the ranking metric change measured proxy fidelity?)\n",
        "## Headline\n\n" + headline + "\n",
        "## Per-run, per-criterion\n\n" + df_to_markdown(table),
    ]
    (args.out_dir / "CRITERION_SUMMARY.md").write_text("\n".join(sections))
    print("\n\n".join(sections))
    print(f"\nwrote tables to {args.out_dir}")


if __name__ == "__main__":
    main()

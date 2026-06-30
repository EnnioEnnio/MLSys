"""Regret metric (SHiFT paper, REGRET.md): how much downstream quality a cheap search
proxy loses versus fine-tuning every model.

We score on **r2** (higher is better), so the per-model values here are r2 from the
*finetune* pass — the expensive ground truth ``t(m, D)``. With a single finetune run per
model (``head_repeats=1``), both expectations in the paper collapse to point estimates,
so the formula is just ``max_M t - max_{S} t`` (REGRET.md note 2).
"""

from __future__ import annotations

from dataclasses import dataclass


def regret(scores: dict[str, float], shortlist: list[str]) -> float:
    """``max_{m in M} t(m) - max_{s in S} t(s)`` — point-estimate regret (r2, higher better).

    ``scores`` maps every model in the pool ``M`` to its finetune r2; ``shortlist`` is the
    subset ``S`` the search returned. Regret >= 0; 0 means the shortlist held a model as good
    as the global best.
    """
    if not scores:
        raise ValueError("regret requires a non-empty scores dict")
    if not shortlist:
        raise ValueError("regret requires a non-empty shortlist")
    best_overall = max(scores.values())
    best_returned = max(scores[m] for m in shortlist)
    return best_overall - best_returned


@dataclass(frozen=True)
class RegretPoint:
    budget: int
    regret: float
    normalized_regret: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "budget": self.budget,
            "regret": self.regret,
            "normalized_regret": self.normalized_regret,
        }


def regret_curve(
    proxy_ranking: list[str],
    finetune_scores: dict[str, float],
) -> list[RegretPoint]:
    """Regret at every budget ``B = 1..|M|``.

    ``proxy_ranking`` is every model in ``M`` ordered by the cheap proxy (frozen r2, desc);
    ``finetune_scores`` is the finetune-r2 ground truth. For each ``B`` the shortlist is the
    top-``B`` of the proxy ranking. Returns absolute regret and a normalized variant
    (absolute / best achievable r2, REGRET.md design note) per budget.

    By construction the curve is non-increasing in ``B`` and 0 at ``B = |M|`` (the shortlist
    is the whole pool). Normalization is only meaningful when the best r2 is positive; when
    it isn't, ``normalized_regret`` is reported as 0.0.
    """
    if not proxy_ranking:
        raise ValueError("regret_curve requires a non-empty proxy_ranking")
    missing = [m for m in proxy_ranking if m not in finetune_scores]
    if missing:
        raise ValueError(f"proxy_ranking models missing from finetune_scores: {missing}")
    best_overall = max(finetune_scores.values())
    norm_denom = best_overall if best_overall > 0 else None

    points: list[RegretPoint] = []
    for budget in range(1, len(proxy_ranking) + 1):
        abs_regret = regret(finetune_scores, proxy_ranking[:budget])
        normalized = abs_regret / norm_denom if norm_denom is not None else 0.0
        points.append(RegretPoint(budget=budget, regret=abs_regret, normalized_regret=normalized))
    return points

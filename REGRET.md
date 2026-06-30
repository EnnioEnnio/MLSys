# The "Regret" Metric (SHiFT paper)

Source: *SHiFT: An Efficient, Flexible Search Engine for Transfer Learning*, Renggli et al., PVLDB 16(2), 2022. Section 2.3 (Model Search Strategies), page 306, Equation (1).

## What it measures

Regret measures **how much downstream accuracy you lose by trusting a search query's shortlist instead of fine-tuning every available model**. It is the gap between the best model you *could* have found and the best model the query actually handed you (after fine-tuning).

## Definition

```
regret = max_{m ∈ M} E[t(m, D)]  −  E[ max_{s ∈ Sₘ} t(s, D) ]
```

Where:

| Symbol | Meaning |
|--------|---------|
| `M` | The full pool of candidate models |
| `Sₘ ⊆ M` | The subset the query returned, with `\|Sₘ\| ≤ B` (the budget) |
| `D` | The downstream task dataset |
| `t(m, D)` | **Test accuracy after actually fine-tuning** model `m` on `D` — the expensive ground-truth signal, *not* the cheap proxy used for ranking |
| `E[·]` | Expectation over fine-tuning randomness (head re-init, SGD stochasticity) |

In words: *(best accuracy achievable if you fine-tuned every model) minus (best accuracy among the models the query handed you).*

- Regret ≥ 0 always.
- **Lower is better.**
- 0 means the shortlist contained a model as good as the global optimum.

## Key implementation notes

1. **It's `max` over the returned set, not `mean`.** A query is credited only with the single best fine-tuned model in `Sₘ`. This is deliberate — it rewards returning a *diverse* shortlist, since a budget `B > 1` only helps if the extra picks cover cases the top pick misses.

2. **The two expectations are asymmetric.**
   - First term: `max` of per-model *expected* accuracies (average each model first, then take the max).
   - Second term: *expectation of the max* (for each fine-tune run, take the max over the subset, then average).
   - These are not equal. If you only have one fine-tune run per model, both collapse to point estimates and the formula simplifies to `max_M t − max_{Sₘ} t`.

3. **Decouple the fine-tune table from the search.** Precompute `t(m, D)` for every model once (the `M × N` benchmark table in the paper's Figure 3). Evaluating regret for *any* query is then just two lookups — cheap. Don't re-fine-tune per query.

4. **The metric judges outcomes, not the proxy.** A query may rank with kNN accuracy, linear-probe accuracy, or task similarity — regret ignores all of that and scores only the fine-tuned result. This is what lets you compare wildly different strategies fairly.

5. **Budget interacts directly with regret.** `B = 1` is hardest; `B = |M|` trivially gives 0. Always report regret *at a fixed B*, or the numbers aren't comparable.

## Design suggestion for another system

The paper reports absolute regret in accuracy points (it cites a 43% gap from a bad pick). When comparing across datasets of differing difficulty, consider also a **normalized regret** (divide by the best achievable accuracy). Whichever you choose, state explicitly which one you're reporting.

## Minimal reference implementation

```python
def regret(finetune_acc: dict[str, float], returned_subset: list[str]) -> float:
    """
    finetune_acc:   {model_id: test_accuracy_after_finetuning} for the FULL pool M
    returned_subset: the model IDs the search query returned (Sₘ)

    Single-run version: each model has one fine-tune accuracy, so both
    expectations collapse to point estimates.
    """
    best_overall = max(finetune_acc.values())            # max_{m in M} t(m, D)
    best_returned = max(finetune_acc[m] for m in returned_subset)  # max_{s in Sₘ} t(s, D)
    return best_overall - best_returned
```

For the multi-run case, store a list of accuracies per model and replace the
point estimates with the appropriate averages (see note 2 above).
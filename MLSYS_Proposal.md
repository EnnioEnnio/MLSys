# Introduction

Given a recurring, well-defined prediction task and a target dataset, **model search** asks: out of a large pool of candidate base models, which one is the most promising starting point for fine-tuning? Answering this cheaply matters because fine-tuning every candidate to find out is prohibitively expensive, so practical systems rely on a cheap *proxy* score to rank candidates and fine-tune only the winners.

Existing model-search systems such as *Alsatian* and *SHiFT* were designed and evaluated for classical computer-vision and NLP **classification** tasks, where proxy scoring relies on cheap classifiers (linear probes, kNN) over extracted features. It is not obvious that the same procedures and the same cost structure carry over to **regression** tasks, where the target is a continuous value rather than a discrete label. Both the proxy-scoring step and its dominant costs may behave differently.

In this project, we investigate two research questions:

* **RQ1: How can existing model-search procedures be adapted to tasks beyond classical text and image classification?** Concretely, we adapt feature-based model search to **regression** over tabular data serialized to text, and study whether a frozen-backbone proxy ranking is a faithful stand-in for actually fine-tuning each candidate.
* **RQ2: How do the bottlenecks of model search shift under these adapted procedures?** *Alsatian* identified feature extraction (model loading, data preparation, inference) as the dominant cost. Does this still hold in the regression setting, and how does the cost of the cheap frozen-backbone proxy compare to the cost of full fine-tuning?

We use the baseline approaches from *Alsatian* and *SHiFT* as our reference implementations and conduct a focused empirical study.

# Approach

Our project proceeds as follows:

1. **Define benchmark scenarios.** We use recurring **regression** workloads over tabular data. Each dataset provides a fixed, human-supplied ground-truth target (e.g., predicting a wine-review score or a real-estate price), so **no LLM-as-Judge is involved** — evaluation is a standard regression metric (MSE/MAE/R²/Spearman) against known labels. Tabular rows are rendered to text via a per-dataset `text_template`, turning each scenario into a text-regression task suitable for text encoders.
2. **Curate a candidate model pool.** We collect small candidate encoders from HuggingFace (sentence-transformers, plain `transformers` encoders, and model2vec static embedders) and integrate them with the *Alsatian*-style feature-based model search. As in the reference work, we deliberately **omit Alsatian's block-level model store**, since its benefits would not justify the integration overhead at our scale.
3. **Adapt the proxy-scoring step (RQ1).** For each candidate we attach a fresh fully-connected regression head on top of the **frozen** backbone, train only the head against ground truth, and score it on the test split. This cheap frozen-backbone score is our proxy ranking signal — the regression analogue of the linear-probe / kNN proxies used for classification in *Alsatian* and *SHiFT*.
4. **Fine-tune candidate models to obtain a ground-truth ranking.** For each scenario we also **unfreeze the backbone and fine-tune backbone+head jointly**, evaluating on a held-out test set. The resulting ranking serves two purposes: (i) it is the reference against which we validate the frozen-backbone proxy from step 3, and (ii) the per-model fine-tuning runtimes feed into the cost analysis. We quantify the gap between the two rankings as **regret**: how much test-set quality the cheap frozen-backbone proxy ranking loses versus actually fine-tuning every model, as a function of the fine-tuning budget.
5. **Bottleneck analysis (RQ2).** Following the methodology of *Alsatian*'s Section 3, we measure the runtime breakdown of the adapted search pipeline across its substeps — preparing the model, preparing the data, running inference, training the head, and evaluation — for varying model sizes and dataset sizes. We compare the frozen proxy pass against the full fine-tuning pass, and against the breakdown reported for the classical classification setting, to identify where the bottleneck sits and how it shifts.
6. **Synthesize findings.** We characterize the conditions under which frozen-backbone proxy search remains a faithful ranking signal for regression, quantify its regret versus full fine-tuning, describe where the runtime bottlenecks lie, and provide guidance on which parts of the search pipeline most warrant optimization.

# Related Work

**Strassenburg et al., *Alsatian: Optimizing Model Search for Deep Transfer Learning*.** Optimizes feature-based model search through partial model access, caching of intermediate inference results, and search-order planning. Provides the model-search methodology and cost breakdown we build on and compare against.

**Renggli et al., *SHiFT: An Efficient, Flexible Search Engine for Transfer Learning*.** Introduces a query language and cost-based optimizer for transfer-learning model search, including successive-halving as a system-level optimization. Provides an alternative search backend and inspiration for incremental execution.

**Renggli et al., *Which Model to Transfer? Finding the Needle in the Growing Haystack*.** Empirically compares task-agnostic, task-aware, and hybrid model-search strategies, informing our choice of proxy-scoring method for diverse model pools.

**Guo et al., *Sommelier: Curating DNN Models for the Masses*.** Indexes models by functional equivalence and resource profile, relevant for pruning the candidate pool before search.

# Anticipated Difficulties

* **Faithfulness of the frozen-backbone proxy.** A frozen backbone with only a trained head may rank candidates differently from full fine-tuning. If the proxy ranking is inconsistent with the fine-tuned ranking, its regret will be high. Our reference ranking — obtained by fully fine-tuning all candidate models — is exactly what lets us measure this.
* **Defining a fair regression comparison.** Candidate encoders differ in embedding dimensionality and output scale. Making the head-training and fine-tuning setups comparable across heterogeneous backbones (learning rate, head width, early stopping) without per-model tuning is non-trivial.
* **Heterogeneous candidate models.** Models in the pool differ in tokenization, pooling, input format (e.g., prefix conventions), and whether they can be fine-tuned at all (static model2vec embedders cannot). Uniform integration into the search pipeline without per-model special cases is non-trivial and may inflate "prepare model" time.
* **Attributing runtime fairly.** In the fine-tuning pass, inference is fused into the joint training loop rather than run as a separate step, so the substep breakdown differs structurally from the frozen pass. Comparing the two cost profiles cleanly, and against the classical classification breakdown, requires care in how substep time is attributed.
* **Workload selection.** The chosen datasets must be diverse enough to be informative, but tractable enough that small candidate encoders have a realistic chance of producing a useful regression signal. Picking these well is itself a small experimental-design problem.

# Introduction

Large Language Models (LLMs) have made machine learning broadly accessible: with a well-crafted prompt, developers can build applications without ML expertise. This         convenience is expensive – flagship LLMs charge significantly per token, exhibit high inference latency, and consume substantial energy. Prior work has shown that for narrow, recurring tasks, much smaller specialized models can match or exceed LLM accuracy after brief fine-tuning.

Just-in-time model replacement (JITR), recently proposed in *Poodle*, exploits this gap by monitoring LLM traffic, identifying recurring tasks, and transparently replacing the LLM with a cheaper fine-tuned surrogate. A central component of any JITR system is **model search**: given a recurring task and a target dataset, identify the most promising base model from a large pool of candidates for subsequent fine-tuning. The quality of this step largely determines whether replacement is feasible at all.

Existing model-search systems such as *Alsatian* and *SHiFT* were designed and evaluated for classical computer-vision and NLP classification tasks, where proxy scoring relies on cheap classifiers (linear probes, kNN) over extracted features. JITR-style workloads differ in two important ways. First, the tasks are not necessarily classification – they may be open-ended generation, extraction, or summarization. Second, in the absence of human-labeled ground truth, evaluation depends on LLM-as-Judge scoring rather than a simple classifier. Both changes have unclear consequences for the design and bottlenecks of model search.

In this project, we investigate two research questions:

* **RQ1: How can existing model-search procedures be adapted to tasks beyond classical text and image classification?**  
* **RQ2: How do the bottlenecks of model search shift under these adapted procedures?** In particular, *Alsatian* identified feature extraction (model loading, data preparation, inference) as the dominant cost. Does this still hold when proxy scoring is performed via LLM-as-Judge instead of a cheap classifier?

We will use the baseline approaches from *Alsatian* and *Poodle* as our reference implementations and conduct a focused empirical study.

# Approach

Our project proceeds as follows:

1. **Define benchmark scenarios.** Select a set of recurring-task workloads that go beyond classical classification – for example, structured information extraction, short-form summarization, or open-ended question answering. For each scenario, fix one SOTA LLM as the reference and assemble a training/evaluation dataset of LLM-generated request-response pairs.  
2. **Curate a candidate model pool.** Collect small candidate models from public repositories (e.g., HuggingFace) suited to the chosen scenarios, and integrate them with the *Alsatian* baseline model search. For this project, we deliberately **omit Alsatian's block-level model store**, since its benefits would not justify the integration overhead at our scale.  
3. **Adapt the proxy-scoring step (RQ1).** Replace the cheap classifier-based proxy scoring used in *Alsatian* and *SHiFT* with an LLM-as-Judge scoring step that ranks candidate models by the quality of their outputs on a small sample of the target dataset. We will design and compare a small number of variants – e.g., direct LLM scoring of generations, embedding-similarity to LLM responses, and hybrid approaches.  
4. **Fine-tune candidate models to obtain a ground-truth ranking.** For each scenario, fine-tune the candidate models on LLM-generated labels and evaluate them on a held-out test set. The resulting ranking serves two purposes: (i) it is the reference against which we validate the proxy-scoring variants from step 3, and (ii) the per-model fine-tuning runtimes feed into the cost analysis.  
5. **Bottleneck analysis (RQ2).** Following the methodology of *Alsatian*'s Section 3, measure the runtime breakdown of the adapted search pipeline across its substeps – preparing the model, preparing the data, running inference, and proxy scoring – for varying model sizes and dataset sizes. Compare against the breakdown reported for the classical setting and identify where the bottleneck shifts.  
6. **Synthesize findings.** Characterize the conditions under which adapted search procedures remain effective, describe the new bottlenecks introduced by LLM-as-Judge scoring, and provide guidance for future JITR systems on which parts of the search pipeline most warrant optimization.

# Related Work

**Strassenburg et al., *Poodle: Seamlessly Scaling Down Large Language Models with Just-in-Time Model Replacement*.** The vision paper introducing JITR and outlining the conceptual workflow this project implements end-to-end.

**Strassenburg et al., *Alsatian: Optimizing Model Search for Deep Transfer Learning*.** Optimizes feature-based model search through partial model access, caching of intermediate inference results, and search-order planning. Provides the model store and search backend we plan to build on.

**Renggli et al., *SHiFT: An Efficient, Flexible Search Engine for Transfer Learning*.** Introduces a query language and cost-based optimizer for transfer-learning model search, including successive-halving as a system-level optimization. Provides an alternative search backend and inspiration for incremental execution.

**Renggli et al., *Which Model to Transfer? Finding the Needle in the Growing Haystack*.** Empirically compares task-agnostic, task-aware, and hybrid model-search strategies, informing our choice of proxy-scoring method for diverse model pools.

**Guo et al., *Sommelier: Curating DNN Models for the Masses*.** Indexes models by functional equivalence and resource profile, relevant for pruning the candidate pool before search.

**Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*.** Foundational reference for knowledge distillation, which we apply to transfer LLM responses into the surrogate.

# Anticipated Difficulties

* Reliability of LLM-as-Judge. LLM evaluators are known to exhibit biases (position, verbosity, self-preference). If the judge is unreliable, the resulting model ranking may be inconsistent with true fine-tuning performance. We will need a reference ranking – likely obtained by fully fine-tuning all candidate models on a small workload – to validate the proxy.  
* Defining task suitability. Without a clean accuracy metric, defining "the best model for this task" is non-trivial. We may need to combine multiple judge signals or restrict to tasks with a more structured output format.  
* Cost of proxy scoring. Each judge call is itself an LLM inference. If proxy scoring becomes the dominant cost, this affects not only RQ2's analysis but also the practical viability of JITR – an interesting result either way.  
* Heterogeneous candidate models. Models in the pool differ in tokenization, input format, and prompt conventions. Uniform integration into the search pipeline without per-model special cases is non-trivial and may inflate "prepare model" time.  
* Workload selection. The chosen scenarios must be diverse enough to be informative, but simple enough that small candidate models have a realistic chance of succeeding. Picking these well is itself a small experimental design problem.
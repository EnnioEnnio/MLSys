## Ennio:
### Conception & Planning

- Conducted model and dataset research (surveyed encoder architectures and
  regression benchmarks/papers) that informed the model pool and dataset
  selection in `config/models.yaml`/`config/datasets.yaml`.
- Co-defined the project's scope and goals with the supervisor, drafting and
  later revising the project proposal as the project pivoted from the
  original JIT-Model-Replacement/LLM-as-Judge idea to the regression
  model-search pipeline.
- Planned and coordinated the team's work across the project lifecycle.
- Kept project hygiene, documentation, set up tooling and guardrails for the team.

### Core Pipeline Implementation

- Designed and built the model search pipeline: dataset/model registry, 
  adapter system for HuggingFace, backbones, FC-head training loop, 
  and evaluation metrics.
- Implemented the `finetune` and `full_eval` strategies plus the
  SHiFT-style regret metric comparing the frozen-backbone proxy ranking
  against full fine-tuning ground truth.
- Built the deterministic analysis/report generator (`mlsys.analysis`) that
  turns proxy/reference CSV pairs into tables, plots, and a summary for the
  report.
- Added the SLURM scripts including job-array execution mode (one model per task) and the
  result-consolidation logic that merges fragments and recomputes regret
  cluster-side.
- Conducted several experiments and implemented target standardization 
  (z-scoring on train-split stats) for both frozen and finetune heads, 
  LP-FT head-warmup for the finetune strategy plus warmup-epoch sweep experiments,
  gradient clipping for the joint fine-tune loop, and CLI flags for
  early-stop patience.
- General pipeline maintenance: W&B run naming/streaming, per-model step
  metrics, experiment reorganization, pinning SLURM runs to code snapshots,
  and pruning the model pool.
- ... etc. for a more detailed list of contributions, see the PRs in the repo.

### Cluster Engineering & Experiments

- Debugged the first cluster scripts through a run of targeted fixes: CUDA
  OOM in the multi-model search loop and on CLS-pooled encoders, an apex
  FusedRMSNorm crash on fp16 T5 checkpoints, `trust_remote_code` support for
  gte/nomic models, and `datasets` version/install issues on NGC containers.
- Conducted the majority of the experiments run on the cluster, including
  the head-warmup sweep and the noise-measurement study.

### Presentation

- Participated in the kick-off presentation.
- Participated in the mid-term presentation.
- Created and participated in the end-of-project presentation.

### Report
- Planned, drafted, co-authored, and revised the final report. 

## Simon:

### Conception & Planning

- Participated in model and dataset research (+ adaption of pipeline).
- Participated in the team meetings including creation of update presentations.

### Core Pipeline Implementation & Experiments

- Adapted model search pipeline to additional model family.
- Implemented and tested linear head averaging.
- Co-implemented the `finetune` and `full_eval` strategies plus the
  SHiFT-style regret metric comparing the frozen-backbone proxy ranking
  against full fine-tuning ground truth.
- Implemented and tested different activation functions.
- Integrated second dataset into pipeline + implementation of log pricing improvement for second dataset.
- Partly executed other experiment runs on cluster.
- Explored concepts (metrics and model families) for summarization pilot.
- Implementation (but unfortunately no testing) of summarization pilot.
 
### Presentation

- Participated in and partly created the kick-off presentation.
- Participated in and partly created the mid-term presentation.
- Participated in and partly created the end-of-project presentation.

### Report
- Co-authored and revised the final report.

## Kiru:

### Pipeline implementation
- implemented deterministic initial seeding across models.
- learning-rate warmup decay scheduling (lr-warmup-decay-schedule branch, not merged after deciding for Z-scoring only -- https://github.com/EnnioEnnio/MLSys/issues/44)

### Conception & Planning
- Adapted model search pipeline to additional model family.
- state-of-the-art summarization research

### Presentation
- Participated in and partly created the kick-off presentation.
- Participated in and partly created the mid-term presentation.
- Participated in the end-of-project presentation.

### Report
- Co-authored and revised the final report.


## Side Note

If a more precise mapping is needed for the report, we can gladly follow up with this.

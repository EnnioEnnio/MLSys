# Gradient clipping — citable literature

Primary sources for the `--grad-clipping` knob (`finetune.FinetuneConfig.grad_clipping`,
wired through `slurm/` as `GRAD_CLIPPING`) and for the fine-tuning instability observed
in the wine_reviews `full_eval` run (albert-base-v2 gradient spike at epoch 5,
`grad_norm_max` ≈ 4.6M vs a healthy mean of ~150–550; see the early-stopping /
grad-norm W&B curves).

## The canonical citation (the method itself)

**Pascanu, Mikolov & Bengio (2013). "On the difficulty of training Recurrent Neural
Networks." ICML 2013, PMLR 28(3):1310–1318.**
[arXiv:1211.5063](https://arxiv.org/abs/1211.5063) ·
[PMLR](https://proceedings.mlr.press/v28/pascanu13.html)

Analyzes the exploding-gradient problem and proposes clipping the **global gradient
norm** as the remedy — exactly what `torch.nn.utils.clip_grad_norm_` implements.
Originally in an RNN context, but this is the standard citation for norm clipping
regardless of architecture.

```bibtex
@inproceedings{pascanu2013difficulty,
  title     = {On the difficulty of training recurrent neural networks},
  author    = {Pascanu, Razvan and Mikolov, Tomas and Bengio, Yoshua},
  booktitle = {Proceedings of the 30th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {28},
  pages     = {1310--1318},
  year      = {2013},
  publisher = {PMLR}
}
```

## Theoretical justification (why it works)

**Zhang, He, Sra & Jadbabaie (2020). "Why gradient clipping accelerates training: A
theoretical justification for adaptivity." ICLR 2020.**
[arXiv:1905.11881](https://arxiv.org/abs/1905.11881) ·
[OpenReview](https://openreview.net/forum?id=BJgnXpVYwS)

Under a relaxed smoothness condition (local smoothness growing with the gradient
norm — empirically true for deep nets), clipped GD provably converges faster than
any fixed-step GD. Cite when arguing clipping is more than a heuristic.

```bibtex
@inproceedings{zhang2020gradient,
  title     = {Why gradient clipping accelerates training: A theoretical justification for adaptivity},
  author    = {Zhang, Jingzhao and He, Tianxing and Sra, Suvrit and Jadbabaie, Ali},
  booktitle = {International Conference on Learning Representations},
  year      = {2020}
}
```

## Instability in transformer fine-tuning (closest to what we observed)

**Mosbach, Andriushchenko & Klakow (2021). "On the Stability of Fine-tuning BERT:
Misconceptions, Explanations, and Strong Baselines." ICLR 2021.**
[arXiv:2006.04884](https://arxiv.org/abs/2006.04884)

Shows BERT-family fine-tuning instability is an **optimization** problem
(vanishing/exploding gradients early in training), not catastrophic forgetting or
small data. Legitimizes the albert/electra spikes as a known phenomenon. Pairs with
the LP-FT warmup citation (Kumar et al. 2022) already used for
`FinetuneConfig.warmup_epochs`.

```bibtex
@inproceedings{mosbach2021stability,
  title     = {On the Stability of Fine-tuning {BERT}: Misconceptions, Explanations, and Strong Baselines},
  author    = {Mosbach, Marius and Andriushchenko, Maksym and Klakow, Dietrich},
  booktitle = {International Conference on Learning Representations},
  year      = {2021}
}
```

## Adaptive / relative clipping (the per-model-threshold question)

**Brock, De, Smith & Simonyan (2021). "High-Performance Large-Scale Image Recognition
Without Normalization." ICML 2021.**
[arXiv:2102.06171](https://arxiv.org/abs/2102.06171)

Introduces **Adaptive Gradient Clipping (AGC)**: clip on the ratio of gradient norm to
parameter norm rather than a fixed absolute threshold. Cite if the report discusses
why models with very different gradient scales could warrant a relative criterion
instead of the single global `GRAD_CLIPPING` value. (Our measure-only run showed a
wide common band — healthy means O(10²), catastrophic spikes O(10⁴–10⁶) — so a single
global threshold ~10³ suffices for the current pool.)

```bibtex
@inproceedings{brock2021high,
  title     = {High-Performance Large-Scale Image Recognition Without Normalization},
  author    = {Brock, Andrew and De, Soham and Smith, Samuel L. and Simonyan, Karen},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {1059--1071},
  year      = {2021},
  publisher = {PMLR}
}
```

## Unverified (double-check before citing)

**Zhang, Wu, Katiyar, Weinberger & Artzi (2021). "Revisiting Few-sample BERT
Fine-tuning." ICLR 2021.** [arXiv:2006.05987](https://arxiv.org/abs/2006.05987)

Fits if the report discusses fine-tuning variance across seeds more broadly. Details
not re-verified — confirm authors/venue before citing.

---

**Suggested usage in the report:** Pascanu et al. for the method, Mosbach et al. for
why it matters in this setting, Zhang et al. (2020) optionally for theory, Brock et
al. only if the threshold-scale discussion makes it in.

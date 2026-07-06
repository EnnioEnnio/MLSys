## Summarization models

**Long documents / "hard" summarization**
- Claude (Opus 4.6/4.8) and Gemini 3.1 Pro strongest
- LongT5 (Google) — T5 variant for long-input attention; plain T5 degrades badly past ~3k words [1]
- PRIMERA (AllenAI) — multi-document summarization

**Everyday short summaries**
- GPT-5.5 and Claude Sonnet 4.6

**Short summaries, "key facts retention" benchmark (OpenMark, March 2026)**
- Minimax M2.5 Lightning 90%, Grok-4 87%, GPT-5-mini 83% [12]
- -> larger models generalise more — not more factual [8]

**Self-hosted**
- Qwen3-30B-A3B-Instruct-2507 — MoE, 30.5B total / 3.3B active params (cheapest to run) [2]
- GLM-4.5V — built on GLM-4.5-Air, 106B total / 12B active params [10]
- GPT-OSS-120B — MoE, 117B total / 5.1B active params (heaviest) [11]
- Note: the general open-weight frontier has since moved to GLM-5.x/DeepSeek V4/Qwen3.5-3.6/Kimi K2.6 — unbenchmarked for summarization so far [3] (aggregator ranking cited here because no primary paper tracks comparative frontier standing across releases; a leaderboard is the closest thing to a source for this kind of claim, though the underlying model facts above come from official model cards)

**Non-LLMs**
- BART (Meta) — encoder-decoder, fine-tuned for summarization; best on XSum/SAMSum [4]
- PEGASUS (Google) — pretrained with sentence-masking objective; best on CNN/DailyMail [4]
- T5 (Google) — summarization is one of many fine-tuned tasks; T5-Base narrows the gap to BART/PEGASUS efficiently, but weak on long docs [1] [4]
- FLAN-T5 (Google) — instruction-tuned T5; beat BART on CNN/DM in one head-to-head (ROUGE/BERTScore/METEOR) [5]
- ProphetNet (Microsoft) — multi-future-token prediction, used for summarization benchmarks
- -> no single winner; best model is dataset-dependent [4] [9]

## Datasets

- CNN/DailyMail — most widely used overall (abisee/cnn_dailymail); more extractive-style summaries
- Newsroom — 1.3M article-summary pairs, 38 outlets (lara-martin/Newsroom)
- XSum — BBC articles, single highly abstractive sentence (EdinburghNLP/xsum)
- SAMSum — dialogue/messenger-style summarization (knkarthick/samsum); **non-commercial licence (CC BY-NC-ND 4.0)** [6] — fine for research/pilot, blocks anything downstream needing permissive/commercial use
- PubMed — long, technical/domain-specific documents; relevant if pairing with LongT5 for the long-doc branch

## Metrics

- No one settled metric
- Most widely used: N-grams (ROUGE, BLEU) — acknowledged inadequate, correlation with human judgment varies a lot by dataset (e.g. Rouge-L: 0.72 relevance correlation on CNN/DM, drops to 0.30 on XSum) [7]
- Dominant: ROUGE, BLEU, BERTScore — correlate poorly with human judgement, especially on abstractive (XSum-style) output [7]
- BLEURT — often the strongest single automatic metric for human-judgment correlation (esp. faithfulness/relevance) [13], though still dataset-dependent (strong on CNN/DM, weak on XSum faithfulness) [7]
- Factual consistency: SummaC, AlignScore, FactScore / MiniCheck / TofuEval / RAGTruth
- LLM-as-a-judge — de facto standard for quality, but not cheap enough to run per-candidate without a separate LLM call

---

### References
[1] Influence of Data Pre-processing and Post-processing on Long Document Summarization — https://arxiv.org/pdf/2112.01660
[2] Qwen3-30B-A3B-Instruct-2507 model card, Hugging Face — https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
[3] Open LLM Leaderboard 2026, LLM Stats — https://llm-stats.com/leaderboards/open-llm-leaderboard
[4] A Comparative Study of PEGASUS, BART, and T5 for Text Summarization Across Diverse Datasets, MDPI — https://www.mdpi.com/1999-5903/17/9/389
[5] Evaluating LLMs and Pre-trained Models for Text Summarization Across Diverse Datasets — https://arxiv.org/html/2502.19339v1
[6] SAMSum dataset card, Hugging Face — https://huggingface.co/datasets/knkarthick/samsum
[7] Benchmarking Large Language Models for News Summarization — https://arxiv.org/pdf/2301.13848
[8] HalluLens: LLM Hallucination Benchmark, Bang et al. (Meta FAIR / HKUST), ACL 2025 — https://arxiv.org/abs/2504.17550
[9] Benchmarking State-of-the-Art Text Summarization Models (BART, T5, PEGASUS, ProphetNet, GPT-3), IEEE — https://ieeexplore.ieee.org/document/11134200/
[10] GLM-4.5V model card / technical report, Hugging Face (arXiv:2507.01006) — https://huggingface.co/zai-org/GLM-4.5V
[11] gpt-oss-120b & gpt-oss-20b Model Card, OpenAI (arXiv:2508.10925) — https://huggingface.co/openai/gpt-oss-120b
[12] Best AI for Summarization 2026: Top 25 Models Benchmarked, OpenMark — https://openmark.ai/best-ai-for-summarization
[13] BLEURT: Learning Robust Metrics for Text Generation, Sellam, Das & Parikh, ACL 2020 — https://arxiv.org/abs/2004.04696

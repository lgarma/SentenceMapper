# SentenceMapper — Theory & Design Rationale

> This document explains the theoretical foundations, design decisions, and key findings behind SentenceMapper. It is intended as onboarding context for LLM agents and collaborators working on this project.

---

## 1. Problem Statement

Large documents (10k–100k+ tokens) are expensive to process with LLMs. Self-attention is quadratic, API costs scale with token count, and long-context models still struggle with information scattered across many pages.

**Goal:** Extract the most information-dense sentences from a document — a small subset that preserves the essential meaning — so an LLM can summarize or reason over a fraction of the original tokens.

---

## 2. Core Intuition

Not all sentences carry equal information. Some sentences are highly representative of their surrounding context (topic sentences, thesis statements, key findings), while others are transitional, repetitive, or low-content.

**Main heuristic:** The best sentences are those that capture the most semantic meaning in the fewest tokens.

We measure this by computing the **cosine similarity** between a sentence's embedding and the embedding of its surrounding context. High similarity means the sentence is a good proxy for the broader text. Combined with the sentence's **length ratio** (how short it is relative to the context), we can identify sentences that pack disproportionate meaning per token.

---

## 3. Original Approach — Chunk-Based

### 3.1 Pipeline

1. **Chunk the document** into fixed-size segments (~2048 characters, no overlap).
2. **Split each chunk into sentences** (merged up to a minimum length threshold).
3. **Embed** all chunks and all sentences using a fast static embedding model (Model2Vec).
4. **Compute cosine similarity** between each sentence and its parent chunk.
5. **Compute length ratio** = `len(sentence) / len(parent_chunk)`.
6. **Fit a power-law frontier** in log-log space to model the baseline relationship between ratio and similarity.
7. **Select sentences** above the frontier — those with higher similarity than expected for their length.

### 3.2 The Power Law

In the original system, a clear power-law relationship emerges:

```
similarity = A × ratio^B
```

In log-log space this is linear:

```
log(similarity) = B × log(ratio) + log(A)
```

We originally fit this to the **lower quantile** (5th percentile) of the data, establishing a floor. However, the lower frontier proved volatile across documents — a few outlier sentences (boilerplate, headers, list items) could drastically shift the lower bound. See Section 5 for the improved upper frontier approach.

### 3.3 Optimization

To select a target percentage of tokens (e.g., 30%), a **bisection algorithm** adjusts the amplitude `A`, shifting the threshold curve until the selected sentences sum to approximately the target token count. The slope `B` remains fixed (it represents the natural scaling relationship).

### 3.4 Limitations Identified

1. **Chunk boundary artifacts:** A sentence at the edge of chunk k may be semantically related to chunk k+1, but is only compared to chunk k. It gets unfairly penalized.

2. **Positional assignment is arbitrary:** Fixed-size chunking creates hard boundaries that don't respect semantic structure. A sentence's score depends on which chunk it lands in.

3. **Self-similarity inflation:** The sentence's own tokens are included in the parent chunk embedding. Longer sentences share more tokens with the chunk → higher similarity "for free." This conflates self-overlap with genuine representativeness.

---

## 4. Improved Approach — Sentence-Centered Context Windows

### 4.1 Key Design Changes

To address all three limitations, the pipeline was restructured:

1. **Sentences first:** Split the full document into sentences upfront (no chunking step).
2. **Build centered context windows:** For each sentence, expand outward by adding complete neighboring sentences (alternating left and right) until a character budget is reached. The target sentence is always roughly centered.
3. **Exclude the sentence from its context:** Compute similarity between the sentence and the surrounding context *without* the sentence itself.

### 4.2 Why Exclude the Sentence

When the sentence is included in the context embedding:

- **Self-overlap artifact:** The sentence's tokens are part of the context. More word overlap → higher cosine similarity, regardless of semantic content.
- **The power law captures two mixed effects:**
  1. Self-overlap (artifact) — longer sentence → more shared tokens → free similarity boost.
  2. Genuine information scaling — longer sentence covers more semantic ground.

When the sentence is excluded:

- The similarity measures pure **representativeness**: how well does this sentence capture what the *rest* of the surrounding text is about?
- There is no free boost from self-overlap.
- The signal is cleaner.

### 4.3 Why Complete Sentences (Not Fixed Character Windows)

A fixed character window centered on the sentence's position would cut sentences in half at the edges, injecting partial-sentence noise into the context embedding. Expanding with complete sentences ensures the context is always semantically clean.

### 4.4 Ratio Definition

```
ratio = len(sentence) / (len(sentence) + len(context))
```

The sentence is included in the denominator because:

- **Normalizes for varying context sizes** — handles document edges and variable window sizes cleanly.
- **Bounded in [0, 1]** — well-behaved for fitting.
- **Comparable across documents** — a ratio of 0.1 means the same thing regardless of context budget.
- Using `len(sentence) / len(context_only)` would be unbounded (→ ∞ as context shrinks).

---

## 5. Key Empirical Finding — The Upper Frontier is Nearly Flat

When examining the ratio-similarity distribution across many documents, two key findings emerged:

1. The **lower frontier** (5th percentile) is highly volatile — it shifts dramatically between documents because a few oddball sentences (boilerplate, headers, formulas) can pull the floor around.
2. The **upper frontier** (95th percentile) is remarkably stable across documents — but its slope is only ~0.1, meaning the ceiling is nearly flat.

### 5.1 Why the Upper Frontier is Stable

The upper frontier represents the **ceiling of representativeness** — the maximum cosine similarity a sentence can achieve for a given length ratio. This ceiling is governed by the fundamental information-theoretic structure of language: how much of a context's semantic content can be packed into a given fraction of its tokens. This relationship is document-invariant.

The lower frontier, by contrast, captures the worst-performing sentences — which are document-specific noise (boilerplate, transitional phrases, list items). Different documents have different types of "junk," making the floor unpredictable.

### 5.2 Nearly Flat Slope → Simplified Selection

Pooled analysis across 50 GovReport documents (~11k sentences) yields an upper frontier slope of ~0.108 ± 0.007 with R² = 0.94. A slope this close to zero means similarity dominates the ceiling — shorter sentences don't gain much extra representativeness.

This motivated replacing the full power-law frontier machinery (fitting + bisection + optimizer) with a simple score-based ranking:

```
score = similarity - α × ratio
```

where α (`length_bias`) is set to 0.5 by default. The additive form applies a direct penalty in similarity-point units and composes naturally with future bias terms (e.g. `+ β × query_similarity` for semantic-biased extraction). Sentences are ranked by score and greedily selected until the token budget is reached.

Additive α = 0.5 is empirically equivalent to the original multiplicative α = 0.1 (Jaccard overlap ≈ 0.95 on the same documents).

A/B comparison across 50 reports at 30% compression confirms the additive formula matches the multiplicative one:

| Metric | α = 0 (similarity only) | Additive α = 0.5 | Multiplicative α = 0.1 |
|--------|--------------------------|-------------------|-------------------------|
| ROUGE-1 | 0.3465 | 0.3475 | 0.3475 |
| ROUGE-2 | 0.1639 | 0.1649 | 0.1652 |
| ROUGE-L | 0.1609 | 0.1638 | 0.1638 |

Win rates (additive α=0.5 vs α=0): ROUGE-1 27–22, ROUGE-2 26–23, ROUGE-L 32–17 — length bias helps, especially on ROUGE-L.

### 5.3 Comparison with Lower Frontier

| Property | Lower Frontier | Upper Frontier |
|----------|---------------|----------------|
| Stability across documents | Low (volatile) | High (stable) |
| What it captures | Worst-case noise | Best-case representativeness |
| Sensitivity to outliers | High (junk sentences shift floor) | Low (ceiling is governed by language structure) |
| Slope consistency | Varies | ~0.1 (nearly flat) |

### 5.4 Implications

- The power-law upper frontier slope is ~0.1, meaning similarity alone drives selection.
- The simple `score = similarity - α × ratio` formula captures the same insight with no frontier fitting.
- The additive form composes naturally with future bias terms: `score = similarity - α·ratio + β·query_similarity`.
- The power-law frontier is retained as an **analysis/visualization tool** but is no longer part of the selection pipeline.
- Sentence-centered context windows eliminate chunk-boundary artifacts and self-similarity inflation.

### 5.5 Open Questions

This should be validated across:

- Multiple documents and domains (legal, scientific, news).
- Different context budgets (512, 1024, 2048, 4096 characters).
- ~~Different embedding models (static vs. transformer-based).~~ → **Resolved.** See Section 8 — embedding model choice has negligible impact on extraction and downstream summary quality.
- Very short vs. very long sentences (does variance change even if mean is flat?).

---

## 6. Architecture Summary

### Current Pipeline (`SentenceMapperPipeline`)

```
text → all sentences → centered context window per sentence (excluding sentence)
     → similarity(sentence, context) → score = similarity - α×ratio → rank & greedy fill
```

The pipeline outputs the following interface:

```python
result = pipeline.process_document(text, objective_percentage=0.3)

result['sentences']          # list[str] — sentence strings
result['similarities']       # np.ndarray — cosine similarities
result['ratios']             # np.ndarray — length ratios
result['tokens']             # np.ndarray — token counts per sentence
result['scores']             # np.ndarray — similarity - α×ratio
result['length_bias']        # float — α used
result['mask']               # np.ndarray — binary selection mask
result['selected_text']      # str — extracted text with (...) separators
result['selected_tokens']    # int — total selected tokens
```

---

## 7. Components

| Module | Purpose |
|--------|---------|
| `SentenceProcessor` | Sentence-centered context windows with exclusion, feature computation |
| `SentenceMapperPipeline` | End-to-end pipeline: scoring + greedy ranking |
| `PowerLawOptimizer` | Power-law frontier analysis & visualization (not used for selection) |
| `SentenceMapperVisualizer` | Matplotlib/Plotly plots + HTML highlighted text export |
| `SentenceSplitter` | Custom regex sentence splitter with domain abbreviations |
| `MapReduceSummarizer` | LLM summarization + LLM-as-Judge evaluation |

---

## 8. Embedding Model

The project uses **Model2Vec** (`minishlab/potion-base-2M`), a static embedding model that computes sentence embeddings as mean-pooled token embeddings. Key properties:

- **No positional encoding** — token order doesn't affect the embedding (bag-of-words).
- **Very fast** — orders of magnitude faster than transformer encoders.
- **No learned positional bias** — but the bag-of-words nature means longer texts are dominated by majority tokens, which creates a content-dilution effect (different from but related to positional bias in transformers).

### 8.1 Model Choice — Does It Matter?

The Model2Vec / Potion family offers four English models of increasing capacity, all distilled from `bge-base-en-v1.5`:

| Model | Parameters | MTEB avg | Embedding dim |
|-------|-----------|----------|---------------|
| `minishlab/potion-base-2M` | 2M | 44.77 | 256 |
| `minishlab/potion-base-4M` | 4M | 48.23 | 256 |
| `minishlab/potion-base-8M` | 8M | 50.03 | 256 |
| `minishlab/potion-base-32M` | 32M | 51.66 | 256 |

Although MTEB scores span nearly 7 points, SentenceMapper uses embeddings for a narrow task: ranking sentences by representativeness within a local context window. The hypothesis was that **relative ranking is preserved across models**, even if absolute similarity values differ — and therefore the cheapest model suffices.

### 8.2 Experiment 1 — Extractive Text Overlap (4 models, 50 docs)

All four models were run through the full extraction pipeline on 50 GovReport documents at 30% compression with α = 0.5. The extracted text was compared against human-written reference summaries using ROUGE.

**ROUGE scores (mean over 50 documents):**

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|----------|
| potion-base-2M | 0.3475 | 0.1635 | 0.1617 |
| potion-base-4M | 0.3490 | 0.1654 | 0.1630 |
| potion-base-8M | 0.3475 | 0.1649 | 0.1638 |
| potion-base-32M | 0.3487 | 0.1663 | 0.1644 |

The spread across all four models is < 0.003 ROUGE points — well within noise.

**Per-document correlation (2M vs 32M):** Pearson r = 0.979 on per-document ROUGE-L scores. The two models agree not just on average but on which documents are easy or hard to extract from.

**Jaccard overlap of selected sentences:**

| Pair | Jaccard overlap |
|------|-----------------|
| 2M vs 4M | 0.82 |
| 4M vs 8M | 0.76 |
| 8M vs 32M | 0.78 |
| 2M vs 32M (extremes) | 0.62 |

Adjacent models share ~76–82% of selected sentences. Even the extreme pair (2M vs 32M) shares 62%, and the remaining 38% of differing sentences produce indistinguishable ROUGE.

### 8.3 Experiment 2 — Spearman Rank Correlation (4 models, 50 docs)

To directly test whether models preserve **sentence ranking**, Spearman rank correlation was computed on the score vectors (similarity − α × ratio) across all sentences in each document, then averaged.

| Pair | Mean Spearman ρ |
|------|------------------|
| 2M vs 4M | 0.964 |
| 4M vs 8M | 0.983 |
| 8M vs 32M | 0.976 |
| 2M vs 8M | 0.953 |
| 4M vs 32M | 0.966 |
| 2M vs 32M | 0.886 |

All pairs show ρ ≥ 0.87. Adjacent models correlate at ρ > 0.95. Even the most distant pair (2M vs 32M) has ρ = 0.886 — strong rank agreement.

### 8.4 Experiment 3 — End-to-End LLM Summary Quality (2M vs 32M, 10 docs)

The strongest test: does the embedding model choice affect the quality of the **final LLM-generated summary**? For 10 GovReport documents:

1. Extract sentences using 2M and 32M independently (30% compression, α = 0.5).
2. Summarize each extraction with `gpt-4.1`.
3. Evaluate each summary against the human reference with an LLM judge (`gpt-5`, scoring rubric 1–10).

**Results:**

| Metric | potion-base-2M | potion-base-32M | Δ (2M − 32M) |
|--------|----------------|-----------------|---------------|
| ROUGE-1 | 0.4766 | 0.4858 | −0.009 |
| ROUGE-2 | 0.1335 | 0.1336 | −0.0001 |
| ROUGE-L | 0.1826 | 0.1878 | −0.005 |
| Judge score | 7.0 | 7.0 | 0.0 |

**Win rates:** ROUGE-L: 2M wins 4, 32M wins 6, ties 0 (coin flip). Judge: **10 ties out of 10** — every document received the same score from both models.

### 8.5 Conclusion — Embedding Model Invariance

The three experiments form a converging evidence stack:

1. **Extractive overlap:** ROUGE spread < 0.003, Jaccard 62–82% — models select largely the same sentences.
2. **Rank preservation:** Spearman ρ ≥ 0.87 for all pairs — models agree on which sentences are best.
3. **End-to-end quality:** LLM judge scores are identical — the abstraction step (summarization) completely washes out whatever small differences the embeddings introduce.

**The embedding model does not meaningfully affect extraction or downstream summary quality.** The LLM's ability to synthesize information from high-density sentences dominates — slight variations in which sentences are selected are absorbed during summarization.

This makes intuitive sense: the scoring formula `similarity − α × ratio` is a relative ranking over cosine similarities within a single document's context windows. All four Potion models are distilled from the same teacher (`bge-base-en-v1.5`) and share the same 256-dimensional embedding space. Their absolute similarity scales differ, but relative ordering is preserved.

**Practical implication:** The project defaults to `potion-base-2M` — the smallest and fastest model — with no quality penalty.

---

## 9. Downstream Use — Map-Reduce Summarization

After extraction, the selected sentences (typically 10–30% of the original tokens) are passed to an LLM for summarization. Modern LLMs can infer missing context from these high-density sentences, producing summaries competitive with those generated from the full document.

Benefits:
- **Token reduction:** 5,000 tokens instead of 50,000.
- **Cost reduction:** Proportional to token count.
- **Latency reduction:** Less input = faster inference.
- **Quality preservation:** Information-dense sentences retain the essential meaning.

---

## 10. Future Directions

### Semantic-Biased Extraction
Allow users to provide query sentences or keywords. Compute similarity between each sentence and the query, and add this as a bias term: `score = similarity - α·ratio + β·query_similarity`. The additive scoring formula was chosen specifically for this composability.

### Multi-Scale Context
Compute similarity at multiple context scales (±2, ±5, ±10 sentences) and combine signals. Sentences that are representative at multiple scales are genuinely informative.

### Benchmarking
- ROUGE and BERTScore on GovReport, Multi-LexSum, and other long-document summarization benchmarks.
- Compare extraction quality at different compression ratios (10%, 20%, 30%).
- Ablation studies on context budget, embedding model, and sentence merging threshold.

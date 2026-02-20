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

## 3. Validation — Does Similarity-Based Extraction Beat Random?

Before building the full pipeline, we validated the core hypothesis: **does similarity-based sentence selection actually outperform random selection?**

### 3.1 Experiment Design

- **Null hypothesis (H₀):** Random sentence selection achieves the same ROUGE score as similarity-based extraction.
- **Alternative hypothesis (H₁):** Similarity-based extraction achieves higher ROUGE scores.
- **Setup:** 100 GovReport documents, **reference-matched compression** (~8.9% mean), α=0.5 length penalty
- **Target tokens:** For each document, `objective_tokens = len(reference_summary)` — extracting the same number of tokens as the human reference for fair ROUGE comparison
- **Random baseline:** For each document, randomly shuffle sentences and greedily select up to the same token budget (10 different random seeds per document, averaged)

### 3.2 Results

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|---------|---------|---------|
| Similarity | 0.5235±0.0102 | **0.1814±0.0081** | **0.2171±0.0050** |
| Random | 0.5188±0.0073 | 0.1558±0.0055 | 0.1996±0.0027 |

**Paired t-test (n=100):**

| Metric | Δ | t-stat | p-value | Cohen's d | Significant? |
|--------|---|--------|---------|-----------|--------------|
| ROUGE-1 | +0.0047 | 0.80 | 0.426 | +0.08 (negligible) | ✗ NO |
| ROUGE-2 | **+0.0256** | 5.40 | **4.55e-07** | +0.54 (medium) | ✓ YES |
| ROUGE-L | **+0.0175** | 4.78 | **6.15e-06** | +0.48 (medium) | ✓ YES |

### 3.3 Interpretation

1. **ROUGE-1 is not significantly different** — with reference-matched extraction (~9% of document), both methods achieve similar unigram overlap since the budget is tighter and less room for noise.

2. **Similarity wins decisively on ROUGE-2 and ROUGE-L** — bigram matches and longest common subsequence reward *coherent, ordered content* over scattered keywords. Similarity-based extraction selects sentences that form meaningful sequences.

3. **Medium effect sizes (d ≈ 0.5)** — the improvement is practically meaningful, not just statistically significant.

4. **Reference-matched evaluation is fairer** — matching the extracted length to reference summary length removes confounding factors from over-extraction.

**Conclusion:** The null hypothesis is rejected for ROUGE-2 and ROUGE-L (p < 0.001). Similarity-based extraction produces statistically significantly better extractions where it matters: content coherence and phrase preservation.

### 3.4 Length Penalty (α) — Does It Help?

Having validated that similarity beats random, we tested whether adding a length penalty improves results further.

**Scoring formula:** `score = similarity - α × ratio`

**Experiment:** 100 GovReport documents, **reference-matched compression** (~9%), α ∈ {0.0, 0.25, 0.5, 1.0}

| α | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---------|---------|---------|
| 0.00 | 0.5202±0.0098 | 0.1760±0.0080 | 0.2124±0.0048 |
| 0.25 | 0.5217±0.0103 | 0.1794±0.0080 | 0.2150±0.0050 |
| 0.50 | 0.5235±0.0102 | 0.1814±0.0081 | **0.2171±0.0050** |
| 1.00 | **0.5270±0.0102** | **0.1825±0.0083** | 0.2159±0.0049 |

**Paired t-test vs α=0:**

| α | Metric | Δ | p-value | Cohen's d | Significant? |
|---|--------|---|---------|-----------|--------------|
| 0.25 | R-2 | +0.0034 | 0.100 | +0.17 | ✗ |
| 0.25 | R-L | +0.0026 | 0.126 | +0.16 | ✗ |
| **0.50** | **R-2** | **+0.0054** | **0.032** | +0.22 | **✓** |
| **0.50** | **R-L** | **+0.0047** | **0.015** | +0.25 | **✓** |
| 1.00 | R-2 | +0.0065 | 0.098 | +0.17 | ✗ |
| 1.00 | R-L | +0.0035 | 0.245 | +0.12 | ✗ |

**Findings:**

1. **α=0.5 is the sweet spot** — it's the only value achieving statistical significance on both ROUGE-2 (p=0.032) and ROUGE-L (p=0.015).

2. **Effect sizes are small** (Cohen's d ≈ 0.2-0.25) — the improvement is real but modest.

3. **α=1.0 overshoots** — too aggressive length penalty loses the significance gains.

4. **ROUGE-1 remains non-significant** — length penalty doesn't help unigram coverage.

**Interpretation:** The length penalty helps select sentences that are both representative *and* concise. At α=0.5, the penalty is strong enough to prefer shorter sentences at equal similarity, but not so strong that it overrides semantic relevance. This confirms **α=0.5 as the recommended default**.

---

## 4. Original Approach — Chunk-Based

### 4.1 Pipeline

1. **Chunk the document** into fixed-size segments (~2048 characters, no overlap).
2. **Split each chunk into sentences** (merged up to a minimum length threshold).
3. **Embed** all chunks and all sentences using a fast static embedding model (Model2Vec).
4. **Compute cosine similarity** between each sentence and its parent chunk.
5. **Compute length ratio** = `len(sentence) / len(parent_chunk)`.
6. **Fit a power-law frontier** in log-log space to model the baseline relationship between ratio and similarity.
7. **Select sentences** above the frontier — those with higher similarity than expected for their length.

### 4.2 The Power Law

In the original system, a clear power-law relationship emerges:

```
similarity = A × ratio^B
```

In log-log space this is linear:

```
log(similarity) = B × log(ratio) + log(A)
```

We originally fit this to the **lower quantile** (5th percentile) of the data, establishing a floor. However, the lower frontier proved volatile across documents — a few outlier sentences (boilerplate, headers, list items) could drastically shift the lower bound. See Section 5 for the improved upper frontier approach.

### 4.3 Optimization

To select a target percentage of tokens (e.g., 30%), a **bisection algorithm** adjusts the amplitude `A`, shifting the threshold curve until the selected sentences sum to approximately the target token count. The slope `B` remains fixed (it represents the natural scaling relationship).

### 4.4 Limitations Identified

1. **Chunk boundary artifacts:** A sentence at the edge of chunk k may be semantically related to chunk k+1, but is only compared to chunk k. It gets unfairly penalized.

2. **Positional assignment is arbitrary:** Fixed-size chunking creates hard boundaries that don't respect semantic structure. A sentence's score depends on which chunk it lands in.

3. **Self-similarity inflation:** The sentence's own tokens are included in the parent chunk embedding. Longer sentences share more tokens with the chunk → higher similarity "for free." This conflates self-overlap with genuine representativeness.

---

## 5. Improved Approach — Sentence-Centered Context Windows

### 5.1 Key Design Changes

To address all three limitations, the pipeline was restructured:

1. **Sentences first:** Split the full document into sentences upfront (no chunking step).
2. **Build centered context windows:** For each sentence, expand outward by adding complete neighboring sentences (alternating left and right) until a character budget is reached. The target sentence is always roughly centered.
3. **Exclude the sentence from its context:** Compute similarity between the sentence and the surrounding context *without* the sentence itself.

### 5.2 Why Exclude the Sentence

When the sentence is included in the context embedding:

- **Self-overlap artifact:** The sentence's tokens are part of the context. More word overlap → higher cosine similarity, regardless of semantic content.
- **The power law captures two mixed effects:**
  1. Self-overlap (artifact) — longer sentence → more shared tokens → free similarity boost.
  2. Genuine information scaling — longer sentence covers more semantic ground.

When the sentence is excluded:

- The similarity measures pure **representativeness**: how well does this sentence capture what the *rest* of the surrounding text is about?
- There is no free boost from self-overlap.
- The signal is cleaner.

### 5.3 Why Complete Sentences (Not Fixed Character Windows)

A fixed character window centered on the sentence's position would cut sentences in half at the edges, injecting partial-sentence noise into the context embedding. Expanding with complete sentences ensures the context is always semantically clean.

### 5.4 Ratio Definition

```
ratio = len(sentence) / (len(sentence) + len(context))
```

The sentence is included in the denominator because:

- **Normalizes for varying context sizes** — handles document edges and variable window sizes cleanly.
- **Bounded in [0, 1]** — well-behaved for fitting.
- **Comparable across documents** — a ratio of 0.1 means the same thing regardless of context budget.
- Using `len(sentence) / len(context_only)` would be unbounded (→ ∞ as context shrinks).

---

## 6. Key Empirical Finding — The Upper Frontier is Nearly Flat

When examining the ratio-similarity distribution across many documents, two key findings emerged:

1. The **lower frontier** (5th percentile) is highly volatile — it shifts dramatically between documents because a few oddball sentences (boilerplate, headers, formulas) can pull the floor around.
2. The **upper frontier** (95th percentile) is remarkably stable across documents — but its slope is only ~0.1, meaning the ceiling is nearly flat.

### 6.1 Why the Upper Frontier is Stable

The upper frontier represents the **ceiling of representativeness** — the maximum cosine similarity a sentence can achieve for a given length ratio. This ceiling is governed by the fundamental information-theoretic structure of language: how much of a context's semantic content can be packed into a given fraction of its tokens. This relationship is document-invariant.

The lower frontier, by contrast, captures the worst-performing sentences — which are document-specific noise (boilerplate, transitional phrases, list items). Different documents have different types of "junk," making the floor unpredictable.

### 6.2 Nearly Flat Slope → Simplified Selection

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

### 6.3 Comparison with Lower Frontier

| Property | Lower Frontier | Upper Frontier |
|----------|---------------|----------------|
| Stability across documents | Low (volatile) | High (stable) |
| What it captures | Worst-case noise | Best-case representativeness |
| Sensitivity to outliers | High (junk sentences shift floor) | Low (ceiling is governed by language structure) |
| Slope consistency | Varies | ~0.1 (nearly flat) |

### 6.4 Implications

- The power-law upper frontier slope is ~0.1, meaning similarity alone drives selection.
- The simple `score = similarity - α × ratio` formula captures the same insight with no frontier fitting.
- The additive form composes naturally with future bias terms: `score = similarity - α·ratio + β·query_similarity`.
- The power-law frontier is retained as an **analysis/visualization tool** but is no longer part of the selection pipeline.
- Sentence-centered context windows eliminate chunk-boundary artifacts and self-similarity inflation.

### 6.5 Open Questions

This should be validated across:

- Multiple documents and domains (legal, scientific, news).
- Different context budgets (512, 1024, 2048, 4096 characters).
- ~~Different embedding models (static vs. transformer-based).~~ → **Resolved.** See Section 8 — embedding model choice has negligible impact on extraction and downstream summary quality.
- Very short vs. very long sentences (does variance change even if mean is flat?).

---

## 7. Architecture Summary

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

## 8. Components

| Module | Purpose |
|--------|---------|
| `SentenceProcessor` | Sentence-centered context windows with exclusion, feature computation |
| `SentenceMapperPipeline` | End-to-end pipeline: scoring + greedy ranking |
| `PowerLawOptimizer` | Power-law frontier analysis & visualization (not used for selection) |
| `SentenceMapperVisualizer` | Matplotlib/Plotly plots + HTML highlighted text export |
| `SentenceSplitter` | Custom regex sentence splitter with domain abbreviations |
| `MapReduceSummarizer` | LLM summarization + LLM-as-Judge evaluation |

---

## 9. Embedding Model

The project uses **Model2Vec** (`minishlab/potion-base-2M`), a static embedding model that computes sentence embeddings as mean-pooled token embeddings. Key properties:

- **No positional encoding** — token order doesn't affect the embedding (bag-of-words).
- **Very fast** — orders of magnitude faster than transformer encoders.
- **No learned positional bias** — but the bag-of-words nature means longer texts are dominated by majority tokens, which creates a content-dilution effect (different from but related to positional bias in transformers).

### 9.1 Model Choice — Does It Matter?

The Model2Vec / Potion family offers four English models of increasing capacity, all distilled from `bge-base-en-v1.5`:

| Model | Parameters | MTEB avg | Embedding dim |
|-------|-----------|----------|---------------|
| `minishlab/potion-base-2M` | 2M | 44.77 | 256 |
| `minishlab/potion-base-4M` | 4M | 48.23 | 256 |
| `minishlab/potion-base-8M` | 8M | 50.03 | 256 |
| `minishlab/potion-base-32M` | 32M | 51.66 | 256 |

Although MTEB scores span nearly 7 points, SentenceMapper uses embeddings for a narrow task: ranking sentences by representativeness within a local context window. The hypothesis was that **relative ranking is preserved across models**, even if absolute similarity values differ — and therefore the cheapest model suffices.

### 9.2 Experiment 1 — Extractive Text Overlap (4 models, 50 docs)

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

### 9.3 Experiment 2 — Spearman Rank Correlation (4 models, 50 docs)

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

### 9.4 Experiment 3 — End-to-End LLM Summary Quality (2M vs 32M, 10 docs)

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

### 9.5 Conclusion — Embedding Model Invariance

The three experiments form a converging evidence stack:

1. **Extractive overlap:** ROUGE spread < 0.003, Jaccard 62–82% — models select largely the same sentences.
2. **Rank preservation:** Spearman ρ ≥ 0.87 for all pairs — models agree on which sentences are best.
3. **End-to-end quality:** LLM judge scores are identical — the abstraction step (summarization) completely washes out whatever small differences the embeddings introduce.

**The embedding model does not meaningfully affect extraction or downstream summary quality.** The LLM's ability to synthesize information from high-density sentences dominates — slight variations in which sentences are selected are absorbed during summarization.

This makes intuitive sense: the scoring formula `similarity − α × ratio` is a relative ranking over cosine similarities within a single document's context windows. All four Potion models are distilled from the same teacher (`bge-base-en-v1.5`) and share the same 256-dimensional embedding space. Their absolute similarity scales differ, but relative ordering is preserved.

**Practical implication:** The project defaults to `potion-base-2M` — the smallest and fastest model — with no quality penalty.

---

## 10. Experiment — Global Context Similarity

Since Model2Vec (static embeddings) has **no context window limit**, we investigated whether adding a "global" similarity signal — comparing each sentence to the full document embedding — could improve extraction quality.

### 10.1 Hypothesis

Local context (±2048 characters) captures neighborhood representativeness. Global context (full document) might capture sentences representative of the entire document's theme — thesis statements, executive summaries, key conclusions that echo throughout.

**Proposed scoring formula:**
```
score = similarity_local - α × ratio + γ × similarity_global
```

### 10.2 Experiment Design

- 100 GovReport documents, **reference-matched tokens** (~9% compression), α = 0.5
- For each sentence: compute cosine similarity to (a) local context window and (b) full document embedding
- Measure correlation between local and global signals
- Evaluate ROUGE at γ values {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}

### 10.3 Results

**Correlation analysis (n=100):**

| Metric | Value |
|--------|-------|
| Mean correlation (local vs global) | 0.620 ± 0.116 |

The moderate correlation (0.62) confirms that local and global similarity identify **partially overlapping but distinct** sentences.

**ROUGE scores by γ:**

| γ (global) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------|---------|---------|---------|
| 0.00 | 0.5235±0.0178 | 0.1814±0.0105 | 0.2171±0.0088 |
| 0.10 | 0.5276±0.0177 | 0.1852±0.0106 | 0.2196±0.0089 |
| 0.25 | 0.5280±0.0175 | 0.1846±0.0101 | 0.2222±0.0087 |
| 0.50 | 0.5252±0.0173 | 0.1805±0.0098 | 0.2229±0.0086 |
| 0.75 | 0.5193±0.0170 | 0.1753±0.0093 | 0.2197±0.0082 |
| 1.00 | 0.5120±0.0168 | 0.1687±0.0090 | 0.2144±0.0080 |

**Statistical significance (paired t-test vs γ=0):**

| γ | Metric | Δ | t-stat | p-value | Cohen's d | Sig |
|------|--------|--------|-------|---------|-----------|-----|
| 0.10 | R-2 | +0.0038 | 2.01 | 0.047 | 0.20 | ✓ |
| 0.25 | R-L | +0.0051 | 2.44 | 0.016 | 0.24 | ✓ |
| 0.50 | R-L | +0.0058 | 2.05 | 0.043 | 0.21 | ✓ |
| 0.75 | R-2 | −0.0061 | −2.15 | 0.034 | −0.22 | ✗ |
| 1.00 | R-2 | −0.0127 | −4.00 | <0.001 | −0.40 | ✗ |

### 10.4 Conclusion — Global Context Worth Keeping

The global context signal shows a **complex pattern**:

1. **Low γ (0.1–0.25) provides small but significant gains** — statistically significant on R-2 or R-L
2. **High γ (≥0.75) significantly hurts performance** — ROUGE-2 drops significantly
3. **Peak appears around γ=0.25** — best balance of improvement without overweighting global signal

**Why does high γ hurt?** For bag-of-words embeddings like Model2Vec, long documents produce diluted embeddings dominated by frequent terms. High γ values cause global similarity to dominate the scoring, rewarding vocabulary overlap rather than information density.

**Practical implication:** The improvement magnitude (Δ ≈ 0.004–0.005, d ≈ 0.2) is comparable to the length penalty benefit. Since consistency matters and the computational overhead is minimal (one extra embedding), **global context is included with γ=0.1 as the default**. Users who prefer pure local context can set γ=0.

---

## 11. Downstream Use — Map-Reduce Summarization

After extraction, the selected sentences (typically 10–30% of the original tokens) are passed to an LLM for summarization. Modern LLMs can infer missing context from these high-density sentences, producing summaries competitive with those generated from the full document.

Benefits:
- **Token reduction:** 5,000 tokens instead of 50,000.
- **Cost reduction:** Proportional to token count.
- **Latency reduction:** Less input = faster inference.
- **Quality preservation:** Information-dense sentences retain the essential meaning.

---

## 12. Future Directions

### Semantic-Biased Extraction
Allow users to provide query sentences or keywords. Compute similarity between each sentence and the query, and add this as a bias term: `score = similarity - α·ratio + β·query_similarity`. The additive scoring formula was chosen specifically for this composability.

### Multi-Scale Context
Compute similarity at multiple context scales (±2, ±5, ±10 sentences) and combine signals. Sentences that are representative at multiple scales are genuinely informative.

### Benchmarking
- ROUGE and BERTScore on GovReport, Multi-LexSum, and other long-document summarization benchmarks.
- Compare extraction quality at different compression ratios (10%, 20%, 30%).
- Ablation studies on context budget, embedding model, and sentence merging threshold.

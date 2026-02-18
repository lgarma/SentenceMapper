# Sentence Mapper

**Extractive summarization for large documents based on similarity to local context.**

## Identifying information-dense sentences

The idea is simple: for each sentence in a document, we build a **centered context window** of surrounding sentences (excluding the target sentence itself) and evaluate two dimensions:

- How well does the sentence represent its surrounding context.
- How short is the sentence relative to the context window.

**Main Heuristic:** The best sentences are those that can capture most semantic meaning.

By excluding the sentence from its own context, we measure pure representativeness without self-overlap artifacts.

Each sentence is scored by:

```
score = similarity - α × ratio
```

where *similarity* is the cosine similarity between the sentence and its context, *ratio* is `len(sentence) / (len(sentence) + len(context))`, and *α* (`length_bias`, default 0.5) is a linear penalty that mildly favours shorter sentences at equal similarity. Setting α = 0 gives pure similarity ranking.

The additive form composes naturally with future bias terms (e.g. `+ β × query_similarity` for semantic-biased extraction).

Sentences are ranked by score and greedily selected from the top until the target token budget is reached.

SentenceMapper is inspired by PatternRank (Schopf et al. 2022)[https://arxiv.org/pdf/2210.05245], a technique used to extract keyphrases using embeddings + Part of Speech patterns. Keyphrases are ranked based on their similarity to the input text.


## Use in Map - Reduce Summarization


LLMs have difficulties processing large documents. Self-Attention is quadratic, and ...

One way to summarize large documents is map-reduce, which consist of spliting large documents into smaller, more manageable chunks that fit LLM context windows. Generate a chunk-level summaries (mapping phase) and aggregate them togheter for a final summary (reduce phase).

This approach is computationally expensive. 

Using SentenceMapper for the mapping phase, is much cheaper and fast. Embeddings computation can be done blazingling fast with Model2Vec models.

LLMs can typically infer missing context from these high-density sentences, significantly reducing:
- Token consumption (e.g., 5,000 tokens instead of 50,000)
- Processing latency
- API costs


## Usage

```python
from src.pipeline import SentenceMapperPipeline

# Initialize with embedding model
pipeline = SentenceMapperPipeline(
    embedding_model_name="minishlab/potion-base-8M",
    context_budget=2048,
    min_sentence_length=256,
    length_bias=0.5,  # α: 0 = pure similarity, higher = penalise longer sentences
)

# Process document without filtering
result = pipeline.process_document(text)

# Or select to a target percentage (e.g., 30% of original tokens)
result = pipeline.process_document(text, objective_percentage=0.3)

print(f"Selected {result['selected_tokens']} tokens from {sum(result['tokens'])} total")
print(result['selected_text'])
```

## Visualization

The package includes visualization tools to understand the selection process:

```python
from src.visualization import SentenceMapperVisualizer

visualizer = SentenceMapperVisualizer()

# Visualize similarity vs. ratio with power-law frontier and selected sentences
visualizer.plot_similarity_vs_ratio(
    result['similarities'],
    result['ratios'],
    mask=result['mask'],
    params=result['params']
)
```

## How It Works

### Sentence-Centered Context Windows

1. **Split the document** into sentences (with configurable minimum length for merging short sentences).
2. **Build context windows**: For each sentence, expand outward with complete neighboring sentences (alternating left/right) until a character budget is reached. The target sentence is excluded from its own context.
3. **Compute similarity**: Cosine similarity between each sentence's embedding and its context embedding.
4. **Compute ratio**: `len(sentence) / (len(sentence) + len(context))` — bounded in [0, 1].

### Scoring & Selection

Each sentence receives a score:

```
score = similarity - α × ratio
```

where α (`length_bias`, default 0.5) mildly favours shorter sentences.

Sentences are ranked by descending score and greedily added until the token budget is reached.

### Why Not a Power-Law Frontier?

Empirical analysis on GovReport shows the upper frontier (95th percentile) of the ratio-similarity distribution has a slope of ~0.1 — nearly flat. This means similarity alone almost entirely determines selection quality (ROUGE-1: 0.3465 for α=0 vs 0.3475 for additive α=0.5 across 50 reports). The simple additive score formula retains the length-bias insight from the power-law analysis without the complexity of frontier fitting + bisection search, and composes naturally with future bias terms. The power-law frontier is still available as an analysis/visualization tool.


## Preliminary Results

<!-- TODO: Add Rouge score of mapping X amount of tokens. -->

## Future Work

### Benchmarking & Evaluation
- Evaluate extractive summarization quality using ROUGE and BERTScore on long summarization datasets (GovReport, Multi-LexSum)
- Benchmark the full map-reduce pipeline against baseline approaches
- Compare token reduction vs. information retention across different percentage targets

### Semantic-Biased Extraction
- Allow users to provide query sentences or keywords that represent their specific interests.
- Compute semantic similarity between each sentence and the user's query
- Add this similarity as a bias term: `score = similarity - α·ratio + β·query_similarity` (the additive scoring formula was chosen for this composability).


## Installation

```bash
uv sync
```

## Requirements

- Python ≥ 3.12
- model2vec (fast embeddings)
- scikit-learn (cosine similarity)
- tiktoken (token counting)
- matplotlib (visualization)
- plotly (visualization)

See `example.ipynb` for a complete demonstration on the GovReport dataset.

# Sentence Mapper

**Extractive summarization for large documents based on similarity to local context.**

## Identifying information-dense sentences

Not all sentences carry equal information. Some sentences are highly representative of their surrounding context (topic sentences, thesis statements, key findings), while others are transitional, repetitive, or low-content.

**Main heuristic:** The best sentences are those that capture the most semantic meaning in the fewest tokens.

We measure this by computing the **cosine similarity** between a sentence's embedding and the embedding of its surrounding context. High similarity means the sentence is a good proxy for the broader text. Combined with the sentence's **length ratio** (how short it is relative to the context), we can identify sentences that pack disproportionate meaning per token.

Each sentence is scored by:

```
score = similarity - α × ratio
```

where *similarity* is the cosine similarity between the sentence and its context, *ratio* is `len(sentence) / (len(sentence) + len(context))`, and *α* (`length_bias`, default 0.5) is a linear penalty that mildly favours shorter sentences at equal similarity. Setting α = 0 gives pure similarity ranking.

Sentences are ranked by score and greedily selected from the top until the target token budget is reached.

SentenceMapper is inspired by PatternRank (Schopf et al. 2022)[https://arxiv.org/pdf/2210.05245], a technique used to extract keyphrases using embeddings + Part of Speech patterns. Keyphrases are ranked based on their similarity to the text.


## Use Cases

### Map - Reduce Summarization

With the rise of context engineering, summarization has become more and more important. Context and attention are limited resources, and each additional token can degrade quality of the output.

The classic way to summarize large documents is map-reduce, which consist of spliting large documents into smaller, more manageable chunks that fit LLM context windows. An LLM is tasked to generate a chunk-level summaries in parallel (mapping phase) and aggregate them togheter for a final summary (reduce phase).

This approach is computationally expensive, and wasteful. Large documents like reports, clincial trials, legal cases can span 100k tokens, while summaries are somewhere in between 500-5000 tokens.  

Using SentenceMapper for the mapping phase, is much cheaper and fast. Embeddings computation can be done blazingling fast with Model2Vec models.

LLMs can typically infer missing context from these representative sentences, significantly reducing:
- Token consumption (compressing the document 70-80%)
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
# Sentence Mapper

**Extractive summarization for large documetns based on information-density.**

## Identifing information-dense sentences

The idea is simple, we can evaluate sentences in two dimensions:

- How well does the sentence represent the surrounding context.
- How short is the sentence relative to its parent chunk.

**Main Heuristic:** The best sentences are those that can capture most semantic meaning in the shortest amount of tokens. 

The similarity between a sentence and its parent chunk naturally increases as there is more word overlap. Longer sentences (higher ratio) tend to have higher similarity simply because they share more content with the chunk. We can model this relation as a power law.

```
similarity = A × ratio^B
```

We can fit this relationship by binning the data and selecting the **lower quantile** (5th percentile by default) representing the minimum expected similarity. Sentences **above** this frontier have higher similarity than the expected for their length, indicating they pack more meaning per token.

Using the residual between the expected value and the actual value, is easy to solve precisely which are the most informative sentences. We can set an objective number of tokens, or an objective percentage of the document.

<!-- TODO: Add plot showing similarity vs ratio in log-log space with fitted power law -->

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
    chunk_size=2048,
    min_sentence_lenght=256
)

# Process document without filtering
result = pipeline.process_document(text)

# Or optimize to a target percentage (e.g., 20% of original tokens)
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
    result['all_similarities'],
    result['ratios'],
    mask=result['mask'],
    params=result['params']
)
```

## How It Works

The optimizer fits a **power-law frontier** in log-log space using linear regression:

```
log(similarity) = B × log(ratio) + log(A)
```

Where:
- **B (slope)**: The natural scaling exponent between ratio and similarity
- **A (amplitude)**: Controls the threshold position—increasing A selects fewer sentences

### Optimization Process

1. **Fit the frontier**: Identify the lower bound (5th percentile by default) of the ratio-similarity distribution to establish the minimum expected relationship
2. **Adjust amplitude**: A bisection algorithm finds the optimal amplitude `A` that yields the target percentage of tokens by raising the threshold curve above the baseline
3. **Select sentences**: All sentences above the adjusted threshold curve are selected

<!-- TODO: Add plot showing how the frontier shifts during optimization -->


## Preliminary Results

<!-- TODO: Add Rouge score of mapping X amount of tokens. -->

## Future Work

### Robust Sentence Selection
- The current sentence selection is sensitive to the chunking strategy. One sentence could have low similarity to their parent chunk, but high similarity to a neighbour chunk.
- Using different chunk sizes or weighting similarity with neighbors could make the process more robust.

### Benchmarking & Evaluation
- Evaluate extractive summarization quality using ROUGE and BERTScore on long summarization datasets (GovReport, Multi-LexSum)
- Benchmark the full map-reduce pipeline against baseline approaches
- Compare token reduction vs. information retention across different percentage targets

### Semantic-Biased Extraction
- Allow users to provide query sentences or keywords that represent their specific interests.
- Compute semantic similarity between each sentence and the user's query
- Add this similarity as a bias term that shifts sentences upward in the ratio-similarity space, increasing the likelihood of being selected.


## Installation

```bash
uv sync
```

## Requirements

- Python ≥ 3.12
- model2vec (fast embeddings)
- chonkie (sentence chunking)
- scikit-learn (cosine similarity)
- scipy (optimization)
- tiktoken (token counting)
- matplotlib (visualization)
- plotly (visualization)

See `example.ipynb` for a complete demonstration on the GovReport dataset.

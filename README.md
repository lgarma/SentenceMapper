# Sentence Mapper

**Extract high-density informative sentences from large documents using embedding-based optimization.**

## Overview

Sentence Mapper provides an intelligent approach to extractive summarization for large documents. Instead of processing entire documents, it identifies and extracts the most information-dense sentences by analyzing the relationship between sentence embeddings and their surrounding context.

## The Problem

Most NLP benchmarks focus on short texts (a few hundred words), but efficient summarization of large documents remains challenging. Traditional map-reduce approaches split documents into chunks that fit LLM context windows, but processing all chunks is token-intensive and slow.

## Sentence Mapper

Sentence Mapper proposes an **embedding-based extractive summarization** technique for the mapping step:

1. **Document Chunking**: Splits documents into chunks
2. **Embedding Analysis**: Computes embeddings for both chunks and individual sentences
3. **Similarity & Ratio Scoring**: Calculates cosine similarity between sentences and their parent chunks, plus sentence-to-chunk length ratios
4. **Power-Law Filtering**: Uses a power-law frontier to identify and select information-dense sentences

### The Power-Law Frontier

The similarity between a sentence and its parent chunk naturally increases as there is more word overlap—longer sentences (higher ratio) tend to have higher similarity simply because they share more content with the chunk.

When plotting similarity vs. ratio in **log-log space**, this relationship follows a clear **power law**:

```
ratio = A × similarity^B
```

<!-- TODO: Add plot showing similarity vs ratio in log-log space with fitted power law -->

We fit this frontier using either:
- **Quantile method (95th percentile)**: Captures the upper envelope of the data
- **Binned max method**: Fits to maximum values within binned similarity ranges

### Information-Dense Sentences

Sentences that fall **below** this power-law frontier are more information-dense than expected for their length. These sentences achieve high semantic similarity to their parent chunk while using fewer words—they pack more meaning per token.

<!-- TODO: Add plot highlighting sentences below the frontier as selected -->

The algorithm targets these below-frontier sentences to extract the most valuable content.

## Usage

```python
from src.pipeline import SentenceMapperPipeline

# Initialize with embedding model
pipeline = SentenceMapperPipeline(
    embedding_model_name="minishlab/potion-base-8M",
    chunk_size=2000,
    chunk_overlap=0
)

# Process document without filtering
result = pipeline.process_document(text)

# Or optimize to a target percentage (e.g., 20% of original tokens)
result = pipeline.process_document(text, objective_percentage=0.2)

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

<!-- TODO: Add example visualization output -->

## Applications

### Map-Reduce Summarization

Use extracted sentences as input for map-reduce summarization pipelines. LLMs can typically infer missing context from these high-density sentences, significantly reducing:
- Token consumption (e.g., 5,000 tokens instead of 50,000)
- Processing latency
- API costs

### Overview-Guided Summarization

For complex documents additional context can be used to guide the summarization:
- Combine extracted sentences with a document overview.
  - Document metadata, table of contents, human annotations, or include the initial pages without modification.

## How It Works

The optimizer fits a **power-law frontier** in log-log space using linear regression:

```
log(ratio) = B × log(similarity) + log(A)
```

Where:
- **B (slope)**: The natural scaling exponent between similarity and ratio
- **A (amplitude)**: Controls the frontier position—lowering A selects fewer sentences

### Optimization Process

1. **Fit the frontier**: Identify the upper envelope of the similarity-ratio distribution using quantile regression (95th percentile) or binned maximum values
2. **Adjust amplitude**: A bisection algorithm finds the optimal amplitude `A` that yields the target percentage of tokens
3. **Select sentences**: All sentences below the adjusted power-law curve are selected

<!-- TODO: Add plot showing how the frontier shifts during optimization -->

This approach provides a principled way to identify information-dense sentences: those that achieve unexpectedly high similarity given their length.

## Future Work

### Enhanced Sentence Segmentation
- Implement a custom sentence chunker with configurable abbreviation handling
- Support domain-specific tokenization rules (e.g., legal citations, scientific notation)
- Add flexibility for users to define custom sentence boundary patterns

### Benchmarking & Evaluation
- Evaluate extractive summarization quality using ROUGE, BERTScore, and other metrics
- Benchmark the full map-reduce pipeline against baseline approaches
- Compare token reduction vs. information retention across different percentage targets
- Measure performance on standard datasets (GovReport, PubMed, arXiv papers)

### Semantic-Biased Extraction
- Allow users to provide query sentences or keywords that represent their specific interests
- Compute semantic similarity between each sentence and the user's query
- Add this similarity as a bias term that shifts sentences rightward in the similarity-ratio space
- Enable query-focused summarization where relevant sentences are prioritized for selection
- Support multi-query scenarios with weighted combinations of semantic biases

### Open Research Questions
- **Universality of the power-law**: Does the similarity-ratio relationship hold across all document types, or is it domain-dependent?
- **Cross-domain generalization**: How does the method perform on technical documentation, code comments, legal texts, or conversational data?
- **Code summarization**: Can sentence mapping be adapted to extract high-density lines or blocks from source code?
- **Optimal frontier fitting**: Which method (quantile vs. binned-max) works best for different document characteristics?

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python ≥ 3.12
- model2vec (fast embeddings)
- chonkie (sentence chunking)
- scikit-learn (cosine similarity)
- scipy (optimization)
- tiktoken (token counting)
- matplotlib (visualization)

See `example.ipynb` for a complete demonstration on the GovReport dataset.

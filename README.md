# Sentence Mapper

**Extract high-density informative sentences from large documents using embedding-based optimization.**

## Overview

Sentence Mapper provides an intelligent approach to extractive summarization for large documents. Instead of processing entire documents, it identifies and extracts the most information-dense sentences by analyzing the relationship between sentence embeddings and their surrounding context.

## The Problem

Most NLP benchmarks focus on short texts (a few hundred words), but efficient summarization of large documents remains challenging. Traditional map-reduce approaches split documents into chunks that fit LLM context windows, but processing all chunks is token-intensive and slow.

## The Solution

Sentence Mapper uses a novel **embedding-based extractive summarization** technique for the mapping step:

1. **Document Chunking**: Splits documents into overlapping chunks
2. **Embedding Analysis**: Computes embeddings for both chunks and individual sentences
3. **Similarity & Ratio Scoring**: Calculates cosine similarity between sentences and their parent chunks, plus sentence-to-chunk length ratios
4. **Intelligent Filtering**: Uses an arctan-based optimizer to select sentences with high information density

### The Three Regions

When plotting sentence-chunk similarity vs. sentence-chunk length ratio, three distinct regions emerge:

- **Low similarity, low ratio**: Connecting/transitional sentences with minimal information
- **High similarity, high ratio**: Tables, lists, or repetitive sections (high similarity due to word overlap)
- **High similarity, low ratio**: ⭐ **Information-dense sentences** — concise statements that capture the chunk's meaning

The algorithm targets this third region to extract the most valuable content.

## Usage

```python
from src.pipeline import SentenceMapper

# Initialize with embedding model
pipeline = SentenceMapper(
    embedding_model_name="minishlab/potion-base-8M",
    chunk_size=2000,
    chunk_overlap=0
)

# Process document without filtering
result = pipeline.process_document(text)

# Or optimize to a target token count
result = pipeline.process_document(text, objective_tokens=5000)

print(f"Selected {result['selected_tokens']} tokens from {result['total_tokens']}")
print(result['selected_text'])
```

## Visualization

The package includes visualization tools to understand the selection process:

```python
from src.visualization import SentenceMapperVisualizer

visualizer = SentenceMapperVisualizer()

# Visualize similarity vs. ratio with selected sentences highlighted
visualizer.plot_similarity_vs_ratio(
    result['all_similarities'],
    result['ratios'],
    mask=result['mask'],
    x_opt=result['x_opt']
)
```

## Applications

### Map-Reduce Summarization

Use extracted sentences as input for map-reduce summarization pipelines. LLMs can typically infer missing context from these high-density sentences, significantly reducing:
- Token consumption (e.g., 5,000 tokens instead of 50,000)
- Processing latency
- API costs

### Overview-Guided Summarization

For complex documents where context is critical:
- Combine extracted sentences with document overviews (table of contents, human annotations, or LLM-generated summaries of initial pages)
- Provides LLMs with both structural context and key details

## How It Works

The optimizer uses an **arctan function** with adaptive parameters to create a decision boundary in similarity-ratio space:

```
y = α × arctan((x - β) / γ)
```

- **α (amplitude)**: Decreases as optimization progresses
- **β (horizontal shift)**: Increases to be more selective
- **γ (steepness)**: Controls curve smoothness

A bisection algorithm finds the optimal parameter `x` that yields the target token count, automatically balancing informativeness and conciseness.

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python ≥ 3.12
- model2vec (fast embeddings)
- chonkie (sentence chunking)
- scikit-learn (cosine similarity)
- tiktoken (token counting)
- matplotlib (visualization)

See `example.ipynb` for a complete demonstration on the GovReport dataset. 


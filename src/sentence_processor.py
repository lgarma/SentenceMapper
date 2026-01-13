"""Sentence processing and analysis module."""

from typing import Any

import numpy as np
import tiktoken
from chonkie import SentenceChunker
from sklearn.metrics.pairwise import cosine_similarity


class SentenceProcessor:
    """Process and analyze sentences from text chunks."""

    def __init__(
        self,
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        encoding_name: str = "cl100k_base",
    ):
        """Initialize the sentence processor.

        Args:
            chunk_size: Size of text chunks in tokens (default: 2048)
            chunk_overlap: Overlap between chunks in tokens (default: 0)
            encoding_name: Name of the tiktoken encoding (default: "cl100k_base")
        """
        self.chunker = SentenceChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.sentence_chunker = SentenceChunker(chunk_size=1, chunk_overlap=0)
        self.encoder = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, text: str) -> list[Any]:
        """Split text into chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        return self.chunker(text)

    def extract_sentences(self, chunks: list[Any]) -> list[list[Any]]:
        """Extract sentences from each chunk.

        Args:
            chunks: List of text chunks

        Returns:
            List of sentence lists, one per chunk
        """
        return [self.sentence_chunker(chunk.text) for chunk in chunks]

    def compute_similarities(
        self, chunk_embeddings: np.ndarray, sentence_embeddings: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Compute cosine similarity between sentences and their parent chunks.

        Args:
            chunk_embeddings: Array of chunk embeddings (num_chunks, embedding_dim)
            sentence_embeddings: List of sentence embedding arrays per chunk

        Returns:
            List of similarity arrays, one per chunk
        """
        similarities = []
        for chunk_idx, sentence_embeds in enumerate(sentence_embeddings):
            chunk_embedding = chunk_embeddings[chunk_idx].reshape(1, -1)
            sim = cosine_similarity(chunk_embedding, sentence_embeds)
            similarities.append(sim.flatten())
        return similarities

    def compute_length_ratios(
        self, chunks: list[Any], sentences: list[list[Any]]
    ) -> list[float]:
        """Compute sentence-to-chunk length ratios.

        Args:
            chunks: List of text chunks
            sentences: List of sentence lists per chunk

        Returns:
            Flattened list of length ratios for all sentences
        """
        ratios = []
        for chunk_idx, sentence_list in enumerate(sentences):
            chunk_length = len(chunks[chunk_idx].text)
            for sentence in sentence_list:
                sentence_length = len(sentence.text)
                ratios.append(sentence_length / chunk_length)
        return ratios

    def count_tokens(self, sentences: list[list[Any]]) -> list[int]:
        """Count tokens for each sentence.

        Args:
            sentences: List of sentence lists per chunk

        Returns:
            Flattened list of token counts for all sentences
        """
        tokens = []
        for sentence_list in sentences:
            for sentence in sentence_list:
                tokens.append(len(self.encoder.encode(sentence.text)))
        return tokens

    def select_sentences(
        self, sentences: list[list[Any]], mask: np.ndarray
    ) -> list[str]:
        """Select sentences based on binary mask.

        Args:
            sentences: List of sentence lists per chunk
            mask: Binary mask indicating which sentences to select

        Returns:
            List of selected sentence texts in order of appearance
        """
        selected = []
        idx = 0
        for sentence_list in sentences:
            for sentence in sentence_list:
                if mask[idx] == 1:
                    selected.append(sentence.text)
                idx += 1
        return selected

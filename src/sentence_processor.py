"""Sentence processing and analysis module."""

from typing import Any, Optional

import numpy as np
import tiktoken
from chonkie import SentenceChunker
from model2vec import StaticModel
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .sentence_splitter import SentenceSplitter
except ImportError:
    SentenceSplitter = None


class SentenceProcessor:
    """Process and analyze sentences from text chunks."""

    def __init__(
        self,
        embedding_model_name: str = "minishlab/potion-base-8M",
        chunk_size: int = 2048,
        min_sentence_length: int = 256,
        encoding_name: str = "cl100k_base",
        custom_parameters: Optional[dict[str, Any]] = None,
    ):
        """Initialize the sentence processor.

        Args:
            embedding_model_name: Name of the embedding model (default: "minishlab/potion-base-8M")
            chunk_size: Size of text chunks in characters (default: 2048)
            min_sentence_length: Minimum sentence length in characters (default: 256)
            encoding_name: Name of the tiktoken encoding (default: "cl100k_base")
            custom_parameters: Optional dict of parameters for SentenceSplitter
                             (e.g., {"prefixes": ["H.R", "S"], "acronyms": "..."}).
                             If None, uses Chonkie's SentenceChunker (default: None)
        """
        self.embedding_model = StaticModel.from_pretrained(embedding_model_name)

        # Initialize chunkers based on whether custom parameters are provided
        if custom_parameters is not None:
            # Use SentenceSplitter with custom parameters for both chunkers
            self.chunker = SentenceSplitter(
                chunk_size=chunk_size, chunk_overlap=0, **custom_parameters
            )
            self.sentence_chunker = SentenceSplitter(
                chunk_size=min_sentence_length, chunk_overlap=0, **custom_parameters
            )
            self.use_custom_splitter = True
        else:
            # Use Chonkie's SentenceChunker for both
            self.chunker = SentenceChunker(chunk_size=chunk_size, chunk_overlap=0)
            self.sentence_chunker = SentenceChunker(
                chunk_size=min_sentence_length, chunk_overlap=0
            )
            self.use_custom_splitter = False

        self.encoder = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, text: str) -> list[Any]:
        """Split text into chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if self.use_custom_splitter:
            # SentenceSplitter returns list of strings, wrap them in objects with .text attribute
            chunks = self.chunker.split_text(text)
            return [type("Chunk", (), {"text": chunk})() for chunk in chunks]
        else:
            # Chonkie returns chunk objects with .text attribute
            return self.chunker(text)

    def extract_sentences(self, chunks: list[Any]) -> list[list[Any]]:
        """Extract sentences from each chunk.

        Args:
            chunks: List of text chunks

        Returns:
            List of sentence lists, one per chunk
        """
        result = []
        for chunk in chunks:
            if self.use_custom_splitter:
                # SentenceSplitter returns list of strings
                sentences = self.sentence_chunker.split_text(chunk.text)
            else:
                # Chonkie returns chunk objects with .text attribute
                sentences = [s.text for s in self.sentence_chunker(chunk.text)]

            # Wrap each sentence string in a simple object with .text attribute
            sentence_objs = [
                type("Sentence", (), {"text": sent})() for sent in sentences
            ]
            result.append(sentence_objs)

        return result

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

    def select_sentences_with_separators(
        self, sentences: list[list[Any]], mask: np.ndarray, separator: str = " (...) "
    ) -> str:
        """Select sentences and join them with separators for non-consecutive sentences.

        Args:
            sentences: List of sentence lists per chunk
            mask: Binary mask indicating which sentences to select
            separator: String to insert between non-consecutive sentences (default: " (...) ")

        Returns:
            String with selected sentences joined, using separator for gaps
        """
        selected_parts = []
        idx = 0
        prev_idx = -2  # Initialize to ensure first sentence doesn't get a separator

        for sentence_list in sentences:
            for sentence in sentence_list:
                if mask[idx] == 1:
                    # Add separator if there's a gap from the previous selected sentence
                    if prev_idx >= 0 and idx != prev_idx + 1:
                        selected_parts.append(separator)
                    selected_parts.append(sentence.text)
                    prev_idx = idx
                idx += 1

        return "".join(selected_parts)

    def compute_document_features(self, text: str) -> dict:
        """Compute similarities, ratios, and tokens for all sentences in a document.

        Args:
            text: Input document text

        Returns:
            Dictionary containing:
                - chunks: List of text chunks
                - sentences: List of sentence lists per chunk
                - similarities: Cosine similarities between sentences and chunks
                - all_similarities: Flattened array of all similarities
                - ratios: Sentence-to-chunk length ratios
                - tokens: Token counts per sentence
                - total_tokens: Total token count
        """
        chunks = self.chunk_text(text)
        chunk_embeddings = self.embedding_model.encode([chunk.text for chunk in chunks])
        sentences = self.extract_sentences(chunks)
        sentence_embeddings = [
            self.embedding_model.encode([sentence.text for sentence in sentence_list])
            for sentence_list in sentences
        ]
        similarities = self.compute_similarities(chunk_embeddings, sentence_embeddings)
        ratios = self.compute_length_ratios(chunks, sentences)
        tokens = self.count_tokens(sentences)
        all_similarities = np.array(
            [sim for sublist in similarities for sim in sublist]
        )
        all_sentences = [
            sentence.text for sentence_list in sentences for sentence in sentence_list
        ]

        return {
            "chunks": chunks,
            "sentences": sentences,
            "all_sentences": all_sentences,
            "similarities": similarities,
            "all_similarities": all_similarities,
            "ratios": np.array(ratios),
            "tokens": np.array(tokens),
            "total_tokens": sum(tokens),
        }

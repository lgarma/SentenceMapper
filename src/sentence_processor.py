"""Sentence processing and analysis module.

Splits the full document into sentences, then for each sentence builds a
centered context window of complete surrounding sentences (excluding the
target sentence itself) up to a character budget.  Cosine similarity is
computed between the sentence embedding and its context embedding, giving a
clean representativeness signal without self-overlap artifacts.
"""

from typing import Any, Optional

import numpy as np
import tiktoken
from model2vec import StaticModel
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .sentence_splitter import SentenceSplitter
except ImportError:
    SentenceSplitter = None


class SentenceProcessor:
    """Process and analyze sentences using centered context windows."""

    def __init__(
        self,
        embedding_model_name: str = "minishlab/potion-base-8M",
        context_budget: int = 2048,
        min_sentence_length: int = 256,
        encoding_name: str = "cl100k_base",
        custom_parameters: Optional[dict[str, Any]] = None,
    ):
        """Initialize the sentence processor.

        Args:
            embedding_model_name: Name of the embedding model
                (default: "minishlab/potion-base-8M")
            context_budget: Maximum character budget for the context window
                surrounding each sentence (default: 2048)
            min_sentence_length: Minimum sentence length in characters.
                Shorter sentences are merged (default: 256)
            encoding_name: Name of the tiktoken encoding
                (default: "cl100k_base")
            custom_parameters: Optional dict of parameters for
                SentenceSplitter (e.g., {"prefixes": ["H.R", "S"]}).
                If None, uses default SentenceSplitter settings
                (default: None)
        """
        self.embedding_model = StaticModel.from_pretrained(embedding_model_name)
        self.context_budget = context_budget
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.context_delimiter = "(...)"

        # Sentence splitter — merges short sentences up to min_sentence_length
        if custom_parameters is None:
            custom_parameters = {}

        self.sentence_splitter = SentenceSplitter(
            chunk_size=min_sentence_length,
            chunk_overlap=0,
            **custom_parameters,
        )

    # ------------------------------------------------------------------
    # Sentence extraction
    # ------------------------------------------------------------------

    def extract_sentences(self, text: str) -> list[str]:
        """Split the full document into sentences.

        Args:
            text: Full document text

        Returns:
            List of sentence strings
        """
        return self.sentence_splitter.split_text(text)

    # ------------------------------------------------------------------
    # Context window construction
    # ------------------------------------------------------------------

    def build_context_window(self, sentences: list[str], target_idx: int) -> str:
        """Build a context window of complete sentences around the target.

        Expands outward from the target sentence, alternating left and right,
        adding complete sentences until the character budget is reached.
        The target sentence itself is EXCLUDED from the context.

        Args:
            sentences: List of all document sentences
            target_idx: Index of the target sentence

        Returns:
            Context string with a delimiter showing where the target
            sentence would be located.
        """
        budget = self.context_budget
        left_idx = target_idx - 1
        right_idx = target_idx + 1
        left_parts: list[str] = []
        right_parts: list[str] = []
        current_length = 0

        while current_length < budget and (left_idx >= 0 or right_idx < len(sentences)):
            # Try adding from the left
            if left_idx >= 0:
                candidate = sentences[left_idx]
                if current_length + len(candidate) <= budget:
                    left_parts.append(candidate)
                    current_length += len(candidate)
                    left_idx -= 1
                else:
                    left_idx = -1  # stop expanding left

            # Try adding from the right
            if right_idx < len(sentences):
                candidate = sentences[right_idx]
                if current_length + len(candidate) <= budget:
                    right_parts.append(candidate)
                    current_length += len(candidate)
                    right_idx += 1
                else:
                    right_idx = len(sentences)  # stop expanding right

            # If neither side could add, break
            if (left_idx < 0 or current_length >= budget) and (
                right_idx >= len(sentences) or current_length >= budget
            ):
                break

        # Reconstruct in document order: reversed left + right
        left_context = " ".join(reversed(left_parts)).strip()
        right_context = " ".join(right_parts).strip()
        delimiter = self.context_delimiter.strip()

        if not delimiter:
            context_parts = [p for p in [left_context, right_context] if p]
            return " ".join(context_parts)

        if left_context and right_context:
            return f"{left_context} {delimiter} {right_context}"
        if left_context:
            return f"{left_context} {delimiter}"
        if right_context:
            return f"{delimiter} {right_context}"
        return delimiter

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def compute_document_features(self, text: str) -> dict:
        """Compute similarities, ratios, and tokens for all sentences.

        For each sentence:
        - Builds a centered context window of surrounding sentences
        - Computes cosine similarity between the sentence embedding and the
          context embedding (sentence excluded from context)
        - Computes ratio = len(sentence) / (len(sentence) + len(context))

        Args:
            text: Full document text

        Returns:
            Dictionary containing:
                - sentences: list[str]
                - similarities: np.ndarray of cosine similarities
                - ratios: np.ndarray of length ratios
                - tokens: np.ndarray of token counts per sentence
                - total_tokens: int
                - contexts: list[str]  (context window for each sentence)
        """
        sentences = self.extract_sentences(text)
        n = len(sentences)

        # Build context windows for every sentence
        contexts = [self.build_context_window(sentences, i) for i in range(n)]

        # Embed sentences and contexts
        sentence_embeddings = self.embedding_model.encode(sentences)
        context_embeddings = self.embedding_model.encode(contexts)

        # Cosine similarity between each sentence and its own context
        similarities = np.array(
            [
                cosine_similarity(
                    sentence_embeddings[i].reshape(1, -1),
                    context_embeddings[i].reshape(1, -1),
                )[0, 0]
                for i in range(n)
            ]
        )

        # Ratios: sentence length / (sentence length + context length)
        ratios = np.array(
            [
                len(sentences[i]) / (len(sentences[i]) + len(contexts[i]))
                if len(contexts[i]) > 0
                else 1.0
                for i in range(n)
            ]
        )

        # Token counts
        tokens = np.array([len(self.encoder.encode(s)) for s in sentences])

        return {
            "sentences": sentences,
            "similarities": similarities,
            "ratios": ratios,
            "tokens": tokens,
            "total_tokens": int(tokens.sum()),
            "contexts": contexts,
        }

    # ------------------------------------------------------------------
    # Sentence selection helpers
    # ------------------------------------------------------------------

    def select_sentences(self, sentences: list[str], mask: np.ndarray) -> list[str]:
        """Select sentences based on binary mask.

        Args:
            sentences: List of sentence strings
            mask: Binary mask indicating which sentences to select

        Returns:
            List of selected sentence texts
        """
        return [s for s, m in zip(sentences, mask) if m == 1]

    def select_sentences_with_separators(
        self,
        sentences: list[str],
        mask: np.ndarray,
        separator: str = " (...) ",
    ) -> str:
        """Select sentences and join with separators for non-consecutive gaps.

        Args:
            sentences: List of sentence strings
            mask: Binary mask indicating which sentences to select
            separator: String to insert between non-consecutive sentences

        Returns:
            Joined string of selected sentences
        """
        selected_parts: list[str] = []
        prev_idx = -2

        for idx, (sentence, m) in enumerate(zip(sentences, mask)):
            if m == 1:
                if prev_idx >= 0 and idx != prev_idx + 1:
                    selected_parts.append(separator)
                selected_parts.append(sentence)
                prev_idx = idx

        return "".join(selected_parts)

"""End-to-end sentence mapping pipeline.

Uses sentence-centered context windows to identify information-dense
sentences and select a subset that preserves the essential meaning of
a document within a target token budget.

Each sentence is scored by:

    score = similarity - length_bias × ratio

where *similarity* is the cosine similarity between the sentence and its
surrounding context (sentence excluded), *ratio* is
``len(sentence) / (len(sentence) + len(context))``, and *length_bias*
(α) is a linear penalty that mildly favours shorter sentences at equal
similarity.  α = 0 gives pure similarity ranking.

The additive form composes naturally with future bias terms
(e.g. ``+ β × query_similarity`` for semantic-biased extraction).

Sentences are ranked by score and greedily selected from the top until
the token budget is reached.
"""

from typing import Optional

import numpy as np

from .sentence_processor import SentenceProcessor


class SentenceMapperPipeline:
    """Complete pipeline for sentence mapping with optimization."""

    def __init__(
        self,
        embedding_model_name: str = "minishlab/potion-base-8M",
        context_budget: int = 2048,
        min_sentence_length: int = 256,
        encoding_name: str = "cl100k_base",
        custom_parameters: Optional[dict] = None,
        length_bias: float = 0.5,
    ):
        """Initialize the pipeline.

        Args:
            embedding_model_name: Name of the embedding model
                (default: "minishlab/potion-base-8M")
            context_budget: Character budget for the context window around
                each sentence (default: 2048)
            min_sentence_length: Minimum sentence length in characters
                (default: 256)
            encoding_name: Name of the tiktoken encoding
                (default: "cl100k_base")
            custom_parameters: Optional dict of parameters for
                SentenceSplitter (e.g., {"prefixes": ["H.R", "S"]}).
                If None, uses default SentenceSplitter settings
                (default: None)
            length_bias: Linear penalty α applied to the length ratio.
                ``score = similarity - α × ratio``.  0 = pure similarity
                ranking, higher values penalise longer sentences more.
                (default: 0.5)
        """
        self.processor = SentenceProcessor(
            embedding_model_name=embedding_model_name,
            context_budget=context_budget,
            min_sentence_length=min_sentence_length,
            encoding_name=encoding_name,
            custom_parameters=custom_parameters,
        )
        self.length_bias = length_bias

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compute_scores(
        similarities: np.ndarray,
        ratios: np.ndarray,
        length_bias: float = 0.5,
    ) -> np.ndarray:
        """Compute sentence scores.

        ``score = similarity - length_bias × ratio``

        Args:
            similarities: Cosine similarities (sentence vs context)
            ratios: Length ratios per sentence
            length_bias: Linear penalty α (default: 0.5)

        Returns:
            Array of scores, one per sentence.
        """
        return similarities - length_bias * ratios

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_by_ranking(
        self,
        features: dict,
        scores: np.ndarray,
        objective_percentage: float,
    ) -> dict:
        """Rank sentences by *scores* and greedily fill the token budget.

        Args:
            features: Dictionary returned by ``compute_document_features``
            scores: Pre-computed score array (one per sentence)
            objective_percentage: Target fraction of tokens to keep (0–1)

        Returns:
            Dictionary with mask, selected_sentences, selected_text,
            selected_tokens.
        """
        tokens = features["tokens"]
        total_tokens = int(np.sum(tokens))
        objective_tokens = total_tokens * objective_percentage

        ranked_indices = np.argsort(-scores)

        mask = np.zeros(len(scores), dtype=int)
        current_tokens = 0
        for idx in ranked_indices:
            if current_tokens + tokens[idx] > objective_tokens:
                continue
            mask[idx] = 1
            current_tokens += tokens[idx]

        selected_sentences = self.processor.select_sentences(
            features["sentences"], mask
        )
        selected_text = self.processor.select_sentences_with_separators(
            features["sentences"], mask
        )

        return {
            "mask": mask,
            "selected_sentences": selected_sentences,
            "selected_text": selected_text,
            "selected_tokens": int(np.sum(tokens * mask)),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_document(
        self,
        text: str,
        objective_percentage: float | None = None,
        length_bias: float | None = None,
    ) -> dict:
        """Process a document and optionally select sentences to a target %.

        Args:
            text: Input document text
            objective_percentage: Target percentage of tokens to select
                (e.g. 0.3 = 30%).  If None, returns features only.
            length_bias: Override the instance-level length_bias for this
                call.  If None, uses ``self.length_bias``.

        Returns:
            Dictionary containing:
                - sentences: list[str]
                - similarities: np.ndarray
                - ratios: np.ndarray
                - tokens: np.ndarray
                - total_tokens: int
                - contexts: list[str]
                - scores: np.ndarray  (similarity - α × ratio)
                - length_bias: float  (α used)
            When *objective_percentage* is set, also:
                - mask: np.ndarray (binary)
                - selected_sentences: list[str]
                - selected_text: str
                - selected_tokens: int
        """
        alpha = length_bias if length_bias is not None else self.length_bias

        features = self.processor.compute_document_features(text)

        scores = self.compute_scores(
            features["similarities"],
            features["ratios"],
            length_bias=alpha,
        )
        features["scores"] = scores
        features["length_bias"] = alpha

        if objective_percentage is not None:
            selection = self._select_by_ranking(features, scores, objective_percentage)
            features.update(selection)

        return features

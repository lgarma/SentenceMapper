"""End-to-end sentence mapping pipeline.

Uses sentence-centered context windows to identify information-dense
sentences and select a subset that preserves the essential meaning of
a document within a target token budget.

Each sentence is scored by:

    score = similarity - α × ratio + γ × global_similarity

where:
- *similarity* is the cosine similarity between the sentence and its
  surrounding context (sentence excluded)
- *ratio* is ``len(sentence) / (len(sentence) + len(context))``
- *length_bias* (α) is a linear penalty that mildly favours shorter
  sentences at equal similarity (default: 0.5)
- *global_similarity* is the cosine similarity between the sentence and
  the full document embedding
- *global_context_weight* (γ) weights the global context term (default: 0.25)

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
        global_context_weight: float = 0.25,
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
                ``score = similarity - α × ratio + γ × global_similarity``.
                0 = pure similarity ranking, higher values penalise longer
                sentences more. (default: 0.5)
            global_context_weight: Weight γ for global context similarity.
                ``score = similarity - α × ratio + γ × global_similarity``.
                0 = local context only. (default: 0.25)
        """
        self.processor = SentenceProcessor(
            embedding_model_name=embedding_model_name,
            context_budget=context_budget,
            min_sentence_length=min_sentence_length,
            encoding_name=encoding_name,
            custom_parameters=custom_parameters,
        )
        self.length_bias = length_bias
        self.global_context_weight = global_context_weight

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compute_scores(
        similarities: np.ndarray,
        ratios: np.ndarray,
        length_bias: float = 0.5,
        global_similarities: np.ndarray | None = None,
        global_context_weight: float = 0.25,
    ) -> np.ndarray:
        """Compute sentence scores.

        ``score = similarity - α × ratio + γ × global_similarity``

        Args:
            similarities: Local cosine similarities (sentence vs context)
            ratios: Length ratios per sentence
            length_bias: Linear penalty α (default: 0.5)
            global_similarities: Global cosine similarities (sentence vs
                full document). If None, global term is omitted.
            global_context_weight: Weight γ for global context (default: 0.25)

        Returns:
            Array of scores, one per sentence.
        """
        score = similarities - length_bias * ratios
        if global_similarities is not None:
            score = score + global_context_weight * global_similarities
        return score

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_target_tokens(
        total_tokens: int,
        objective_percentage: float | None = None,
        objective_tokens: int | None = None,
    ) -> int:
        """Compute the target token budget from percentage and/or absolute tokens.

        Args:
            total_tokens: Total tokens in the document
            objective_percentage: Target fraction of tokens (0-1), or None
            objective_tokens: Absolute target token count, or None

        Returns:
            Target token budget. If both are provided, returns the minimum.
            If neither is provided, raises ValueError.
        """
        if objective_percentage is None and objective_tokens is None:
            raise ValueError(
                "At least one of objective_percentage or objective_tokens must be provided"
            )

        candidates = []
        if objective_percentage is not None:
            candidates.append(int(total_tokens * objective_percentage))
        if objective_tokens is not None:
            candidates.append(int(objective_tokens))

        return min(candidates)

    def _select_by_ranking(
        self,
        features: dict,
        scores: np.ndarray,
        objective_percentage: float | None = None,
        objective_tokens: int | None = None,
    ) -> dict:
        """Rank sentences by *scores* and greedily fill the token budget.

        Args:
            features: Dictionary returned by ``compute_document_features``
            scores: Pre-computed score array (one per sentence)
            objective_percentage: Target fraction of tokens to keep (0-1), or None
            objective_tokens: Absolute target token count, or None.
                If both are provided, uses the minimum.

        Returns:
            Dictionary with mask, selected_sentences, selected_text,
            selected_tokens, target_tokens.
        """
        tokens = features["tokens"]
        total_tokens = int(np.sum(tokens))
        target_tokens = self._compute_target_tokens(
            total_tokens, objective_percentage, objective_tokens
        )

        ranked_indices = np.argsort(-scores)

        mask = np.zeros(len(scores), dtype=int)
        current_tokens = 0
        for idx in ranked_indices:
            if current_tokens + tokens[idx] > target_tokens:
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
            "target_tokens": target_tokens,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_document(
        self,
        text: str,
        objective_percentage: float | None = None,
        objective_tokens: int | None = None,
        length_bias: float | None = None,
        global_context_weight: float | None = None,
    ) -> dict:
        """Process a document and optionally select sentences to a target.

        Args:
            text: Input document text
            objective_percentage: Target fraction of tokens to select
                (e.g. 0.3 = 30%). If None and objective_tokens is None,
                returns features only.
            objective_tokens: Absolute target token count. If both
                objective_percentage and objective_tokens are provided,
                uses the minimum of the two.
            length_bias: Override the instance-level length_bias for this
                call. If None, uses ``self.length_bias``.
            global_context_weight: Override the instance-level
                global_context_weight for this call. If None, uses
                ``self.global_context_weight``.

        Returns:
            Dictionary containing:
                - sentences: list[str]
                - similarities: np.ndarray (local)
                - global_similarities: np.ndarray
                - ratios: np.ndarray
                - tokens: np.ndarray
                - total_tokens: int
                - contexts: list[str]
                - scores: np.ndarray  (similarity - α × ratio + γ × global)
                - length_bias: float  (α used)
                - global_context_weight: float  (γ used)
            When objective_percentage or objective_tokens is set, also:
                - mask: np.ndarray (binary)
                - selected_sentences: list[str]
                - selected_text: str
                - selected_tokens: int
                - target_tokens: int
        """
        alpha = length_bias if length_bias is not None else self.length_bias
        gamma = (
            global_context_weight
            if global_context_weight is not None
            else self.global_context_weight
        )

        features = self.processor.compute_document_features(text)

        scores = self.compute_scores(
            features["similarities"],
            features["ratios"],
            length_bias=alpha,
            global_similarities=features["global_similarities"],
            global_context_weight=gamma,
        )
        features["scores"] = scores
        features["length_bias"] = alpha
        features["global_context_weight"] = gamma

        if objective_percentage is not None or objective_tokens is not None:
            selection = self._select_by_ranking(
                features, scores, objective_percentage, objective_tokens
            )
            features.update(selection)

        return features

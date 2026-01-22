"""End-to-end sentence mapping pipeline."""

from typing import Optional

import numpy as np

from .powerlaw_optimizer import PowerLawOptimizer, fit_frontier_curve
from .sentence_processor import SentenceProcessor


class SentenceMapperPipeline:
    """Complete pipeline for sentence mapping with optimization."""

    def __init__(
        self,
        embedding_model_name: str = "minishlab/potion-base-8M",
        chunk_size: int = 2048,
        min_sentence_length: int = 256,
        encoding_name: str = "cl100k_base",
        custom_parameters: Optional[dict] = None,
    ):
        """Initialize the pipeline.

        Args:
            embedding_model_name: Name of the embedding model (default: "minishlab/potion-base-8M")
            chunk_size: Size of text chunks in characters (default: 2048)
            min_sentence_length: Minimum sentence length in characters (default: 256)
            encoding_name: Name of the tiktoken encoding (default: "cl100k_base")
            custom_parameters: Optional dict of parameters for SentenceSplitter
                             (e.g., {"prefixes": ["H.R", "S"], "acronyms": "..."}).
                             If None, uses Chonkie's SentenceChunker (default: None)
        """
        self.processor = SentenceProcessor(
            embedding_model_name,
            chunk_size,
            min_sentence_length,
            encoding_name,
            custom_parameters,
        )

    def apply_sentence_filter(
        self,
        features: dict,
        mask: np.ndarray,
        x_opt: float,
    ) -> dict:
        """Apply filtering to select sentences based on optimizer criteria.

        Args:
            features: Dictionary returned by compute_document_features
            objective_percentage: Target percentage of tokens to select (e.g., 0.2 = 20%)

        Returns:
            Dictionary containing:
                - mask: Binary mask of selected sentences
                - x_opt: Optimal parameter value
                - selected_sentences: List of selected sentence texts
                - selected_text: Selected sentences joined with separators
                - selected_tokens: Total token count of selected sentences
                - params: Optimizer parameters used
        """
        selected_sentences = self.processor.select_sentences(
            features["sentences"], mask
        )
        selected_text = self.processor.select_sentences_with_separators(
            features["sentences"], mask
        )

        return {
            "mask": mask,
            "x_opt": x_opt,
            "selected_sentences": selected_sentences,
            "selected_text": selected_text,
            "selected_tokens": np.sum(features["tokens"] * mask),
        }

    def process_document(
        self,
        text: str,
        objective_percentage: float | None = None,
        fit_method: str = "quantile",
    ) -> dict:
        """Process a document and optionally filter sentences to target percentage.

        Args:
            text: Input document text
            objective_percentage: Target percentage of tokens to select (e.g., 0.2 = 20%).
                                 If None, no filtering is applied (default: None)

        Returns:
            Dictionary containing:
                - chunks: List of text chunks
                - sentences: List of sentence lists per chunk
                - similarities: Cosine similarities between sentences and chunks
                - ratios: Sentence-to-chunk length ratios
                - tokens: Token counts per sentence
                - selected_sentences: List of selected sentence texts (if objective_percentage is set)
                - mask: Binary mask of selected sentences (if objective_percentage is set)
                - x_opt: Optimal parameter value (if objective_percentage is set)
                - params: Power law parameters used (if objective_percentage is set)
        """
        features = self.processor.compute_document_features(text)

        slope, intercept, _ = fit_frontier_curve(
            features["all_similarities"],
            features["ratios"],
            quantile=0.95,
            method=fit_method,
        )

        optimizer = PowerLawOptimizer(slope=slope, intercept=intercept)
        mask, x_opt = optimizer.filter_sentences(
            features["all_similarities"],
            features["ratios"],
            features["tokens"],
            objective_percentage,
        )
        features["params"] = optimizer.get_params(x_opt)

        # Apply filtering if objective percentage is specified
        if objective_percentage is not None:
            filter_results = self.apply_sentence_filter(features, mask, x_opt)
            features.update(filter_results)

        return features

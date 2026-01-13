"""End-to-end sentence mapping pipeline."""

import numpy as np
from model2vec import StaticModel
from typing import Literal

from .sigmoid_optimizer import SigmoidOptimizer
from .sentence_processor import SentenceProcessor


class SentenceMapperPipeline:
    """Complete pipeline for sentence mapping with optimization."""

    def __init__(
        self,
        embedding_model_name: str = "minishlab/potion-base-8M",
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        encoding_name: str = "cl100k_base",
        strategy: Literal[
            "balanced", "short_sentences", "high_similarity"
        ] = "balanced",
    ):
        """Initialize the pipeline.

        Args:
            embedding_model_name: Name of the embedding model (default: "minishlab/potion-base-8M")
            chunk_size: Size of text chunks in tokens (default: 2048)
            chunk_overlap: Overlap between chunks in tokens (default: 0)
            encoding_name: Name of the tiktoken encoding (default: "cl100k_base")
            strategy: Selection strategy (default: "balanced")
                - "balanced": Balances similarity and length
                - "short_sentences": Prefers shorter sentences
                - "high_similarity": Prefers high similarity sentences
        """
        self.embedding_model = StaticModel.from_pretrained(embedding_model_name)
        self.processor = SentenceProcessor(chunk_size, chunk_overlap, encoding_name)
        self.optimizer = SigmoidOptimizer(strategy)

    def process_document(
        self, text: str, objective_percentage: float | None = None
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
                - params: Sigmoid parameters used (if objective_percentage is set)
        """
        chunks = self.processor.chunk_text(text)
        chunk_embeddings = self.embedding_model.encode([chunk.text for chunk in chunks])
        sentences = self.processor.extract_sentences(chunks)
        sentence_embeddings = [
            self.embedding_model.encode([sentence.text for sentence in sentence_list])
            for sentence_list in sentences
        ]
        similarities = self.processor.compute_similarities(
            chunk_embeddings, sentence_embeddings
        )
        ratios = self.processor.compute_length_ratios(chunks, sentences)
        tokens = self.processor.count_tokens(sentences)
        all_similarities = np.array(
            [sim for sublist in similarities for sim in sublist]
        )

        result = {
            "chunks": chunks,
            "sentences": sentences,
            "similarities": similarities,
            "all_similarities": all_similarities,
            "ratios": np.array(ratios),
            "tokens": np.array(tokens),
            "total_tokens": sum(tokens),
        }

        # Apply filtering if objective percentage is specified
        if objective_percentage is not None:
            mask, x_opt = self.optimizer.filter_sentences(
                all_similarities,
                np.array(ratios),
                np.array(tokens),
                objective_percentage,
            )
            selected_sentences = self.processor.select_sentences(sentences, mask)

            result.update(
                {
                    "mask": mask,
                    "x_opt": x_opt,
                    "selected_sentences": selected_sentences,
                    "selected_text": "\n".join(selected_sentences),
                    "selected_tokens": np.sum(np.array(tokens) * mask),
                    "params": self.optimizer.get_params(x_opt),
                }
            )

        return result

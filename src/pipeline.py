"""End-to-end sentence mapping pipeline."""

import numpy as np
from model2vec import StaticModel
from typing import Literal

from .optimizer import ArctanOptimizer
from .sentence_processor import SentenceProcessor


class SentenceMapperPipeline:
    """Complete pipeline for sentence mapping with optimization."""

    def __init__(
        self,
        embedding_model_name: str = "minishlab/potion-base-8M",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        encoding_name: str = "cl100k_base",
        gamma: float = 0.2,
        strategy: Literal["balanced", "short_sentences", "high_similarity"] = "balanced"
    ):
        """Initialize the pipeline.
        
        Args:
            embedding_model_name: Name of the embedding model (default: "minishlab/potion-base-8M")
            chunk_size: Size of text chunks in tokens (default: 512)
            chunk_overlap: Overlap between chunks in tokens (default: 128)
            encoding_name: Name of the tiktoken encoding (default: "cl100k_base")
            gamma: Steepness parameter for arctan optimizer (default: 0.2)
            strategy: Selection strategy (default: "balanced")
                - "balanced": Balances similarity and length
                - "short_sentences": Prefers shorter sentences
                - "high_similarity": Prefers high similarity sentences
        """
        self.embedding_model = StaticModel.from_pretrained(embedding_model_name)
        self.processor = SentenceProcessor(chunk_size, chunk_overlap, encoding_name)
        self.optimizer = ArctanOptimizer(gamma, strategy)

    def process_document(
        self, 
        text: str, 
        objective_percentage: float | None = None
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
        """
        # Chunk the text
        chunks = self.processor.chunk_text(text)
        
        # Generate chunk embeddings
        chunk_embeddings = self.embedding_model.encode([chunk.text for chunk in chunks])
        
        # Extract sentences
        sentences = self.processor.extract_sentences(chunks)
        
        # Generate sentence embeddings
        sentence_embeddings = [
            self.embedding_model.encode([sentence.text for sentence in sentence_list])
            for sentence_list in sentences
        ]
        
        # Compute similarities
        similarities = self.processor.compute_similarities(chunk_embeddings, sentence_embeddings)
        
        # Compute length ratios
        ratios = self.processor.compute_length_ratios(chunks, sentences)
        
        # Count tokens
        tokens = self.processor.count_tokens(sentences)
        
        # Flatten similarities for optimization
        all_similarities = np.array([sim for sublist in similarities for sim in sublist])
        
        result = {
            "chunks": chunks,
            "sentences": sentences,
            "similarities": similarities,
            "all_similarities": all_similarities,
            "ratios": np.array(ratios),
            "tokens": np.array(tokens),
            "total_tokens": sum(tokens)
        }
        
        # Apply filtering if objective percentage is specified
        if objective_percentage is not None:
            mask, x_opt = self.optimizer.filter_sentences(
                all_similarities,
                np.array(ratios),
                np.array(tokens),
                objective_percentage
            )
            selected_sentences = self.processor.select_sentences(sentences, mask)
            
            result.update({
                "mask": mask,
                "x_opt": x_opt,
                "selected_sentences": selected_sentences,
                "selected_text": "\n".join(selected_sentences),
                "selected_tokens": np.sum(np.array(tokens) * mask)
            })
        
        return result

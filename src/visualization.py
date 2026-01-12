"""Visualization utilities for sentence mapping analysis."""

import numpy as np
import matplotlib.pyplot as plt


class SentenceMapperVisualizer:
    """Visualization tools for sentence mapping results."""

    @staticmethod
    def plot_similarity_vs_ratio(
        similarities: np.ndarray,
        ratios: np.ndarray,
        mask: np.ndarray | None = None,
        x_opt: float | None = None,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> None:
        """Plot scatter plot of similarity vs. length ratio.

        Args:
            similarities: Array of cosine similarities
            ratios: Array of sentence-to-chunk length ratios
            mask: Optional binary mask indicating selected sentences
            x_opt: Optional optimal parameter value for title
            figsize: Figure size (default: (10, 6))
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=figsize)

        if mask is not None:
            # Plot all sentences with low opacity
            plt.scatter(similarities, ratios, alpha=0.2, label="All Sentences", s=20)

            # Highlight filtered sentences
            plt.scatter(
                similarities[mask == 1],
                ratios[mask == 1],
                color="red",
                alpha=0.5,
                label="Selected Sentences",
                s=20,
            )

            title = "Filtered Sentences"
            if x_opt is not None:
                title += f" (x_opt={x_opt:.3f})"
        else:
            plt.scatter(similarities, ratios, alpha=0.5, s=20)
            title = "Sentence Similarity vs. Length Ratio"

        plt.xlabel("Cosine Similarity to Chunk", fontsize=12)
        plt.ylabel("Sentence to Chunk Length Ratio", fontsize=12)
        plt.title(title, fontsize=14)

        if mask is not None:
            plt.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

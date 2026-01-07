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
        save_path: str | None = None
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
                color='red',
                alpha=0.5,
                label="Selected Sentences",
                s=20
            )
            
            title = f"Filtered Sentences"
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
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    @staticmethod
    def plot_token_distribution(
        tokens: np.ndarray,
        mask: np.ndarray | None = None,
        bins: int = 50,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None
    ) -> None:
        """Plot histogram of token distribution.
        
        Args:
            tokens: Array of token counts per sentence
            mask: Optional binary mask indicating selected sentences
            bins: Number of histogram bins (default: 50)
            figsize: Figure size (default: (10, 6))
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=figsize)
        
        if mask is not None:
            plt.hist(tokens, bins=bins, alpha=0.5, label="All Sentences", edgecolor='black')
            plt.hist(tokens[mask == 1], bins=bins, alpha=0.7, label="Selected Sentences", edgecolor='black')
            plt.legend()
        else:
            plt.hist(tokens, bins=bins, alpha=0.7, edgecolor='black')
        
        plt.xlabel("Token Count", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Token Distribution Across Sentences", fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    @staticmethod
    def plot_arctan_curve(
        alpha: float,
        beta: float,
        gamma: float,
        similarities: np.ndarray | None = None,
        ratios: np.ndarray | None = None,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None
    ) -> None:
        """Plot the arctan filtering curve.
        
        Args:
            alpha: Amplitude parameter
            beta: Horizontal shift parameter
            gamma: Steepness parameter
            similarities: Optional array of actual similarities to overlay
            ratios: Optional array of actual ratios to overlay
            figsize: Figure size (default: (10, 6))
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=figsize)
        
        # Generate curve
        x = np.linspace(0, 1, 1000)
        y = alpha * np.arctan((x - beta) / gamma)
        
        plt.plot(x, y, 'b-', linewidth=2, label='Arctan Filter Curve')
        
        # Overlay actual data if provided
        if similarities is not None and ratios is not None:
            plt.scatter(similarities, ratios, alpha=0.3, s=10, color='gray', label='Sentences')
        
        plt.xlabel("Cosine Similarity", fontsize=12)
        plt.ylabel("Length Ratio Threshold", fontsize=12)
        plt.title(f"Arctan Filter Curve (α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f})", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

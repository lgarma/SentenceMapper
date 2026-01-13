"""Visualization utilities for sentence mapping analysis."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from .sigmoid_optimizer import SigmoidOptimizer


class SentenceMapperVisualizer:
    """Visualization tools for sentence mapping results."""

    def __init__(self, strategy) -> None:
        """Initialize the visualizer."""
        self.strategy = strategy
        self.optimizer = SigmoidOptimizer(strategy)

    def plot_similarity_vs_ratio(
        self,
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

            # Plot sigmoid curve
            if x_opt is not None:
                amplitude, midpoint, steepness = self.optimizer.get_params(x_opt)
                sim_range = np.linspace(0, 1, 1000)
                y = self.optimizer.sigmoid(sim_range, amplitude, midpoint, steepness)
                plt.plot(
                    sim_range,
                    y,
                    color="green",
                    linewidth=2,
                    linestyle="--",
                    label="Sigmoid Threshold",
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

    @staticmethod
    def export_highlighted_text(
        sentences: list[list[Any]],
        mask: np.ndarray,
        output_path: str,
        title: str = "Sentence Mapping Results",
    ) -> None:
        """Export text with highlighted selected sentences to an HTML file.

        Args:
            sentences: List of sentence lists per chunk (from SentenceProcessor)
            mask: Binary mask indicating selected sentences (1 = selected, 0 = not selected)
            output_path: Path to save the output file (should end with .html or .md)
            title: Title for the document (default: "Sentence Mapping Results")
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <meta charset='UTF-8'>",
            f"    <title>{title}</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; }",
            "        h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }",
            "        .highlight { background-color: #ffff00; padding: 2px 0; }",
            "        .sentence { display: inline; }",
            "        .chunk { margin-bottom: 20px; padding: 15px; background-color: #f9f9f9; border-left: 4px solid #ddd; }",
            "        .chunk-header { font-weight: bold; color: #666; margin-bottom: 10px; }",
            "    </style>",
            "</head>",
            "<body>",
            f"    <h1>{title}</h1>",
        ]

        idx = 0
        for chunk_idx, sentence_list in enumerate(sentences):
            html_parts.append("    <div class='chunk'>")
            html_parts.append(
                f"        <div class='chunk-header'>Chunk {chunk_idx + 1}</div>"
            )
            html_parts.append("        <div>")

            for sentence in sentence_list:
                sentence_text = sentence.text.strip()
                if mask[idx] == 1:
                    html_parts.append(
                        f"            <span class='sentence highlight'>{sentence_text}</span> "
                    )
                else:
                    html_parts.append(
                        f"            <span class='sentence'>{sentence_text}</span> "
                    )
                idx += 1

            html_parts.append("        </div>")
            html_parts.append("    </div>")

        html_parts.extend(
            [
                "</body>",
                "</html>",
            ]
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

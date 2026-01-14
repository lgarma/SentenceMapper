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
        mask: np.ndarray | list[np.ndarray] | None = None,
        x_opt: float | list[float] | None = None,
        labels: str | list[str] | None = None,
        title: str = "Sentence Similarity vs. Length Ratio",
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> None:
        """Plot scatter plot of similarity vs. length ratio.

        Args:
            similarities: Array of cosine similarities
            ratios: Array of sentence-to-chunk length ratios
            mask: Optional binary mask(s) indicating selected sentences.
                  Can be a single mask or list of masks for multiple cuts.
            x_opt: Optional optimal parameter value(s) for title.
                   Can be a single value or list of values corresponding to masks.
            labels: Optional label(s) for each cut. Can be a single string or list of strings.
                    If not provided, defaults to percentage labels when x_opt is given.
            figsize: Figure size (default: (10, 6))
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=figsize)
        plt.style.use("seaborn-v0_8-poster")

        # Convert single mask/x_opt to lists for uniform handling
        if mask is not None:
            masks = [mask] if isinstance(mask, np.ndarray) else mask
            x_opts = None
            if x_opt is not None:
                x_opts = [x_opt] if isinstance(x_opt, (int, float)) else x_opt
                # Ensure x_opts matches masks length
                if len(x_opts) != len(masks):
                    raise ValueError(
                        "Number of x_opt values must match number of masks"
                    )

            # Handle labels
            if labels is not None:
                label_list = [labels] if isinstance(labels, str) else labels
                if len(label_list) != len(masks):
                    raise ValueError("Number of labels must match number of masks")
            else:
                label_list = None
        else:
            masks = None
            x_opts = None
            label_list = None

        if masks is not None:
            # Define colors for different masks
            colors = [
                "red",
                "blue",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
            ]

            # Initialize color assignment array (-1 means unassigned)
            point_colors = np.full(len(similarities), -1, dtype=int)

            # Compute sigmoid curves and assign colors to points below curves
            if x_opts is not None:
                sim_range = np.linspace(0, 1, 1000)
                for idx, x in enumerate(x_opts):
                    amplitude, midpoint, steepness = self.optimizer.get_params(x)

                    # For each point, check if it's below the sigmoid curve
                    threshold_ratios = self.optimizer.sigmoid(
                        similarities, amplitude, midpoint, steepness
                    )
                    below_curve = ratios <= threshold_ratios

                    # Assign color only if not already assigned (first curve wins)
                    unassigned = point_colors == -1
                    point_colors[below_curve & unassigned] = idx

            # Plot points by color
            # First, plot unassigned points (not below any curve)
            unassigned_mask = point_colors == -1
            if np.any(unassigned_mask):
                plt.scatter(
                    similarities[unassigned_mask],
                    ratios[unassigned_mask],
                    alpha=0.2,
                    color="gray",
                    label="Not Selected",
                    s=20,
                )

            # Then plot points colored by their assigned curve
            for idx in range(len(masks)):
                color = colors[idx % len(colors)]
                assigned_to_this = point_colors == idx

                if np.any(assigned_to_this):
                    plt.scatter(
                        similarities[assigned_to_this],
                        ratios[assigned_to_this],
                        color=color,
                        alpha=0.5,
                        s=20,
                    )

            # Plot sigmoid curves
            if x_opts is not None:
                sim_range = np.linspace(0, 1, 1000)
                for idx, x in enumerate(x_opts):
                    amplitude, midpoint, steepness = self.optimizer.get_params(x)
                    y = self.optimizer.sigmoid(
                        sim_range, amplitude, midpoint, steepness
                    )
                    color = colors[idx % len(colors)]

                    # Determine curve label
                    if label_list is not None:
                        curve_label = f"Threshold: {label_list[idx]}"
                    else:
                        curve_label = f"Threshold ({x:.1%})"

                    plt.plot(
                        sim_range,
                        y,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        label=curve_label,
                    )

        else:
            plt.scatter(similarities, ratios, alpha=0.5, s=20)

        plt.xlabel("Cosine Similarity to Chunk")
        plt.ylabel("Sentence to Chunk Length Ratio")
        plt.title(title)

        if masks is not None:
            plt.legend(frameon=False)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_three_regions(
        self,
        similarities: np.ndarray,
        ratios: np.ndarray,
        title: str = "Three Regions of Sentence Characteristics",
        figsize: tuple[int, int] = (12, 8),
        save_path: str | None = None,
    ) -> None:
        """Plot scatter plot highlighting the three characteristic regions.

        The three regions are:
        - Low similarity, low ratio: Transitional sentences (bottom-left)
        - High similarity, high ratio: Tables/lists/repetitive sections (top-right)
        - High similarity, low ratio: Information-dense sentences (bottom-right) ⭐

        Args:
            similarities: Array of cosine similarities
            ratios: Array of sentence-to-chunk length ratios
            title: Plot title (default: "Three Regions of Sentence Characteristics")
            figsize: Figure size (default: (12, 8))
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=figsize)
        plt.style.use("seaborn-v0_8-poster")

        # Define region boundaries (these are heuristic thresholds)
        sim_threshold = 0.6  # Boundary between low and high similarity
        ratio_threshold = 0.15  # Boundary between low and high ratio

        # Classify points into regions
        # Region 1: Low similarity, low ratio (transitional)
        region1 = (similarities < sim_threshold) & (ratios < ratio_threshold)

        # Region 2: High similarity, high ratio (tables/lists)
        region2 = (similarities >= sim_threshold) & (ratios >= ratio_threshold)

        # Region 3: High similarity, low ratio (information-dense) ⭐
        region3 = (similarities >= sim_threshold) & (ratios < ratio_threshold)

        # Other points (low similarity, high ratio - uncommon)
        other = (similarities < sim_threshold) & (ratios >= ratio_threshold)

        # Plot each region with distinct colors and labels
        if np.any(region1):
            plt.scatter(
                similarities[region1],
                ratios[region1],
                alpha=0.6,
                s=30,
                color="lightcoral",
                label="Transitional\n(Low Sim, Low Ratio)",
                edgecolors="darkred",
                linewidth=0.5,
            )

        if np.any(region2):
            plt.scatter(
                similarities[region2],
                ratios[region2],
                alpha=0.6,
                s=30,
                color="lightblue",
                label="Tables/Lists\n(High Sim, High Ratio)",
                edgecolors="darkblue",
                linewidth=0.5,
            )

        if np.any(region3):
            plt.scatter(
                similarities[region3],
                ratios[region3],
                alpha=0.6,
                s=30,
                color="lightgreen",
                label="Information-Dense\n(High Sim, Low Ratio)",
                edgecolors="darkgreen",
                linewidth=0.5,
            )

        if np.any(other):
            plt.scatter(
                similarities[other],
                ratios[other],
                alpha=0.3,
                s=20,
                color="gray",
                label="Other",
                edgecolors="black",
                linewidth=0.5,
            )

        # Add vertical and horizontal lines to show boundaries
        plt.axvline(
            x=sim_threshold,
            color="black",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label="Region Boundaries",
        )
        plt.axhline(
            y=ratio_threshold, color="black", linestyle="--", alpha=0.5, linewidth=1.5
        )

        # Add region annotations
        annotation_style = dict(
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
            ),
            fontsize=18,
            ha="center",
        )

        # Annotate each region at appropriate positions
        plt.text(
            0.25,
            0.05,
            "Short Transitional\nSentences",
            annotation_style,
            color="darkred",
        )
        plt.text(
            0.75,
            0.30,
            "Large Sentences\n(May contain good information,\nbut not in an efficient manner)",
            annotation_style,
            color="darkblue",
        )
        plt.text(
            0.75,
            0.05,
            "Information-Dense\nSentences",
            annotation_style,
            color="darkgreen",
            fontweight="bold",
        )

        plt.xlabel("Cosine Similarity to Chunk")
        plt.ylabel("Sentence to Chunk Length Ratio")
        plt.title(title)
        plt.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
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

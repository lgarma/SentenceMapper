"""Visualization utilities for sentence mapping analysis."""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from .powerlaw_optimizer import PowerLawOptimizer


class SentenceMapperVisualizer:
    """Visualization tools for sentence mapping results."""

    def __init__(
        self,
        slope: float = 1.0,
        initial_intercept: float = 0.0,
    ) -> None:
        """Initialize the visualizer.

        Args:
            slope: Fixed slope for power law optimizer
            initial_intercept: Initial intercept for power law optimizer
        """
        self.optimizer = PowerLawOptimizer(
            slope=slope,
            intercept=initial_intercept,
        )

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

            # Compute threshold curves and assign colors to points below curves
            if x_opts is not None:
                for idx, x in enumerate(x_opts):
                    amplitude, slope = self.optimizer.get_params(x)
                    ratio_clipped = np.maximum(ratios, self.optimizer.min_ratio)
                    threshold_similarities = self.optimizer.powerlaw(
                        ratio_clipped, amplitude, slope
                    )

                    above_curve = similarities >= threshold_similarities

                    # Assign color only if not already assigned (first curve wins)
                    unassigned = point_colors == -1
                    point_colors[above_curve & unassigned] = idx

            # Plot points by color
            # First, plot unassigned points (not above any curve)
            unassigned_mask = point_colors == -1
            if np.any(unassigned_mask):
                plt.scatter(
                    ratios[unassigned_mask],
                    similarities[unassigned_mask],
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
                        ratios[assigned_to_this],
                        similarities[assigned_to_this],
                        color=color,
                        alpha=0.5,
                        s=20,
                    )

            # Plot threshold curves
            if x_opts is not None:
                ratio_range = np.linspace(0.001, 1, 1000)
                for idx, x in enumerate(x_opts):
                    amplitude, slope = self.optimizer.get_params(x)
                    ratio_range_clipped = np.maximum(
                        ratio_range, self.optimizer.min_ratio
                    )
                    y = self.optimizer.powerlaw(ratio_range_clipped, amplitude, slope)

                    color = colors[idx % len(colors)]

                    # Determine curve label
                    if label_list is not None:
                        curve_label = f"Threshold: {label_list[idx]}"
                    else:
                        curve_label = f"Threshold ({x:.1%})"

                    plt.plot(
                        ratio_range,
                        y,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        label=curve_label,
                    )

        else:
            plt.scatter(ratios, similarities, alpha=0.5, s=20)

        plt.xlabel("Sentence to Chunk Length Ratio")
        plt.ylabel("Cosine Similarity to Chunk")
        plt.title(title)

        if masks is not None:
            plt.legend(frameon=False)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def display_highlighted_text(
        sentences: list[list[Any]],
        mask: np.ndarray,
        title: str = "Sentence Mapping Results",
        max_chunks: int | None = None,
        dark_mode: bool = False,
    ):
        """Display text with highlighted selected sentences in a Jupyter notebook.

        Args:
            sentences: List of sentence lists per chunk (from SentenceProcessor)
            mask: Binary mask indicating selected sentences (1 = selected, 0 = not selected)
            title: Title for the document (default: "Sentence Mapping Results")
            max_chunks: Maximum number of chunks to display (default: None = all chunks)
            dark_mode: Use dark theme styling (default: False)

        Returns:
            IPython.display.HTML object for notebook display
        """
        from IPython.display import HTML

        # Define theme colors
        if dark_mode:
            bg_color = "#1e1e1e"
            text_color = "#e0e0e0"
            title_color = "#4fc3f7"
            border_color = "#4fc3f7"
            chunk_bg = "#2d2d2d"
            chunk_border = "#555"
            chunk_header_color = "#9e9e9e"
            highlight_bg = "#ffd700"
            highlight_text = "#000"
        else:
            bg_color = "#fff"
            text_color = "#000"
            title_color = "#333"
            border_color = "#333"
            chunk_bg = "#f9f9f9"
            chunk_border = "#ddd"
            chunk_header_color = "#666"
            highlight_bg = "#ffff00"
            highlight_text = "#000"

        html_parts = [
            f"<div style='font-family: Arial, sans-serif; line-height: 1.6; background-color: {bg_color}; color: {text_color}; padding: 10px;'>",
            f"<h2 style='color: {title_color}; border-bottom: 2px solid {border_color}; padding-bottom: 10px;'>{title}</h2>",
        ]

        # Limit chunks if specified
        chunks_to_display = sentences[:max_chunks] if max_chunks else sentences

        idx = 0
        for chunk_idx, sentence_list in enumerate(chunks_to_display):
            html_parts.append(
                f"<div style='margin-bottom: 20px; padding: 15px; background-color: {chunk_bg}; border-left: 4px solid {chunk_border};'>"
            )
            html_parts.append(
                f"<div style='font-weight: bold; color: {chunk_header_color}; margin-bottom: 10px;'>Chunk {chunk_idx + 1}</div>"
            )
            html_parts.append("<div>")

            for sentence in sentence_list:
                sentence_text = sentence.text.strip()
                if mask[idx] == 1:
                    html_parts.append(
                        f"<span style='background-color: {highlight_bg}; color: {highlight_text}; padding: 2px 0;'>{sentence_text}</span> "
                    )
                else:
                    html_parts.append(f"<span>{sentence_text}</span> ")
                idx += 1

            html_parts.append("</div>")
            html_parts.append("</div>")

        # Skip remaining sentences if max_chunks was specified
        if max_chunks and len(sentences) > max_chunks:
            idx += sum(len(chunk) for chunk in sentences[max_chunks:])
            html_parts.append(
                f"<p style='color: #666; font-style: italic;'>... {len(sentences) - max_chunks} more chunks not displayed</p>"
            )

        html_parts.append("</div>")

        return HTML("\n".join(html_parts))

    @staticmethod
    def export_highlighted_text(
        sentences: list[list[Any]],
        mask: np.ndarray,
        output_path: str,
        title: str = "Sentence Mapping Results",
        dark_mode: bool = False,
    ) -> None:
        """Export text with highlighted selected sentences to an HTML file.

        Args:
            sentences: List of sentence lists per chunk (from SentenceProcessor)
            mask: Binary mask indicating selected sentences (1 = selected, 0 = not selected)
            output_path: Path to save the output file (should end with .html or .md)
            title: Title for the document (default: "Sentence Mapping Results")
            dark_mode: Use dark theme styling (default: False)
        """
        # Define theme colors and styles
        if dark_mode:
            body_bg = "#1e1e1e"
            body_color = "#e0e0e0"
            h1_color = "#4fc3f7"
            h1_border = "#4fc3f7"
            highlight_bg = "#ffd700"
            highlight_color = "#000"
            chunk_bg = "#2d2d2d"
            chunk_border = "#555"
            chunk_header_color = "#9e9e9e"
        else:
            body_bg = "#fff"
            body_color = "#000"
            h1_color = "#333"
            h1_border = "#333"
            highlight_bg = "#ffff00"
            highlight_color = "#000"
            chunk_bg = "#f9f9f9"
            chunk_border = "#ddd"
            chunk_header_color = "#666"

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <meta charset='UTF-8'>",
            f"    <title>{title}</title>",
            "    <style>",
            f"        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; background-color: {body_bg}; color: {body_color}; }}",
            f"        h1 {{ color: {h1_color}; border-bottom: 2px solid {h1_border}; padding-bottom: 10px; }}",
            f"        .highlight {{ background-color: {highlight_bg}; color: {highlight_color}; padding: 2px 0; }}",
            "        .sentence { display: inline; }",
            f"        .chunk {{ margin-bottom: 20px; padding: 15px; background-color: {chunk_bg}; border-left: 4px solid {chunk_border}; }}",
            f"        .chunk-header {{ font-weight: bold; color: {chunk_header_color}; margin-bottom: 10px; }}",
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

    def plot_with_frontier(
        self,
        similarities: np.ndarray,
        ratios: np.ndarray,
        slope: float,
        intercept: float,
        info: dict,
        mask: np.ndarray | None = None,
        x_opt: float | list[float] | None = None,
        labels: str | list[str] | None = None,
        title: str = "Similarity vs Ratio with Frontier Curve",
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> None:
        """Plot similarity vs ratio with fitted frontier curve in log-log space.

        This unified visualization combines residual-based coloring with multiple
        threshold curves. Points are colored by their distance from the frontier
        (residual), while multiple threshold curves can be overlaid with distinct colors.

        Args:
            similarities: Array of cosine similarities
            ratios: Array of sentence-to-chunk length ratios
            slope: Slope of the fitted frontier line
            intercept: Intercept of the fitted frontier line
            info: Dictionary with fit information
            mask: Optional binary mask indicating selected sentences (deprecated, not used)
            x_opt: Optional optimal parameter value(s) for threshold curves.
                   Can be a single value or list of values for multiple thresholds.
            labels: Optional label(s) for each threshold. Can be a single string or list of strings.
                    If not provided, defaults to percentage labels when x_opt is given.
            title: Plot title
            figsize: Figure size (default: (10, 6))
            save_path: Optional path to save the figure
        """
        # Create plot
        plt.figure(figsize=figsize)
        plt.style.use("seaborn-v0_8-poster")

        # Filter valid points for log scale
        valid_mask = (similarities > 0) & (ratios > 0)
        sim_plot = similarities[valid_mask]
        ratio_plot = ratios[valid_mask]

        # Calculate residuals (distance above frontier)
        log_ratio = np.log10(ratio_plot)
        log_sim = np.log10(sim_plot)
        expected_log_sim = slope * log_ratio + intercept
        residuals = log_sim - expected_log_sim  # Positive = above frontier

        # Separate points: above frontier (colored by residual) vs below frontier (grey)
        above_frontier = residuals >= 0

        # Plot points below the frontier in grey
        if np.any(~above_frontier):
            plt.scatter(
                ratio_plot[~above_frontier],
                sim_plot[~above_frontier],
                color="lightgrey",
                alpha=0.3,
                s=20,
                label=None,  # "Below Frontier",
            )

        # Plot points above the frontier, colored by residual
        if np.any(above_frontier):
            scatter = plt.scatter(
                ratio_plot[above_frontier],
                sim_plot[above_frontier],
                c=residuals[above_frontier],
                cmap="RdYlGn",  # Red = at frontier, Green = well above
                alpha=0.6,
                s=20,
            )
            plt.colorbar(scatter, label="Residual (log scale)")

        # Plot frontier line
        ratio_range = np.logspace(
            np.log10(ratio_plot.min()), np.log10(ratio_plot.max()), 100
        )
        sim_frontier = 10 ** (slope * np.log10(ratio_range) + intercept)
        plt.plot(
            ratio_range,
            sim_frontier,
            "b--",
            linewidth=2,
            label="Sentence Frontier",
        )

        # Plot frontier points if available
        if info.get("frontier_ratio") is not None:
            frontier_ratio_orig = 10 ** info["frontier_ratio"]
            frontier_sim_orig = 10 ** info["frontier_sim"]
            plt.scatter(
                frontier_ratio_orig,
                frontier_sim_orig,
                color="blue",
                s=50,
                marker="x",
                label="Frontier Points",
                zorder=5,
            )

        # Plot threshold curves if provided
        if x_opt is not None:
            # Convert to list for uniform handling
            x_opts = [x_opt] if isinstance(x_opt, (int, float)) else x_opt

            # Handle labels
            if labels is not None:
                label_list = [labels] if isinstance(labels, str) else labels
                if len(label_list) != len(x_opts):
                    raise ValueError(
                        "Number of labels must match number of x_opt values"
                    )
            else:
                label_list = None

            # Define colors for different threshold curves
            colors = [
                "red",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
                "cyan",
            ]

            # Plot each threshold curve
            ratio_curve_range = np.linspace(0.001, 1, 1000)
            for idx, x in enumerate(x_opts):
                amplitude, curve_slope = self.optimizer.get_params(x)
                ratio_range_clipped = np.maximum(
                    ratio_curve_range, self.optimizer.min_ratio
                )
                y = self.optimizer.powerlaw(ratio_range_clipped, amplitude, curve_slope)

                color = colors[idx % len(colors)]

                # Filter out zeros for log scale
                valid_curve = y > 0
                ratio_curve = ratio_curve_range[valid_curve]
                y_curve = y[valid_curve]

                # Determine curve label
                if label_list is not None:
                    curve_label = f"{label_list[idx]}"
                else:
                    curve_label = f"Threshold ({x:.1%})"

                plt.plot(
                    ratio_curve,
                    y_curve,
                    color=color,
                    linewidth=2,
                    linestyle="-",
                    label=curve_label,
                )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(np.min(ratio_plot) * 0.9, np.max(ratio_plot) * 1.1)
        plt.ylim(np.min(sim_plot) * 0.9, np.max(sim_plot) * 1.1)
        plt.xlabel("Sentence to Chunk Length Ratio (log scale)")
        plt.ylabel("Cosine Similarity to Chunk (log scale)")
        plt.title(f"{title}")
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_with_frontier_interactive(
        self,
        similarities: np.ndarray,
        ratios: np.ndarray,
        slope: float,
        intercept: float,
        info: dict,
        sentences: list[str] | None = None,
        x_opt: float | list[float] | None = None,
        labels: str | list[str] | None = None,
        title: str = "Similarity vs Ratio with Frontier Curve",
        figsize: tuple[int, int] = (900, 600),
        save_path: str | None = None,
    ) -> go.Figure:
        """Plot interactive similarity vs ratio with fitted frontier curve using Plotly.

        This interactive visualization allows hovering over points to see their
        residual, sentence text, and coordinates. Points are colored by their
        distance from the frontier (residual).

        Args:
            similarities: Array of cosine similarities
            ratios: Array of sentence-to-chunk length ratios
            slope: Slope of the fitted frontier line
            intercept: Intercept of the fitted frontier line
            info: Dictionary with fit information
            sentences: Optional list of sentence texts for hover display
            x_opt: Optional optimal parameter value(s) for threshold curves.
                   Can be a single value or list of values for multiple thresholds.
            labels: Optional label(s) for each threshold. Can be a single string or list of strings.
                    If not provided, defaults to percentage labels when x_opt is given.
            title: Plot title
            figsize: Figure size as (width, height) in pixels (default: (900, 600))
            save_path: Optional path to save the figure (HTML format)

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Filter valid points for log scale
        valid_mask = (similarities > 0) & (ratios > 0)
        sim_plot = similarities[valid_mask]
        ratio_plot = ratios[valid_mask]

        # Filter sentences if provided
        if sentences is not None:
            sentences_plot = [s for s, v in zip(sentences, valid_mask) if v]
        else:
            sentences_plot = [f"Sentence {i}" for i in range(len(sim_plot))]

        # Calculate residuals (distance above frontier)
        log_ratio = np.log10(ratio_plot)
        log_sim = np.log10(sim_plot)
        expected_log_sim = slope * log_ratio + intercept
        residuals = log_sim - expected_log_sim  # Positive = above frontier

        # Separate points: above frontier vs below frontier
        above_frontier = residuals >= 0

        # Create hover text
        def make_hover_text(ratio, sim, residual, sentence):
            # Truncate sentence for display
            truncated = sentence[:100] + "..." if len(sentence) > 100 else sentence
            return (
                f"{truncated}<br>"
                f"<b>Ratio:</b> {ratio:.4f}<br>"
                f"<b>Similarity:</b> {sim:.4f}<br>"
                f"<b>Residual:</b> {residual:.4f}"
            )

        # Plot points below the frontier in grey
        if np.any(~above_frontier):
            hover_texts_below = [
                make_hover_text(r, s, res, sent)
                for r, s, res, sent in zip(
                    ratio_plot[~above_frontier],
                    sim_plot[~above_frontier],
                    residuals[~above_frontier],
                    [
                        sent
                        for sent, below in zip(sentences_plot, ~above_frontier)
                        if below
                    ],
                )
            ]
            fig.add_trace(
                go.Scatter(
                    x=ratio_plot[~above_frontier],
                    y=sim_plot[~above_frontier],
                    mode="markers",
                    marker=dict(color="lightgrey", size=8, opacity=0.5),
                    name="Below Frontier",
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_texts_below,
                )
            )

        # Plot points above the frontier, colored by residual
        if np.any(above_frontier):
            hover_texts_above = [
                make_hover_text(r, s, res, sent)
                for r, s, res, sent in zip(
                    ratio_plot[above_frontier],
                    sim_plot[above_frontier],
                    residuals[above_frontier],
                    [
                        sent
                        for sent, above in zip(sentences_plot, above_frontier)
                        if above
                    ],
                )
            ]
            fig.add_trace(
                go.Scatter(
                    x=ratio_plot[above_frontier],
                    y=sim_plot[above_frontier],
                    mode="markers",
                    marker=dict(
                        color=residuals[above_frontier],
                        colorscale="RdYlGn",
                        size=8,
                        opacity=0.7,
                        colorbar=dict(title="Residual<br>(log scale)"),
                    ),
                    name="Above Frontier",
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_texts_above,
                )
            )

        # Plot frontier line
        ratio_range = np.logspace(
            np.log10(ratio_plot.min()), np.log10(ratio_plot.max()), 100
        )
        sim_frontier = 10 ** (slope * np.log10(ratio_range) + intercept)
        fig.add_trace(
            go.Scatter(
                x=ratio_range,
                y=sim_frontier,
                mode="lines",
                line=dict(color="blue", width=2, dash="dash"),
                name="Sentence Frontier",
                hoverinfo="skip",
            )
        )

        # Plot frontier points if available
        if info.get("frontier_ratio") is not None:
            frontier_ratio_orig = 10 ** info["frontier_ratio"]
            frontier_sim_orig = 10 ** info["frontier_sim"]

            # Find nearest sentences to each frontier point for hover display
            frontier_hover_texts = []
            for f_ratio, f_sim in zip(frontier_ratio_orig, frontier_sim_orig):
                # Calculate distance to all points in log space for better matching
                distances = np.sqrt(
                    (np.log10(ratio_plot) - np.log10(f_ratio)) ** 2
                    + (np.log10(sim_plot) - np.log10(f_sim)) ** 2
                )
                nearest_idx = np.argmin(distances)
                nearest_sentence = sentences_plot[nearest_idx]
                truncated = (
                    nearest_sentence[:100] + "..."
                    if len(nearest_sentence) > 100
                    else nearest_sentence
                )
                frontier_hover_texts.append(
                    f"<b>Frontier Point</b><br>{truncated}<br>"
                    f"<b>Ratio:</b> {f_ratio:.4f}<br>"
                    f"<b>Similarity:</b> {f_sim:.4f}"
                )

            fig.add_trace(
                go.Scatter(
                    x=frontier_ratio_orig,
                    y=frontier_sim_orig,
                    mode="markers",
                    marker=dict(color="blue", size=10, symbol="x"),
                    name="Frontier Points",
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=frontier_hover_texts,
                )
            )

        # Plot threshold curves if provided
        if x_opt is not None:
            x_opts = [x_opt] if isinstance(x_opt, (int, float)) else x_opt

            if labels is not None:
                label_list = [labels] if isinstance(labels, str) else labels
                if len(label_list) != len(x_opts):
                    raise ValueError(
                        "Number of labels must match number of x_opt values"
                    )
            else:
                label_list = None

            colors = [
                "red",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
                "cyan",
            ]

            ratio_curve_range = np.linspace(0.001, 1, 1000)
            for idx, x in enumerate(x_opts):
                amplitude, curve_slope = self.optimizer.get_params(x)
                ratio_range_clipped = np.maximum(
                    ratio_curve_range, self.optimizer.min_ratio
                )
                y = self.optimizer.powerlaw(ratio_range_clipped, amplitude, curve_slope)

                color = colors[idx % len(colors)]

                valid_curve = y > 0
                ratio_curve = ratio_curve_range[valid_curve]
                y_curve = y[valid_curve]

                if label_list is not None:
                    curve_label = f"{label_list[idx]}"
                else:
                    curve_label = f"Threshold ({x:.1%})"

                fig.add_trace(
                    go.Scatter(
                        x=ratio_curve,
                        y=y_curve,
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=curve_label,
                        hoverinfo="skip",
                    )
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Sentence to Chunk Length Ratio (log scale)",
                type="log",
                range=[
                    np.log10(ratio_plot.min() * 0.9),
                    np.log10(ratio_plot.max() * 1.1),
                ],
                gridcolor="rgba(128, 128, 128, 0.3)",
            ),
            yaxis=dict(
                title="Cosine Similarity to Chunk (log scale)",
                type="log",
                range=[
                    np.log10(sim_plot.min() * 0.9),
                    np.log10(sim_plot.max() * 1.1),
                ],
                gridcolor="rgba(128, 128, 128, 0.3)",
            ),
            width=figsize[0],
            height=figsize[1],
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
            hovermode="closest",
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()
        return fig

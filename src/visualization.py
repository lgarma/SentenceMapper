"""Visualization utilities for sentence mapping analysis."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class SentenceMapperVisualizer:
    """Visualization tools for sentence mapping results."""

    def __init__(self) -> None:
        """Initialize the visualizer."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _greedy_select(
        scores: np.ndarray,
        tokens: np.ndarray,
        budget: float,
    ) -> np.ndarray:
        """Greedy token-budget fill by descending score."""
        mask = np.zeros(len(scores), dtype=bool)
        current = 0
        for idx in np.argsort(-scores):
            if current + tokens[idx] > budget:
                continue
            mask[idx] = True
            current += tokens[idx]
        return mask

    # ------------------------------------------------------------------
    # Matplotlib: baseline vs length-biased
    # ------------------------------------------------------------------

    @staticmethod
    def plot_similarity_vs_ratio(
        similarities: np.ndarray,
        ratios: np.ndarray,
        tokens: np.ndarray,
        objective_percentage: float = 0.3,
        length_bias: float = 0.5,
        title: str | None = None,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
    ) -> None:
        """Scatter plot comparing baseline (α=0) vs length-biased selection.

        Points are coloured by category:
        - **grey**: not selected by either method
        - **blue**: selected by both baseline and biased
        - **red**: selected only by baseline (α=0)
        - **green**: selected only by length-biased (α>0)

        An additive iso-score threshold line is drawn for the biased
        selection (the best rejected sentence's score).

        Args:
            similarities: Cosine similarities (sentence vs context)
            ratios: Length ratios per sentence
            tokens: Token counts per sentence
            objective_percentage: Target fraction of tokens to keep (0–1)
            length_bias: α for the additive formula (default: 0.5)
            title: Plot title.  If *None*, auto-generated.
            figsize: Figure size (default: (10, 6))
            save_path: Optional path to save the figure
        """
        total_tokens = int(np.sum(tokens))
        budget = total_tokens * objective_percentage

        scores_baseline = similarities.copy()  # α = 0
        scores_biased = similarities - length_bias * ratios

        mask_base = SentenceMapperVisualizer._greedy_select(
            scores_baseline,
            tokens,
            budget,
        )
        mask_bias = SentenceMapperVisualizer._greedy_select(
            scores_biased,
            tokens,
            budget,
        )

        both = mask_base & mask_bias
        only_base = mask_base & ~mask_bias
        only_bias = mask_bias & ~mask_base
        neither = ~mask_base & ~mask_bias

        with plt.style.context("seaborn-v0_8-poster"):
            fig, ax = plt.subplots(figsize=figsize)

            ax.scatter(
                ratios[neither],
                similarities[neither],
                color="lightgrey",
                alpha=0.3,
                label="Not selected",
            )
            ax.scatter(
                ratios[both],
                similarities[both],
                color="tab:blue",
                alpha=0.6,
                label=f"Both ({both.sum()})",
            )
            ax.scatter(
                ratios[only_base],
                similarities[only_base],
                color="tab:red",
                alpha=0.7,
                marker="x",
                label=f"Only similarity ({only_base.sum()})",
            )
            ax.scatter(
                ratios[only_bias],
                similarities[only_bias],
                color="tab:green",
                alpha=0.7,
                label=f"Only α={length_bias} ({only_bias.sum()})",
            )

            # Threshold line for biased selection
            non_selected = scores_biased[~mask_bias]
            if non_selected.size > 0:
                threshold = float(np.max(non_selected))
                r_grid = np.linspace(ratios.min(), ratios.max(), 200)
                ax.plot(
                    r_grid,
                    threshold + length_bias * r_grid,
                    "k--",
                    alpha=0.3,
                    label="Threshold",
                )

            ax.set_xlabel("Ratio")
            ax.set_ylabel("Similarity")

            # Set axis limits using percentiles to avoid outliers compressing the plot
            ratio_min = np.percentile(ratios, 1)
            ratio_max = np.percentile(ratios, 99)
            ratio_margin = (ratio_max - ratio_min) * 0.05
            ax.set_xlim(ratio_min - ratio_margin, ratio_max + ratio_margin)

            sim_min = np.percentile(similarities, 1)
            sim_max = np.percentile(similarities, 99)
            sim_margin = (sim_max - sim_min) * 0.05
            ax.set_ylim(sim_min - sim_margin, sim_max + sim_margin)

            if title is None:
                jaccard = both.sum() / max(
                    both.sum() + only_base.sum() + only_bias.sum(), 1
                )
                title = (
                    f"Baseline (α=0) vs Additive α={length_bias} "
                    f"— {objective_percentage:.0%} compression, "
                    f"Jaccard={jaccard:.3f}"
                )
            ax.set_title(title)
            ax.legend(loc="lower right", frameon=False)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_similarity_vs_ratio_multiple_alphas(
        similarities: np.ndarray,
        ratios: np.ndarray,
        tokens: np.ndarray,
        objective_percentage: float = 0.3,
        alphas: list[float] | None = None,
        title: str | None = None,
        figsize: tuple[int, int] = (14, 12),
        save_path: str | None = None,
    ) -> plt.Figure:
        """2×2 grid comparing different alpha values at a fixed compression level.

        Each subplot shows the sentence selection for a different alpha value,
        with overlap analysis against the baseline (α=0).

        Colour scheme per subplot:
        - **grey**: not selected by any method
        - **blue**: selected by both baseline and current alpha
        - **red**: selected only by baseline (α=0)
        - **green**: selected only by current alpha

        Args:
            similarities: Cosine similarities (sentence vs context)
            ratios: Length ratios per sentence
            tokens: Token counts per sentence
            objective_percentage: Target fraction of tokens to keep (0–1)
            alphas: List of alpha values to compare (default: [0.0, 0.5, 1.0, 2.0])
            title: Figure title. If None, auto-generated.
            figsize: Figure size (default: (14, 12))
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure
        """
        if alphas is None:
            alphas = [0.0, 0.5, 1.0, 2.0]

        total_tokens = int(np.sum(tokens))
        budget = total_tokens * objective_percentage

        with plt.style.context("seaborn-v0_8-poster"):
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes_flat = axes.flatten()

            for idx, (alpha, ax) in enumerate(zip(alphas, axes_flat)):
                # Compute scores and selection for this alpha
                scores_baseline = similarities.copy()
                scores_biased = similarities - alpha * ratios

                # Greedy selection
                mask_base = SentenceMapperVisualizer._greedy_select(
                    scores_baseline, tokens, budget
                )
                mask_bias = SentenceMapperVisualizer._greedy_select(
                    scores_biased, tokens, budget
                )

                both = mask_base & mask_bias
                only_base = mask_base & ~mask_bias
                only_bias = mask_bias & ~mask_base
                neither = ~mask_base & ~mask_bias

                # Plot
                ax.scatter(
                    ratios[neither],
                    similarities[neither],
                    color="lightgrey",
                    alpha=0.7,
                    s=50,
                    label="Not selected",
                )
                ax.scatter(
                    ratios[both],
                    similarities[both],
                    color="tab:blue",
                    alpha=0.7,
                    label=f"Both ({both.sum()})",
                )
                ax.scatter(
                    ratios[only_base],
                    similarities[only_base],
                    color="tab:red",
                    alpha=0.7,
                    marker="x",
                    label=f"Only baseline ({only_base.sum()})",
                )
                ax.scatter(
                    ratios[only_bias],
                    similarities[only_bias],
                    color="tab:green",
                    alpha=0.7,
                    label=f"Only α={alpha} ({only_bias.sum()})",
                )

                # Threshold line
                non_selected = scores_biased[~mask_bias]
                if non_selected.size > 0:
                    threshold = float(np.max(non_selected))
                    r_grid = np.linspace(ratios.min(), ratios.max(), 200)
                    ax.plot(
                        r_grid,
                        threshold + alpha * r_grid,
                        "k--",
                        alpha=0.3,
                        label="Threshold",
                    )

                # Set axis limits using percentiles
                ratio_min = np.percentile(ratios, 1)
                ratio_max = np.percentile(ratios, 99)
                ratio_margin = (ratio_max - ratio_min) * 0.05
                ax.set_xlim(ratio_min - ratio_margin, ratio_max + ratio_margin)

                sim_min = np.percentile(similarities, 1)
                sim_max = np.percentile(similarities, 99)
                sim_margin = (sim_max - sim_min) * 0.05
                ax.set_ylim(sim_min - sim_margin, sim_max + sim_margin)

                # Labels and styling
                ax.set_xlabel("Ratio")
                ax.set_ylabel("Similarity")
                jaccard = both.sum() / max(
                    both.sum() + only_base.sum() + only_bias.sum(), 1
                )
                ax.set_title(f"α = {alpha} (Jaccard = {jaccard:.3f})")
                ax.legend(loc="lower right", frameon=False)
                ax.grid(True, alpha=0.2)

            if title is None:
                title = (
                    f"Length Bias Comparison at {objective_percentage:.0%} Compression"
                )
            fig.suptitle(title, y=0.995)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_similarity_vs_ratio_multiple_objectives(
        similarities: np.ndarray,
        ratios: np.ndarray,
        results: list[dict],
        objective_percentages: list[float],
        colors: list[str] | None = None,
        title: str = "Sentence Selection at Different Compression Levels",
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
        show_thresholds: bool = True,
    ) -> None:
        """Plot selections at multiple compression levels.

        Shows how different objective percentages select different sets of
        sentences, with smaller percentages plotted on top for visibility.

        Args:
            similarities: Cosine similarities (sentence vs context)
            ratios: Length ratios per sentence
            results: List of result dicts, each containing 'mask', 'selected_tokens',
                and 'length_bias'
            objective_percentages: List of objective percentages (e.g., [0.1, 0.3, 0.5])
            colors: Colors for each level. If None, generates colors using a colormap:
                - 1 value: red
                - 2 values: red, green
                - 3 values: red, orange, green
                - 4+ values: uses RdYlGn colormap
            title: Plot title
            figsize: Figure size (default: (10, 6))
            save_path: Optional path to save the figure
            show_thresholds: Whether to show threshold lines (default: True)

        """
        if colors is None:
            n = len(objective_percentages)
            if n == 1:
                colors = ["red"]
            elif n == 2:
                colors = ["red", "green"]
            elif n == 3:
                colors = ["red", "orange", "green"]
            else:
                # Use RdYlGn colormap for 4+ values
                cmap = plt.cm.RdYlGn
                colors = [cmap(i / (n - 1)) for i in range(n)]

        with plt.style.context("seaborn-v0_8-poster"):
            fig, ax = plt.subplots(figsize=figsize)

            # Plot unselected (not in any mask)
            any_selected = np.zeros(len(similarities), dtype=bool)
            for r in results:
                any_selected |= r["mask"].astype(bool)

            ax.scatter(
                ratios[~any_selected],
                similarities[~any_selected],
                color="lightgrey",
                alpha=0.7,
                s=50,
                label="Not selected",
            )

            # Plot each selection level in REVERSE order so smallest appears on top
            for pct, r, color in reversed(
                list(zip(objective_percentages, results, colors))
            ):
                mask = r["mask"].astype(bool)
                ax.scatter(
                    ratios[mask],
                    similarities[mask],
                    color=color,
                    alpha=0.7,
                    label=f"{pct * 100:.0f}%: {r['selected_tokens']} tokens",
                )

            # Draw threshold lines for each level
            if show_thresholds and len(results) > 0:
                # Get length_bias from first result (assumed same for all)
                length_bias = results[0].get("length_bias", 0.5)
                r_grid = np.linspace(float(ratios.min()), float(ratios.max()), 200)

                for pct, r, color in zip(objective_percentages, results, colors):
                    mask = r["mask"].astype(bool)
                    scores = similarities - length_bias * ratios
                    non_selected_scores = scores[~mask]

                    if non_selected_scores.size > 0:
                        threshold = float(np.max(non_selected_scores))
                        s_line = threshold + length_bias * r_grid
                        ax.plot(r_grid, s_line, "--", color=color, alpha=0.4)

            ax.set_xlabel("Ratio")
            ax.set_ylabel("Similarity")

            # Set axis limits using percentiles to avoid outliers compressing the plot
            ratio_min = np.percentile(ratios, 1)
            ratio_max = np.percentile(ratios, 99)
            ratio_margin = (ratio_max - ratio_min) * 0.05
            ax.set_xlim(ratio_min - ratio_margin, ratio_max + ratio_margin)

            sim_min = np.percentile(similarities, 1)
            sim_max = np.percentile(similarities, 99)
            sim_margin = (sim_max - sim_min) * 0.05
            ax.set_ylim(sim_min - sim_margin, sim_max + sim_margin)

            ax.set_title(title)
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

    # ------------------------------------------------------------------
    # Plotly: baseline vs length-biased (interactive with hover)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_similarity_vs_ratio_interactive(
        similarities: np.ndarray,
        ratios: np.ndarray,
        tokens: np.ndarray,
        sentences: list[str] | np.ndarray | None = None,
        objective_percentage: float = 0.3,
        length_bias: float = 0.5,
        title: str | None = None,
        figsize: tuple[int, int] = (900, 600),
        save_path: str | None = None,
        show: bool = True,
    ) -> go.Figure:
        """Interactive scatter comparing baseline vs length-biased selection.

        Hover shows sentence text, ratio, similarity and score.

        Colour scheme:
        - **grey**: not selected by either
        - **blue**: selected by both
        - **red**: only baseline (α=0)
        - **green**: only length-biased

        Args:
            similarities: Cosine similarities
            ratios: Length ratios
            tokens: Token counts per sentence
            sentences: Optional sentence texts for hover
            objective_percentage: Target token fraction (0–1)
            length_bias: α for additive formula (default: 0.5)
            title: Plot title.  Auto-generated if *None*.
            figsize: (width, height) in pixels
            save_path: Optional HTML output path
            show: Whether to call ``fig.show()``

        Returns:
            Plotly Figure
        """
        total_tokens = int(np.sum(tokens))
        budget = total_tokens * objective_percentage

        scores_baseline = similarities.copy()
        scores_biased = similarities - length_bias * ratios

        mask_base = SentenceMapperVisualizer._greedy_select(
            scores_baseline,
            tokens,
            budget,
        )
        mask_bias = SentenceMapperVisualizer._greedy_select(
            scores_biased,
            tokens,
            budget,
        )

        both = mask_base & mask_bias
        only_base = mask_base & ~mask_bias
        only_bias = mask_bias & ~mask_base
        neither = ~mask_base & ~mask_bias

        if sentences is None:
            sentence_list = [f"Sentence {i}" for i in range(len(similarities))]
        else:
            sentence_list = list(sentences)

        def _hover(indices: np.ndarray, category: str) -> list[str]:
            texts = []
            for i in indices:
                s = sentence_list[i]
                # Show up to 3 lines of 88 characters each
                max_chars = 88 * 3
                if len(s) > max_chars:
                    s_trunc = s[:max_chars] + "..."
                else:
                    s_trunc = s

                # Break into lines of 88 chars
                lines = []
                for line_start in range(0, len(s_trunc), 88):
                    lines.append(s_trunc[line_start : line_start + 88])

                sentence_display = "<br>".join(lines)

                texts.append(
                    f"{sentence_display}<br>"
                    f"<b>Ratio:</b> {ratios[i]:.4f}<br>"
                    f"<b>Similarity:</b> {similarities[i]:.4f}<br>"
                    f"<b>Score (α={length_bias}):</b> {scores_biased[i]:.4f}"
                )
            return texts

        fig = go.Figure()

        groups = [
            (neither, "Not selected", "lightgrey", 0.3, 6, "circle"),
            (both, f"Both ({both.sum()})", "#1f77b4", 0.7, 8, "circle"),
            (only_base, f"Only similarity ({only_base.sum()})", "#d62728", 0.8, 9, "x"),
            (
                only_bias,
                f"Only α={length_bias} ({only_bias.sum()})",
                "#2ca02c",
                0.8,
                9,
                "circle",
            ),
        ]

        for mask, name, color, opacity, size, symbol in groups:
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=ratios[mask],
                    y=similarities[mask],
                    mode="markers",
                    marker=dict(color=color, size=size, opacity=opacity, symbol=symbol),
                    name=name,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=_hover(idxs, name),
                )
            )

        # Threshold line
        non_selected = scores_biased[~mask_bias]
        if non_selected.size > 0:
            threshold = float(np.max(non_selected))
            r_grid = np.linspace(float(ratios.min()), float(ratios.max()), 200)
            s_line = threshold + length_bias * r_grid
            fig.add_trace(
                go.Scatter(
                    x=r_grid,
                    y=s_line,
                    mode="lines",
                    line=dict(color="black", width=1.5, dash="dash"),
                    name="Threshold",
                    hoverinfo="skip",
                )
            )

        if title is None:
            jaccard = both.sum() / max(
                both.sum() + only_base.sum() + only_bias.sum(), 1
            )
            title = (
                f"Baseline (α=0) vs Additive α={length_bias} "
                f"— {objective_percentage:.0%} compression, "
                f"Jaccard={jaccard:.3f}"
            )

        # Set axis ranges using percentiles to avoid outliers compressing the plot
        ratio_min = float(np.percentile(ratios, 1))
        ratio_max = float(np.percentile(ratios, 99))
        ratio_margin = (ratio_max - ratio_min) * 0.05

        sim_min = float(np.percentile(similarities, 1))
        sim_max = float(np.percentile(similarities, 99))
        sim_margin = (sim_max - sim_min) * 0.05

        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Ratio",
                gridcolor="rgba(128,128,128,0.3)",
                range=[ratio_min - ratio_margin, ratio_max + ratio_margin],
            ),
            yaxis=dict(
                title="Similarity",
                gridcolor="rgba(128,128,128,0.3)",
                range=[sim_min - sim_margin, sim_max + sim_margin],
            ),
            width=figsize[0],
            height=figsize[1],
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            hovermode="closest",
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        return fig

    @staticmethod
    def display_highlighted_text(
        sentences: list[str],
        mask: np.ndarray,
        title: str = "Sentence Mapping Results",
        dark_mode: bool = False,
        max_display_tokens: int | None = None,
    ):
        """Display text with highlighted selected sentences in a Jupyter notebook.

        Args:
            sentences: List of sentence strings
            mask: Binary mask indicating selected sentences (1 = selected, 0 = not selected)
            title: Title for the document (default: "Sentence Mapping Results")
            dark_mode: Use dark theme styling (default: False)
            max_display_tokens: Optional maximum number of whitespace-delimited
                tokens to render. If None, render all tokens.

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
            highlight_bg = "#ffd700"
            highlight_text = "#000"
        else:
            bg_color = "#fff"
            text_color = "#000"
            title_color = "#333"
            border_color = "#333"
            highlight_bg = "#ffff00"
            highlight_text = "#000"

        html_parts = [
            f"<div style='font-family: Arial, sans-serif; line-height: 1.6; background-color: {bg_color}; color: {text_color}; padding: 10px;'>",
            f"<h2 style='color: {title_color}; border-bottom: 2px solid {border_color}; padding-bottom: 10px;'>{title}</h2>",
            "<div>",
        ]

        displayed_tokens = 0
        was_truncated = False

        for idx, sentence in enumerate(sentences):
            sentence_text = sentence.strip()

            if max_display_tokens is not None:
                remaining = max_display_tokens - displayed_tokens
                if remaining <= 0:
                    was_truncated = True
                    break

                sentence_tokens = sentence_text.split()
                if len(sentence_tokens) > remaining:
                    sentence_text = " ".join(sentence_tokens[:remaining]) + " ..."
                    displayed_tokens += remaining
                    was_truncated = True
                else:
                    displayed_tokens += len(sentence_tokens)

            if mask[idx] == 1:
                html_parts.append(
                    f"<span style='background-color: {highlight_bg}; color: {highlight_text}; padding: 2px 0;'>{sentence_text}</span> "
                )
            else:
                html_parts.append(f"<span>{sentence_text}</span> ")

            if was_truncated:
                break

        if was_truncated and max_display_tokens is not None:
            html_parts.append(
                "<p style='margin-top: 12px; opacity: 0.8;'><em>Display truncated at "
                f"{max_display_tokens} tokens.</em></p>"
            )

        html_parts.append("</div>")
        html_parts.append("</div>")

        return HTML("\n".join(html_parts))

    @staticmethod
    def export_highlighted_text(
        sentences: list[str],
        mask: np.ndarray,
        output_path: str,
        title: str = "Sentence Mapping Results",
        dark_mode: bool = False,
    ) -> None:
        """Export text with highlighted selected sentences to an HTML file.

        Args:
            sentences: List of sentence strings
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

        html_parts.append("        <div>")

        for idx, sentence in enumerate(sentences):
            sentence_text = sentence.strip()
            if mask[idx] == 1:
                html_parts.append(
                    f"            <span class='sentence highlight'>{sentence_text}</span> "
                )
            else:
                html_parts.append(
                    f"            <span class='sentence'>{sentence_text}</span> "
                )

        html_parts.append("        </div>")

        html_parts.extend(
            [
                "</body>",
                "</html>",
            ]
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

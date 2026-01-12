"""Sigmoid-based sentence optimizer for filtering sentences based on similarity and length ratio."""

import numpy as np
from scipy.optimize import bisect
from typing import Literal


class SigmoidOptimizer:
    """Class to optimize sigmoid parameters for filtering sentences based on similarity and length ratio."""

    def __init__(
        self,
        strategy: Literal[
            "balanced", "short_sentences", "high_similarity"
        ] = "balanced",
    ):
        """Initialize the optimizer.

        Args:
            steepness: Controls the steepness of the sigmoid curve (default: 10.0)
                      Higher values = sharper transition
            strategy: Selection strategy (default: "balanced")
                - "balanced": Default behavior, balances similarity and length
                - "short_sentences": Prefers shorter sentences (amplitude grows slowly, midpoint shifts quickly)
                - "high_similarity": Prefers high similarity sentences (amplitude grows quickly, midpoint shifts slowly)
        """
        self.strategy = strategy

        # Strategy-specific parameters
        if strategy == "balanced":
            self.amplitude_growth = 1.0
            self.midpoint_base = 0.5
            self.midpoint_scale = 0.3
            self.steepness = 10.0
        elif strategy == "short_sentences":
            self.amplitude_growth = 0.8  # Slower amplitude growth
            self.midpoint_base = 0.4  # Start midpoint lower
            self.midpoint_scale = 0.4  # Faster midpoint displacement
            self.steepness = 5.0  # Reduce steepness
        elif strategy == "high_similarity":
            self.amplitude_growth = 2  # Faster amplitude growth
            self.midpoint_base = 0.6  # Start midpoint higher
            self.midpoint_scale = 0.2  # Slower midpoint displacement
            self.steepness = 20.0  # Increase steepness

    def get_params(self, x: float) -> tuple[float, float, float]:
        """Get parameters for sigmoid function based on input x.

        When x=1, the amplitude is 0, so the function is flat at y=0, and total tokens is 0.
        When x=0, the amplitude is maximum, the sigmoid reaches y≈1, so total tokens is maximum.

        Args:
            x: Optimization parameter between 0 and 1

        Returns:
            Tuple of (amplitude, midpoint, steepness) parameters
        """
        amplitude = (
            self.amplitude_growth * (1 - x) + 1e-6
        )  # Amplitude decreases as x increases
        midpoint = (
            self.midpoint_base + self.midpoint_scale * x
        )  # Midpoint shifts right as x increases
        return amplitude, midpoint, self.steepness

    @staticmethod
    def sigmoid(
        x: float | np.ndarray, amplitude: float, midpoint: float, steepness: float
    ) -> float | np.ndarray:
        """Sigmoid function with given parameters.

        Formula: amplitude / (1 + exp(-steepness * (x - midpoint)))

        Args:
            x: Input value(s)
            amplitude: Maximum value of the sigmoid
            midpoint: X-value where sigmoid reaches half of amplitude
            steepness: Controls how sharp the transition is

        Returns:
            Output of the sigmoid function
        """
        return amplitude / (1 + np.exp(-steepness * (x - midpoint)))

    def optimize_sigmoid(
        self,
        x: float,
        similarities: np.ndarray,
        ratios: np.ndarray,
        tokens: np.ndarray,
        objective_tokens: float,
    ) -> float:
        """Calculate the difference between current token count and objective.

        The root is the x value where the total tokens equals the objective tokens.

        Args:
            x: Optimization parameter
            similarities: Array of cosine similarities
            ratios: Array of sentence-to-chunk length ratios
            tokens: Array of token counts per sentence
            objective_tokens: Target number of tokens

        Returns:
            Difference between actual and objective token count
        """
        amplitude, midpoint, steepness = self.get_params(x)
        mask = np.where(
            ratios < self.sigmoid(similarities, amplitude, midpoint, steepness), 1, 0
        )
        return np.sum(tokens * mask) - objective_tokens

    def filter_sentences(
        self,
        similarities: np.ndarray,
        ratios: np.ndarray,
        tokens: np.ndarray,
        objective_percentage: float = 0.2,
        xtol: float = 1e-3,
    ) -> tuple[np.ndarray, float]:
        """Filter sentences to achieve target percentage of total tokens.

        Args:
            similarities: Array of cosine similarities between sentences and chunks
            ratios: Array of sentence-to-chunk length ratios
            tokens: Array of token counts per sentence
            objective_percentage: Target percentage of total tokens to select (default: 0.2 = 20%)
            xtol: Tolerance for the bisection method (default: 1e-3)

        Returns:
            Tuple of (mask, x_opt) where mask is a binary array indicating selected sentences
            and x_opt is the optimal parameter value
        """
        total_tokens = np.sum(tokens)
        objective_tokens = total_tokens * objective_percentage

        x_opt = bisect(
            lambda x: self.optimize_sigmoid(
                x, similarities, ratios, tokens, objective_tokens
            ),
            0,
            1,
            xtol=xtol,
        )
        amplitude, midpoint, steepness = self.get_params(x_opt)
        mask = np.where(
            ratios < self.sigmoid(similarities, amplitude, midpoint, steepness), 1, 0
        )
        return mask, x_opt

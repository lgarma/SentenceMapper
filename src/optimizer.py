"""Arctan-based sentence optimizer for filtering sentences based on similarity and length ratio."""

import numpy as np
from scipy.optimize import bisect
from typing import Literal


class ArctanOptimizer:
    """Class to optimize arctan parameters for filtering sentences based on similarity and length ratio."""

    def __init__(
        self, 
        gamma: float = 0.2,
        strategy: Literal["balanced", "short_sentences", "high_similarity"] = "balanced"
    ):
        """Initialize the optimizer.
        
        Args:
            gamma: Controls the steepness of the arctan curve (default: 0.2)
            strategy: Selection strategy (default: "balanced")
                - "balanced": Default behavior, balances similarity and length
                - "short_sentences": Prefers shorter sentences (alpha grows slowly, beta shifts quickly)
                - "high_similarity": Prefers high similarity sentences (alpha grows quickly, beta shifts slowly)
        """
        self.gamma = gamma
        self.strategy = strategy
        
        # Strategy-specific parameters
        if strategy == "short_sentences":
            self.alpha_growth = 0.5  # Slower alpha growth
            self.beta_base = 0.3     # Start beta lower
            self.beta_scale = 0.5    # Faster beta displacement
        elif strategy == "high_similarity":
            self.alpha_growth = 1.5  # Faster alpha growth
            self.beta_base = 0.6     # Start beta higher
            self.beta_scale = 0.2    # Slower beta displacement
        else:  # balanced
            self.alpha_growth = 1.0
            self.beta_base = 0.5
            self.beta_scale = 0.3

    def get_params(self, x: float) -> tuple[float, float, float]:
        """Get parameters for arctan function based on input x.
        
        When x=1, the amplitude is 0, so the function is a horizontal line at y=0, and total tokens is 0.
        When x=0, the amplitude is maximum, the arctan reaches y=1, so total tokens is maximum.
        
        Args:
            x: Optimization parameter between 0 and 1
            
        Returns:
            Tuple of (alpha, beta, gamma) parameters
        """
        # With normalization, arctan at infinity gives 1, so we don't need the division
        alpha = self.alpha_growth * (1 - x) + 1e-6  # Amplitude decreases as x increases
        beta = self.beta_base + self.beta_scale * x  # Horizontal shift increases as x increases
        return alpha, beta, self.gamma

    @staticmethod
    def arctan(x: float | np.ndarray, alpha: float, beta: float, gamma: float) -> float | np.ndarray:
        """Arctan function with given parameters, normalized to range [-1, 1].
        
        Args:
            x: Input value(s)
            alpha: Amplitude
            beta: Horizontal shift
            gamma: Controls the steepness of the curve
            
        Returns:
            Output of the arctan function, scaled to [-1, 1] range
        """
        # Normalize arctan output from [-pi/2, pi/2] to [-1, 1]
        return alpha * (2 / np.pi) * np.arctan((x - beta) / gamma)

    def optimize_arctan(
        self, 
        x: float, 
        similarities: np.ndarray, 
        ratios: np.ndarray, 
        tokens: np.ndarray, 
        objective_tokens: float
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
        alpha, beta, gamma = self.get_params(x)
        mask = np.where(ratios < self.arctan(similarities, alpha, beta, gamma), 1, 0)
        return np.sum(tokens * mask) - objective_tokens
    
    def filter_sentences(
        self, 
        similarities: np.ndarray, 
        ratios: np.ndarray, 
        tokens: np.ndarray, 
        objective_percentage: float = 0.2,
        xtol: float = 1e-3
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
            lambda x: self.optimize_arctan(x, similarities, ratios, tokens, objective_tokens),
            0, 1, xtol=xtol
        )
        alpha, beta, gamma = self.get_params(x_opt)
        mask = np.where(ratios < self.arctan(similarities, alpha, beta, gamma), 1, 0)
        return mask, x_opt

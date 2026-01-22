"""Power law-based sentence optimizer for filtering sentences based on frontier analysis."""

import numpy as np
from scipy.optimize import bisect
from sklearn.linear_model import LinearRegression


class PowerLawOptimizer:
    """Optimizer that uses a power law frontier to select information-dense sentences.

    The power law relationship is: similarity = A * ratio^B
    Where:
    - ratio is the independent variable (sentence length / chunk length)
    - B (slope) is fixed and represents the natural scaling relationship
    - A (amplitude) is adjusted to select more or fewer sentences

    Sentences ABOVE the power law curve are selected (information-dense).
    """

    def __init__(
        self, slope: float = 1.0, intercept: float = 0.0, min_ratio: float = 0.01
    ):
        """Initialize the optimizer.

        Args:
            slope: Fixed power law exponent (default: 1.0)
                   Higher values = steeper relationship between ratio and similarity
            intercept: Initial intercept from fitted frontier in log-log space (default: 0.0)
                      From the log-log linear regression: log(similarity) = slope * log(ratio) + intercept
            min_ratio: Minimum ratio threshold to avoid numerical issues (default: 0.01)
        """
        self.slope = slope
        self.initial_intercept = intercept
        self.min_ratio = min_ratio

    def get_params(self, x: float) -> tuple[float, float]:
        """Get parameters for power law function based on input x.

        When x=0, the intercept equals the initial fitted baseline value.
        When x=1, the intercept is moved up significantly to select fewer sentences.
        The intercept moves up as x increases while slope remains fixed.

        Args:
            x: Optimization parameter between 0 and 1

        Returns:
            Tuple of (amplitude, slope) parameters
        """
        # Move intercept UP as x increases (to select fewer sentences)
        # At x=0: intercept = initial_intercept (baseline - about 50% selected)
        # At x=1: intercept is increased by ~3 orders of magnitude in log space
        intercept = self.initial_intercept + 3 * x

        # Convert intercept to amplitude: amplitude = 10^intercept
        amplitude = 10**intercept

        return amplitude, self.slope

    @staticmethod
    def powerlaw(
        ratio: float | np.ndarray, amplitude: float, slope: float
    ) -> float | np.ndarray:
        """Power law function: similarity = amplitude * ratio^slope.

        Args:
            ratio: Input ratio value(s) (sentence length / chunk length)
            amplitude: Scaling factor (intercept in log-log space)
            slope: Power law exponent

        Returns:
            Expected similarity threshold for given ratio
        """
        return amplitude * np.power(ratio, slope)

    def optimize_powerlaw(
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
            Difference between current token count and objective
        """
        amplitude, slope = self.get_params(x)

        # Calculate threshold similarity for each ratio
        # Clip ratios to avoid numerical issues with very small values
        ratio_clipped = np.maximum(ratios, self.min_ratio)
        threshold_similarities = self.powerlaw(ratio_clipped, amplitude, slope)

        # Select sentences above the power law curve
        mask = similarities >= threshold_similarities
        current_tokens = np.sum(tokens * mask)

        return current_tokens - objective_tokens

    def filter_sentences(
        self,
        similarities: np.ndarray,
        ratios: np.ndarray,
        tokens: np.ndarray,
        objective_percentage: float,
    ) -> tuple[np.ndarray, float]:
        """Filter sentences to achieve a target percentage of total tokens.

        Sentences ABOVE the power law frontier are considered information-dense
        and are selected.

        Args:
            similarities: Array of cosine similarities
            ratios: Array of sentence-to-chunk length ratios
            tokens: Array of token counts per sentence
            objective_percentage: Target percentage of tokens to keep (0 to 1)

        Returns:
            Tuple of (mask, x_opt) where:
                - mask: Binary array indicating selected sentences (1) or not (0)
                - x_opt: Optimal x parameter value found
        """
        total_tokens = np.sum(tokens)
        objective_tokens = total_tokens * objective_percentage

        # Handle edge cases
        if objective_percentage >= 1.0:
            return np.ones(len(similarities), dtype=int), 0.0
        if objective_percentage <= 0.0:
            return np.zeros(len(similarities), dtype=int), 1.0

        # Use bisection to find optimal x
        try:
            x_opt = bisect(
                lambda x: self.optimize_powerlaw(
                    x, similarities, ratios, tokens, objective_tokens
                ),
                0.0,  # x=0 selects most sentences
                1.0,  # x=1 selects fewest sentences
                xtol=1e-4,
                maxiter=100,
            )
        except ValueError:
            # If bisection fails, return the closest extreme
            tokens_at_0 = self.optimize_powerlaw(
                0.0, similarities, ratios, tokens, objective_tokens
            )
            if tokens_at_0 < 0:
                # Can't reach target even at maximum selection
                x_opt = 0.0
            else:
                # Target too low, select minimum
                x_opt = 1.0

        # Generate final mask with optimal x
        amplitude, slope = self.get_params(x_opt)
        ratio_clipped = np.maximum(ratios, self.min_ratio)
        threshold_similarities = self.powerlaw(ratio_clipped, amplitude, slope)
        mask = (similarities >= threshold_similarities).astype(int)

        return mask, x_opt


def fit_frontier_curve(
    similarities: np.ndarray,
    ratios: np.ndarray,
    quantile: float = 0.05,
    method: str = "binned_min",
    n_bins: int = 20,
    min_points_per_bin: int = 5,
) -> tuple[float, float, dict]:
    """Fit a linear function to the baseline relationship in log-log space.

    This identifies the lower-bound ratio-similarity relationship. Sentences above
    this baseline have higher similarity than the minimum expected for their length.

    Args:
        similarities: Array of cosine similarities
        ratios: Array of sentence-to-chunk length ratios
        quantile: Quantile to use for baseline (default: 0.05, i.e., 5th percentile)
        method: Method to use ('quantile' or 'binned_min')
            - 'quantile': Quantile regression on all points
            - 'binned_min': Fit to low quantile values in binned ranges (recommended)
        n_bins: Number of bins for 'binned_min' method
        min_points_per_bin: Minimum points per bin for 'binned_min' method

    Returns:
        slope: Slope of the line in log-log space (power law exponent)
        intercept: Intercept of the line in log-log space
        info: Dictionary with additional information:
            - 'log_ratio': Log-transformed ratios used for fitting (X-axis)
            - 'log_sim': Log-transformed similarities used for fitting (Y-axis)
            - 'equation': String representation of the fitted equation
            - 'r_squared': R² value of the fit
    """
    # Filter out zeros and negative values for log transform
    valid_mask = (similarities > 0) & (ratios > 0)
    sim_valid = similarities[valid_mask]
    ratio_valid = ratios[valid_mask]

    # Log transform
    log_ratio = np.log10(ratio_valid)
    log_sim = np.log10(sim_valid)

    if method == "quantile":
        # Fit to the specified quantile (default: median/lower bound)
        # We'll use a sliding window approach to estimate the quantile frontier
        sort_idx = np.argsort(log_ratio)
        log_ratio_sorted = log_ratio[sort_idx]
        log_sim_sorted = log_sim[sort_idx]

        # Calculate rolling quantile
        window_size = max(len(log_ratio) // 20, 10)
        frontier_ratio = []
        frontier_sim = []

        for i in range(0, len(log_ratio_sorted) - window_size, window_size // 2):
            window_sims = log_sim_sorted[i : i + window_size]
            frontier_sim.append(np.quantile(window_sims, quantile))
            frontier_ratio.append(np.median(log_ratio_sorted[i : i + window_size]))

        frontier_ratio = np.array(frontier_ratio)
        frontier_sim = np.array(frontier_sim)

        # Fit linear regression to frontier points
        X = frontier_ratio.reshape(-1, 1)
        y = frontier_sim
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        # Calculate R²
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    elif method == "binned_min":
        # Divide ratio range into bins and take minimum similarity in each bin
        bins = np.linspace(log_ratio.min(), log_ratio.max(), n_bins + 1)
        bin_indices = np.digitize(log_ratio, bins)

        frontier_ratio = []
        frontier_sim = []

        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if np.sum(mask) >= min_points_per_bin:
                frontier_ratio.append(log_ratio[mask].mean())
                frontier_sim.append(log_sim[mask].min())

        frontier_ratio = np.array(frontier_ratio)
        frontier_sim = np.array(frontier_sim)

        # Fit linear regression to frontier points
        X = frontier_ratio.reshape(-1, 1)
        y = frontier_sim
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        # Calculate R²
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    else:
        raise ValueError(f"Unknown method: {method}. Use 'quantile' or 'binned_min'")

    # Create equation string
    # In log-log space: log(similarity) = slope * log(ratio) + intercept
    # In original space: similarity = 10^intercept * ratio^slope
    equation = f"similarity = {10**intercept:.4e} * ratio^{slope:.3f}"

    info = {
        "log_ratio": log_ratio,
        "log_sim": log_sim,
        "equation": equation,
        "r_squared": r_squared,
        "frontier_ratio": frontier_ratio,
        "frontier_sim": frontier_sim,
    }

    return slope, intercept, info

"""Power law-based sentence optimizer for filtering sentences based on frontier analysis."""

import numpy as np
from scipy.optimize import bisect
from sklearn.linear_model import LinearRegression


class PowerLawOptimizer:
    """Optimizer that uses a power law frontier to select information-dense sentences.

    The power law relationship is: ratio = A * similarity^B
    Where:
    - B (slope) is fixed and represents the natural scaling relationship
    - A (amplitude) is adjusted to select more or fewer sentences

    Sentences BELOW the power law curve are selected (information-dense).
    """

    def __init__(
        self, slope: float = 1.0, intercept: float = 0.0, min_similarity: float = 0.01
    ):
        """Initialize the optimizer.

        Args:
            slope: Fixed power law exponent (default: 1.0)
                   Higher values = steeper relationship between similarity and ratio
            intercept: Initial intercept from fitted frontier in log-log space (default: 0.0)
                      From the log-log linear regression: log(ratio) = slope * log(sim) + intercept
            min_similarity: Minimum similarity threshold to avoid numerical issues (default: 0.01)
        """
        self.slope = slope
        self.initial_intercept = intercept
        self.min_similarity = min_similarity

    def get_params(self, x: float) -> tuple[float, float]:
        """Get parameters for power law function based on input x.

        When x=0, the intercept equals the initial fitted frontier value.
        When x=1, the intercept is moved down significantly to select fewer sentences.
        The intercept moves down as x increases while slope remains fixed.

        Args:
            x: Optimization parameter between 0 and 1

        Returns:
            Tuple of (amplitude, slope) parameters
        """
        # Move intercept down as x increases
        # At x=0: intercept = initial_intercept (frontier)
        # At x=1: intercept is reduced by ~3 orders of magnitude in log space
        intercept = self.initial_intercept - 3 * x

        # Convert intercept to amplitude: amplitude = 10^intercept
        amplitude = 10**intercept

        return amplitude, self.slope

    @staticmethod
    def powerlaw(
        similarity: float | np.ndarray, amplitude: float, slope: float
    ) -> float | np.ndarray:
        """Power law function: ratio = amplitude * similarity^slope.

        Args:
            similarity: Input similarity value(s)
            amplitude: Scaling factor (intercept in log-log space)
            slope: Power law exponent

        Returns:
            Expected ratio threshold for given similarity
        """
        return amplitude * np.power(similarity, slope)

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

        # Calculate threshold ratio for each similarity
        # Clip similarities to avoid numerical issues with very small values
        sim_clipped = np.maximum(similarities, self.min_similarity)
        threshold_ratios = self.powerlaw(sim_clipped, amplitude, slope)

        # Select sentences below the power law curve
        mask = ratios <= threshold_ratios
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

        Sentences BELOW the power law frontier are considered information-dense
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
        sim_clipped = np.maximum(similarities, self.min_similarity)
        threshold_ratios = self.powerlaw(sim_clipped, amplitude, slope)
        mask = (ratios <= threshold_ratios).astype(int)

        return mask, x_opt


def fit_frontier_curve(
    similarities: np.ndarray,
    ratios: np.ndarray,
    quantile: float = 0.95,
    method: str = "quantile",
    n_bins: int = 20,
    min_points_per_bin: int = 5,
) -> tuple[float, float, dict]:
    """Fit a linear function to the frontier (upper envelope) in log-log space.

    This identifies the "expected" similarity-length relationship. Sentences below
    this frontier are more information-dense than expected for their length.

    Args:
        similarities: Array of cosine similarities
        ratios: Array of sentence-to-chunk length ratios
        quantile: Quantile to use for frontier (default: 0.95, i.e., 95th percentile)
        method: Method to use ('quantile' or 'binned_max')
            - 'quantile': Quantile regression on all points
            - 'binned_max': Fit to maximum values in binned ranges
        n_bins: Number of bins for 'binned_max' method
        min_points_per_bin: Minimum points per bin for 'binned_max' method

    Returns:
        slope: Slope of the line in log-log space (power law exponent)
        intercept: Intercept of the line in log-log space
        info: Dictionary with additional information:
            - 'log_sim': Log-transformed similarities used for fitting
            - 'log_ratio': Log-transformed ratios used for fitting
            - 'equation': String representation of the fitted equation
            - 'r_squared': R² value of the fit
    """
    # Filter out zeros and negative values for log transform
    valid_mask = (similarities > 0) & (ratios > 0)
    sim_valid = similarities[valid_mask]
    ratio_valid = ratios[valid_mask]

    # Log transform
    log_sim = np.log10(sim_valid)
    log_ratio = np.log10(ratio_valid)

    if method == "quantile":
        # Fit to the upper quantile (frontier)
        # We'll use a sliding window approach to estimate the quantile frontier
        sort_idx = np.argsort(log_sim)
        log_sim_sorted = log_sim[sort_idx]
        log_ratio_sorted = log_ratio[sort_idx]

        # Calculate rolling quantile
        window_size = max(len(log_sim) // 20, 10)
        frontier_sim = []
        frontier_ratio = []

        for i in range(0, len(log_sim_sorted) - window_size, window_size // 2):
            window_ratios = log_ratio_sorted[i : i + window_size]
            frontier_ratio.append(np.quantile(window_ratios, quantile))
            frontier_sim.append(np.median(log_sim_sorted[i : i + window_size]))

        frontier_sim = np.array(frontier_sim)
        frontier_ratio = np.array(frontier_ratio)

        # Fit linear regression to frontier points
        X = frontier_sim.reshape(-1, 1)
        y = frontier_ratio
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        # Calculate R²
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    elif method == "binned_max":
        # Divide similarity range into bins and take max ratio in each bin
        bins = np.linspace(log_sim.min(), log_sim.max(), n_bins + 1)
        bin_indices = np.digitize(log_sim, bins)

        frontier_sim = []
        frontier_ratio = []

        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if np.sum(mask) >= min_points_per_bin:
                frontier_sim.append(log_sim[mask].mean())
                frontier_ratio.append(log_ratio[mask].max())

        frontier_sim = np.array(frontier_sim)
        frontier_ratio = np.array(frontier_ratio)

        # Fit linear regression to frontier points
        X = frontier_sim.reshape(-1, 1)
        y = frontier_ratio
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
        raise ValueError(f"Unknown method: {method}. Use 'quantile' or 'binned_max'")

    # Create equation string
    # In log-log space: log(ratio) = slope * log(sim) + intercept
    # In original space: ratio = 10^intercept * sim^slope
    equation = f"ratio = {10**intercept:.4e} * similarity^{slope:.3f}"

    info = {
        "log_sim": log_sim,
        "log_ratio": log_ratio,
        "equation": equation,
        "r_squared": r_squared,
        "frontier_sim": frontier_sim,
        "frontier_ratio": frontier_ratio,
    }

    return slope, intercept, info

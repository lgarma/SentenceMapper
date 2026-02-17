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

    The frontier is fitted to the UPPER bound (ceiling) of the ratio-similarity
    distribution. The upper frontier is more stable across documents than the
    lower frontier. To select sentences, the ceiling is lowered until the
    target token budget is reached. Sentences ABOVE the lowered curve are
    selected (information-dense).
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

        The frontier is fitted to the upper bound (ceiling). To select
        sentences we lower the ceiling. At x=0 the curve is far below the
        ceiling (most sentences selected). At x=1 the curve sits at the
        ceiling (fewest sentences selected).

        Args:
            x: Optimization parameter between 0 and 1

        Returns:
            Tuple of (amplitude, slope) parameters
        """
        # Move intercept DOWN from the ceiling as x decreases
        # At x=1: intercept = initial_intercept (at the ceiling, fewest selected)
        # At x=0: intercept = initial_intercept - 3 (well below, most selected)
        intercept = self.initial_intercept - 3 * (1 - x)

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

        The upper frontier (ceiling) is lowered until the selected sentences
        sum to approximately the target token count. Sentences ABOVE the
        lowered curve are considered information-dense and are selected.

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
    quantile: float = 0.95,
    method: str = "binned_max",
    n_bins: int = 20,
    min_points_per_bin: int = 5,
    robust_fit: bool = True,
    outlier_sigma: float = 3.0,
) -> tuple[float, float, dict]:
    """Fit a linear function to the upper frontier relationship in log-log space.

    This identifies the upper-bound ratio-similarity relationship — the ceiling
    of representativeness. The upper frontier is more stable across documents
    than the lower frontier, as it reflects the fundamental information-theoretic
    limit of how much context a sentence of a given length can capture.

    Sentences are selected by lowering this ceiling until the target token
    budget is reached.

    Args:
        similarities: Array of cosine similarities
        ratios: Array of sentence-to-chunk length ratios
        quantile: Quantile to use for frontier (default: 0.95, i.e., 95th percentile)
        method: Method to use ('quantile', 'binned_min', or 'binned_max')
            - 'quantile': Quantile regression on all points
            - 'binned_max': Fit to high quantile values in binned ranges (recommended)
            - 'binned_min': Fit to low quantile values in binned ranges (legacy)
        n_bins: Number of bins for binned methods
        min_points_per_bin: Minimum points per bin for binned methods
        robust_fit: If True, perform a robust refit by removing frontier-point
            outliers based on median absolute deviation (MAD) of residuals.
        outlier_sigma: Outlier cutoff (in robust sigma units) used when
            robust_fit=True.

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

    def _fit_with_optional_robust_refit(
        x_frontier: np.ndarray,
        y_frontier: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray]:
        """Fit linear frontier with optional robust outlier rejection."""
        X = x_frontier.reshape(-1, 1)
        y = y_frontier

        model = LinearRegression()
        model.fit(X, y)

        inlier_mask = np.ones_like(y, dtype=bool)

        # Optional robust refit to reduce sensitivity to single bad bins.
        if robust_fit and len(y) >= 6:
            residuals = y - model.predict(X)
            med = np.median(residuals)
            mad = np.median(np.abs(residuals - med))

            if mad > 0:
                robust_sigma = 1.4826 * mad
                candidate_mask = np.abs(residuals - med) <= outlier_sigma * robust_sigma

                # Keep enough points to avoid unstable fits.
                if np.sum(candidate_mask) >= max(3, len(y) // 2):
                    inlier_mask = candidate_mask
                    X_in = X[inlier_mask]
                    y_in = y[inlier_mask]
                    model = LinearRegression()
                    model.fit(X_in, y_in)

        slope_ = model.coef_[0]
        intercept_ = model.intercept_

        # Report R² over the points actually used in the final fit.
        X_eval = X[inlier_mask]
        y_eval = y[inlier_mask]
        y_pred = model.predict(X_eval)
        ss_res = np.sum((y_eval - y_pred) ** 2)
        ss_tot = np.sum((y_eval - np.mean(y_eval)) ** 2)
        r_squared_ = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return slope_, intercept_, r_squared_, inlier_mask

    if method == "quantile":
        # Fit to the specified quantile (default: 95th percentile / upper bound)
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

        slope, intercept, r_squared, frontier_inlier_mask = (
            _fit_with_optional_robust_refit(frontier_ratio, frontier_sim)
        )

    elif method == "binned_max":
        # Divide ratio range into bins and take maximum similarity in each bin
        bins = np.linspace(log_ratio.min(), log_ratio.max(), n_bins + 1)
        bin_indices = np.digitize(log_ratio, bins)

        frontier_ratio = []
        frontier_sim = []

        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if np.sum(mask) >= min_points_per_bin:
                frontier_ratio.append(log_ratio[mask].mean())
                frontier_sim.append(log_sim[mask].max())

        frontier_ratio = np.array(frontier_ratio)
        frontier_sim = np.array(frontier_sim)

        slope, intercept, r_squared, frontier_inlier_mask = (
            _fit_with_optional_robust_refit(frontier_ratio, frontier_sim)
        )

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

        slope, intercept, r_squared, frontier_inlier_mask = (
            _fit_with_optional_robust_refit(frontier_ratio, frontier_sim)
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'quantile', 'binned_max', or 'binned_min'"
        )

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
        "frontier_inlier_mask": frontier_inlier_mask,
        "robust_fit": robust_fit,
        "outlier_sigma": outlier_sigma,
    }

    return slope, intercept, info

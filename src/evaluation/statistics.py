"""Statistical validation module for summarization evaluation.

Provides non-parametric bootstrap confidence intervals and paired bootstrap hypothesis tests
to rigorously compare proposed submodular methods against baselines.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class StatisticalTester:
    """Computes bootstrap confidence intervals and paired hypothesis tests."""

    @staticmethod
    def bootstrap_ci(
        scores: list[float] | np.ndarray,
        confidence_level: float = 0.95,
        n_bootstraps: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """Compute mean and bootstrap confidence interval (low, high)."""
        arr = np.asarray(scores, dtype=np.float64)
        if len(arr) == 0:
            return 0.0, 0.0, 0.0

        rng = np.random.default_rng(seed)
        mean_val = float(np.mean(arr))

        boot_means = np.empty(n_bootstraps, dtype=np.float64)
        n = len(arr)
        for i in range(n_bootstraps):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        alpha = 1.0 - confidence_level
        low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
        high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

        return mean_val, low, high

    @staticmethod
    def paired_bootstrap_test(
        proposed_scores: list[float] | np.ndarray,
        baseline_scores: list[float] | np.ndarray,
        n_bootstraps: int = 1000,
        seed: int = 42,
    ) -> dict[str, float]:
        """Perform paired bootstrap hypothesis test: H0: mean(diff) <= 0.

        Returns:
            dict containing mean_diff, p_value, and stat_significant_at_05 (bool as float).
        """
        prop = np.asarray(proposed_scores, dtype=np.float64)
        base = np.asarray(baseline_scores, dtype=np.float64)

        if len(prop) != len(base) or len(prop) == 0:
            return {"mean_diff": 0.0, "p_value": 1.0, "significant_05": 0.0}

        diffs = prop - base
        observed_mean_diff = float(np.mean(diffs))

        # Centered bootstrap under null hypothesis
        centered_diffs = diffs - observed_mean_diff
        rng = np.random.default_rng(seed)
        n = len(diffs)

        boot_diffs = np.empty(n_bootstraps, dtype=np.float64)
        for i in range(n_bootstraps):
            boot_sample = rng.choice(centered_diffs, size=n, replace=True)
            boot_diffs[i] = np.mean(boot_sample)

        # One-tailed p-value
        p_val = float(np.mean(boot_diffs >= observed_mean_diff))

        return {
            "mean_diff": observed_mean_diff,
            "p_value": p_val,
            "significant_05": float(p_val < 0.05),
        }


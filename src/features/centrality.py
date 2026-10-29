"""Sentence centrality calculation.

Calculates the singleton centrality reward:
    r_j = (1 / N) * sum_{i in V} w_ij
representing how representative sentence j is of the entire document cluster.
"""

from __future__ import annotations

import numpy as np


class CentralityCalculator:
    """Computes global centrality scores for candidate sentences."""

    @staticmethod
    def compute(W: np.ndarray) -> np.ndarray:
        """Compute centrality vector r of shape [N].

        Args:
            W: Symmetric similarity matrix of shape [N, N].

        Returns:
            np.ndarray: Non-negative centrality vector of shape [N].
        """
        n = W.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        # Average similarity to all other sentences in the cluster
        r = np.sum(W, axis=0) / float(n)
        return np.maximum(r, 0.0)


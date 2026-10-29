"""Truncated Coverage Objective L_1(S).

Implementation of the truncated coverage function proposed in Lin & Bilmes (2011):
    C_i(S) = sum_{j in S} w_ij
    C_i(V) = sum_{k in V} w_ik
    L_1(S) = sum_{i in V} min(C_i(S), alpha * C_i(V))

Mathematical Proof of Monotone Submodularity:
1. Modular Base:
   For any sentence i, the coverage function C_i(S) = sum_{j in S} w_ij is modular
   because C_i(A union {j}) - C_i(A) = w_ij is independent of A.
   Since w_ij >= 0 (clipped cosine similarity), C_i(S) is non-negative and monotone non-decreasing.

2. Concave Non-Decreasing Composition:
   Let g_i(x) = min(x, alpha * C_i(V)), where alpha in [0, 1].
   The function g_i(x) is concave (minimum of two affine functions) and monotonically non-decreasing
   on [0, infinity).
   By the composition theorem (Nemhauser, Wolsey, & Fisher, 1978), the composition of any concave
   non-decreasing function with a non-negative monotone submodular (or modular) function is
   itself monotone submodular. Hence, g_i(C_i(S)) is monotone submodular for every i in V.

3. Non-Negative Linear Combination:
   L_1(S) = sum_{i in V} g_i(C_i(S)) is a sum of monotone submodular functions with weights = 1 >= 0.
   Submodularity and monotonicity are closed under non-negative linear combinations.
   Therefore, L_1(S) is strictly monotone submodular.
"""

from __future__ import annotations

import numpy as np


class TruncatedCoverageObjective:
    """Evaluates L_1(S) with incremental state tracking for efficient greedy search."""

    def __init__(self, W: np.ndarray, alpha: float = 0.20) -> None:
        """Initialize coverage objective.

        Args:
            W: Non-negative symmetric similarity matrix of shape [N, N].
            alpha: Saturation threshold parameter in [0, 1].
        """
        self.W = W
        self.n = W.shape[0]
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

        # Total coverage of each sentence in ground set V: C_i(V) = sum_k w_ik
        self.total_coverage_V = np.sum(W, axis=1) if self.n > 0 else np.zeros(0)
        # Saturation thresholds: alpha * C_i(V)
        self.thresholds = self.alpha * self.total_coverage_V

        # Incremental state: C_i(S) for currently selected subset S
        self.current_coverage = np.zeros(self.n, dtype=np.float64)
        self.selected_set: set[int] = set()

    def reset_state(self) -> None:
        """Reset incremental state to empty selection S = {}."""
        self.current_coverage.fill(0.0)
        self.selected_set.clear()

    def evaluate(self, selected_ids: set[int] | list[int]) -> float:
        """Evaluate L_1(S) from scratch on an arbitrary subset S."""
        if not selected_ids or self.n == 0:
            return 0.0

        selected_arr = list(selected_ids)
        # C_i(S) = sum_{j in S} w_ij
        cov = np.sum(self.W[:, selected_arr], axis=1)
        # min(C_i(S), alpha * C_i(V))
        truncated = np.minimum(cov, self.thresholds)
        return float(np.sum(truncated))

    def marginal_gain(self, candidate_id: int, selected_ids: set[int] | None = None) -> float:
        """Compute Delta_L1(j | S) = L_1(S union {j}) - L_1(S).

        Uses incremental current_coverage state if selected_ids matches selected_set,
        otherwise evaluates incrementally using current_coverage.
        """
        if candidate_id in self.selected_set or self.n == 0:
            return 0.0

        # w_i,candidate for all i
        w_cand = self.W[:, candidate_id]

        # Old contribution: min(current_coverage[i], threshold[i])
        old_val = np.minimum(self.current_coverage, self.thresholds)
        # New contribution: min(current_coverage[i] + w_cand[i], threshold[i])
        new_val = np.minimum(self.current_coverage + w_cand, self.thresholds)

        gain = np.sum(new_val - old_val)
        return float(max(0.0, gain))

    def update_state_after_select(self, candidate_id: int) -> None:
        """Update incremental coverage vector after candidate_id is selected into S."""
        if candidate_id not in self.selected_set and self.n > 0:
            self.current_coverage += self.W[:, candidate_id]
            self.selected_set.add(candidate_id)


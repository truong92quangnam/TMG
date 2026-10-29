"""Query-Aware Truncated Coverage Objective L_Q(S).

Mathematical Extension of Lin & Bilmes (2011) Document Summarization:

In the original Lin & Bilmes (2011) formulation, the coverage function:
    L_1(S) = sum_{i in V} min(C_i(S), alpha * C_i(V))
is entirely query-agnostic: every sentence i in V has identical coverage demand,
irrespective of its relevance to the user's query Q.

This module implements Query-Aware Coverage L_Q(S), combining:
1. Relevance-Weighted Sentence Coverage:
   Sentences that are semantically relevant to query Q have higher coverage priority.
   rho_i(Q) = eta + (1 - eta) * r_iQ,  where eta in [0, 1] is a background coverage floor.
2. Direct Target Query Coverage:
   The query Q itself is treated as a conceptual target that candidate sentences directly cover:
   C_Q(S) = sum_{j in S} w_jQ
   L_{Q, direct}(S) = min(C_Q(S), alpha_Q * C_Q(V))

Combined Objective Formulation:
    L_Q(S) = (1 - gamma) * sum_{i in V} rho_i(Q) * min(C_i(S), alpha * C_i(V))
           + gamma * min(C_Q(S), alpha_Q * C_Q(V))

Mathematical Proof of Monotone Submodularity:
1. Modular Base:
   - For every i in V, C_i(S) = sum_{j in S} w_ij is modular and non-negative (w_ij >= 0).
   - C_Q(S) = sum_{j in S} w_jQ is modular and non-negative (w_jQ >= 0).
2. Concave Non-Decreasing Composition:
   - g_i(x) = min(x, alpha * C_i(V)) is concave and non-decreasing on [0, infinity).
   - g_Q(x) = min(x, alpha_Q * C_Q(V)) is concave and non-decreasing on [0, infinity).
   By the Lovasz/Nemhauser composition theorem, g_i(C_i(S)) and g_Q(C_Q(S)) are
   monotone submodular functions.
3. Non-Negative Linear Combination:
   - Since rho_i(Q) >= 0, gamma in [0, 1], and (1 - gamma) >= 0, L_Q(S) is a sum of
     monotone submodular functions with strictly non-negative coefficients.
   Therefore, L_Q(S) is rigorously proven to be MONOTONE SUBMODULAR.
4. Smooth Generalization:
   When gamma = 0 and rho_i(Q) = 1 (or query is None), L_Q(S) exactly reduces to L_1(S).
"""

from __future__ import annotations

from typing import Any
import numpy as np


class QueryAwareCoverageObjective:
    """Evaluates query-aware coverage L_Q(S) with incremental state tracking."""

    def __init__(
        self,
        W: np.ndarray,
        query_sim: np.ndarray | None = None,
        alpha: float = 0.20,
        alpha_q: float = 0.50,
        gamma: float = 0.30,
        eta: float = 0.20,
    ) -> None:
        """Initialize Query-Aware Coverage Objective.

        Args:
            W: Non-negative symmetric similarity matrix of shape [N, N].
            query_sim: Query similarity vector w_jQ of shape [N] in [0, 1].
                       If None, defaults to uniform 0.0 (equivalent to generic L_1).
            alpha: Saturation threshold for sentence coverage in [0, 1].
            alpha_q: Saturation threshold for direct query coverage in [0, 1].
            gamma: Weight for direct query coverage in [0, 1].
                   (1 - gamma) is the weight for query-weighted sentence coverage.
            eta: Background smoothing factor in [0, 1].
                 rho_i(Q) = eta + (1 - eta) * query_sim[i].
        """
        self.W = W
        self.n = W.shape[0]
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.alpha_q = float(np.clip(alpha_q, 0.0, 1.0))
        self.gamma = float(np.clip(gamma, 0.0, 1.0)) if query_sim is not None else 0.0
        self.eta = float(np.clip(eta, 0.0, 1.0))

        # Query similarities w_jQ
        if query_sim is not None and len(query_sim) == self.n:
            self.w_query = np.maximum(np.asarray(query_sim, dtype=np.float64), 0.0)
            self.rho = self.eta + (1.0 - self.eta) * self.w_query
        else:
            self.w_query = np.zeros(self.n, dtype=np.float64)
            self.rho = np.ones(self.n, dtype=np.float64)
            self.gamma = 0.0

        # Total document coverage in V: C_i(V) = sum_k w_ik
        self.total_coverage_V = np.sum(W, axis=1) if self.n > 0 else np.zeros(0)
        self.thresholds = self.alpha * self.total_coverage_V

        # Total query coverage in V: C_Q(V) = sum_k w_kQ
        self.total_query_coverage_V = float(np.sum(self.w_query))
        self.threshold_query = self.alpha_q * self.total_query_coverage_V

        # Incremental states
        self.current_coverage = np.zeros(self.n, dtype=np.float64)
        self.current_query_cov = 0.0
        self.selected_set: set[int] = set()

    def reset_state(self) -> None:
        """Reset incremental state to S = empty set."""
        self.current_coverage.fill(0.0)
        self.current_query_cov = 0.0
        self.selected_set.clear()

    def evaluate(self, selected_ids: set[int] | list[int]) -> float:
        """Evaluate L_Q(S) from scratch on an arbitrary subset S."""
        if not selected_ids or self.n == 0:
            return 0.0

        selected_arr = list(selected_ids)

        # 1. Document coverage: sum_{i in V} rho_i * min(C_i(S), alpha * C_i(V))
        cov_vec = np.sum(self.W[:, selected_arr], axis=1)
        truncated_doc = np.minimum(cov_vec, self.thresholds)
        weighted_doc_score = float(np.sum(self.rho * truncated_doc))

        # 2. Direct query coverage: min(C_Q(S), alpha_Q * C_Q(V))
        if self.gamma > 0.0:
            query_cov = float(np.sum(self.w_query[selected_arr]))
            truncated_query = min(query_cov, self.threshold_query)
            total_val = (1.0 - self.gamma) * weighted_doc_score + self.gamma * truncated_query
        else:
            total_val = weighted_doc_score

        return float(total_val)

    def marginal_gain(self, candidate_id: int, selected_ids: set[int] | None = None) -> float:
        """Compute Delta_LQ(j | S) = L_Q(S union {j}) - L_Q(S) in O(N) time."""
        if candidate_id in self.selected_set or self.n == 0:
            return 0.0

        # Document coverage marginal gain:
        w_cand = self.W[:, candidate_id]
        old_truncated = np.minimum(self.current_coverage, self.thresholds)
        new_truncated = np.minimum(self.current_coverage + w_cand, self.thresholds)
        gain_doc = np.sum(self.rho * (new_truncated - old_truncated))

        # Direct query coverage marginal gain:
        if self.gamma > 0.0:
            cand_q = self.w_query[candidate_id]
            old_q_trunc = min(self.current_query_cov, self.threshold_query)
            new_q_trunc = min(self.current_query_cov + cand_q, self.threshold_query)
            gain_query = new_q_trunc - old_q_trunc

            total_gain = (1.0 - self.gamma) * gain_doc + self.gamma * gain_query
        else:
            total_gain = gain_doc

        return float(max(0.0, total_gain))

    def update_state_after_select(self, candidate_id: int) -> None:
        """Update incremental state after candidate_id is added to S."""
        if candidate_id not in self.selected_set and self.n > 0:
            self.current_coverage += self.W[:, candidate_id]
            self.current_query_cov += self.w_query[candidate_id]
            self.selected_set.add(candidate_id)


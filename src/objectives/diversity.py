"""Cluster Diversity Objective R_1(S).

Implementation of the cluster diversity reward function proposed in Lin & Bilmes (2011):
    R_1(S) = sum_{k=1}^K sqrt( sum_{j in S intersection P_k} r_j )

Mathematical Proof of Monotone Submodularity:
1. Modular Base within Cluster:
   For partition cluster P_k and non-negative centrality rewards r_j >= 0,
   the cluster reward sum m_k(S) = sum_{j in S intersection P_k} r_j is modular.
   It is non-negative and monotone non-decreasing as more items from P_k are added.

2. Concave-over-Modular Diminishing Returns:
   The transformation function g(x) = sqrt(x) (or log(1 + x)) is concave and strictly non-decreasing
   on [0, infinity) with derivative g'(x) = 1 / (2 * sqrt(x)) decreasing as x grows.
   By the composition theorem, g(m_k(S)) is monotone submodular for each cluster k.

3. Sum of Submodular Functions:
   R_1(S) = sum_{k=1}^K g(m_k(S)) is a sum of monotone submodular functions, hence R_1(S) is
   strictly monotone submodular.
   Intuitively, when a cluster P_k already has representative sentences in S, the marginal gain
   of adding another sentence from the same cluster is penalized by the concave function,
   encouraging selection from previously unrepresented clusters.
"""

from __future__ import annotations

from typing import Callable, Literal
import numpy as np


class ClusterDiversityObjective:
    """Evaluates R_1(S) with O(1) incremental marginal gain evaluations."""

    def __init__(
        self,
        cluster_labels: np.ndarray,
        num_clusters: int,
        rewards: np.ndarray,
        concave_func: Literal["sqrt", "log"] = "sqrt",
    ) -> None:
        """Initialize diversity objective.

        Args:
            cluster_labels: Array mapping sentence_id -> cluster_id (shape [N]).
            num_clusters: Total number of clusters K.
            rewards: Centrality or query-combined rewards of shape [N] (r_j >= 0).
            concave_func: Concave diminishing return function ("sqrt" or "log").
        """
        self.labels = cluster_labels
        self.k = num_clusters
        self.rewards = np.maximum(rewards, 0.0)
        self.n = len(cluster_labels)
        self.concave_mode = concave_func

        if concave_func == "sqrt":
            self.g: Callable[[float], float] = np.sqrt
        elif concave_func == "log":
            self.g = lambda x: float(np.log1p(x))
        else:
            raise ValueError(f"Unsupported concave function: {concave_func}")

        # Incremental state: sum of rewards for each cluster k in selected set S
        self.current_cluster_rewards = np.zeros(self.k, dtype=np.float64)
        self.selected_set: set[int] = set()

    def reset_state(self) -> None:
        """Reset incremental state to empty selection S = {}."""
        self.current_cluster_rewards.fill(0.0)
        self.selected_set.clear()

    def evaluate(self, selected_ids: set[int] | list[int]) -> float:
        """Evaluate R_1(S) from scratch on an arbitrary subset S."""
        if not selected_ids or self.k == 0:
            return 0.0

        cluster_sums = np.zeros(self.k, dtype=np.float64)
        for idx in selected_ids:
            c = self.labels[idx]
            cluster_sums[c] += self.rewards[idx]

        return float(np.sum([self.g(val) for val in cluster_sums]))

    def marginal_gain(self, candidate_id: int, selected_ids: set[int] | None = None) -> float:
        """Compute Delta_R1(j | S) = R_1(S union {j}) - R_1(S) in O(1) time."""
        if candidate_id in self.selected_set or self.k == 0:
            return 0.0

        c = self.labels[candidate_id]
        r_j = self.rewards[candidate_id]
        cur_sum = self.current_cluster_rewards[c]

        # Marginal gain only comes from candidate's cluster c:
        gain = self.g(cur_sum + r_j) - self.g(cur_sum)
        return float(max(0.0, gain))

    def update_state_after_select(self, candidate_id: int) -> None:
        """Update incremental cluster reward sum after candidate_id is selected into S."""
        if candidate_id not in self.selected_set and self.k > 0:
            c = self.labels[candidate_id]
            self.current_cluster_rewards[c] += self.rewards[candidate_id]
            self.selected_set.add(candidate_id)


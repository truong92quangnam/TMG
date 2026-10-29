"""Composite Submodular Objective F(S).

Combines coverage L_1(S) and one or more diversity objectives:
    F(S) = L_1(S) + sum_{ell=1}^L lambda_ell * R_ell(S)

Guarantees Monotone Submodularity:
All component functions L_1 and R_ell are monotone submodular, and all coefficients
lambda_ell >= 0 are non-negative. Since submodular functions are closed under non-negative
linear combinations, F(S) is strictly monotone submodular.
"""

from __future__ import annotations

from typing import Any, Union
from src.objectives.coverage import TruncatedCoverageObjective
from src.objectives.query_coverage import QueryAwareCoverageObjective
from src.objectives.diversity import ClusterDiversityObjective

CoverageType = Union[TruncatedCoverageObjective, QueryAwareCoverageObjective]


class CompositeSubmodularObjective:
    """Composite monotone submodular objective combining coverage and diversity."""

    def __init__(
        self,
        coverage_obj: CoverageType,
        diversity_objs: list[tuple[float, ClusterDiversityObjective]],
    ) -> None:
        """Initialize composite objective.

        Args:
            coverage_obj: TruncatedCoverageObjective L_1(S).
            diversity_objs: List of pairs (lambda_weight, ClusterDiversityObjective).
        """
        self.coverage_obj = coverage_obj
        # Enforce non-negative weights to preserve monotonicity
        self.diversity_objs = [
            (max(0.0, float(weight)), div_obj) for weight, div_obj in diversity_objs
        ]
        self.selected_set: set[int] = set()

    def reset_state(self) -> None:
        """Reset state of all component objectives."""
        self.coverage_obj.reset_state()
        for _, div_obj in self.diversity_objs:
            div_obj.reset_state()
        self.selected_set.clear()

    def evaluate(self, selected_ids: set[int] | list[int]) -> float:
        """Evaluate composite function F(S) from scratch."""
        score = self.coverage_obj.evaluate(selected_ids)
        for weight, div_obj in self.diversity_objs:
            if weight > 0:
                score += weight * div_obj.evaluate(selected_ids)
        return float(score)

    def marginal_gain(self, candidate_id: int, selected_ids: set[int] | None = None) -> float:
        """Compute composite marginal gain Delta_F(j | S) = F(S union {j}) - F(S)."""
        gain = self.coverage_obj.marginal_gain(candidate_id, selected_ids)
        for weight, div_obj in self.diversity_objs:
            if weight > 0:
                gain += weight * div_obj.marginal_gain(candidate_id, selected_ids)
        return float(gain)

    def update_state_after_select(self, candidate_id: int) -> None:
        """Update states across all component objectives."""
        self.coverage_obj.update_state_after_select(candidate_id)
        for _, div_obj in self.diversity_objs:
            div_obj.update_state_after_select(candidate_id)
        self.selected_set.add(candidate_id)

    def get_breakdown(self, selected_ids: set[int] | list[int]) -> dict[str, float]:
        """Compute detailed breakdown of objective values for analysis."""
        cov_val = self.coverage_obj.evaluate(selected_ids)
        div_vals = [div_obj.evaluate(selected_ids) for _, div_obj in self.diversity_objs]
        weighted_div = sum(
            weight * div_val for (weight, _), div_val in zip(self.diversity_objs, div_vals)
        )
        total_val = cov_val + weighted_div

        return {
            "coverage_score": float(cov_val),
            "raw_diversity_sum": float(sum(div_vals)),
            "weighted_diversity_score": float(weighted_div),
            "final_objective_score": float(total_val),
        }


r"""Budgeted greedy submodular maximization optimizer.

Implements the modified cost-effective greedy algorithm under knapsack (word/byte budget) constraints:
    j* = argmax_{j in V \ S, cost(S union {j}) <= B} [ Delta(j | S) / cost[j] ]
with monotonic stopping criteria and chronological/original document order sorting.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from src.data.duc_loader import Sentence
from src.objectives.composite import CompositeSubmodularObjective

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Encapsulates the generated summary and optimization telemetry."""
    selected_indices: list[int]
    ordered_sentences: list[Sentence]
    summary_text: str
    total_cost: int
    budget_limit: int
    num_sentences: int
    budget_type: Literal["words", "bytes"]
    breakdown: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0


class BudgetedGreedyOptimizer:
    """Greedy submodular maximization under word or byte knapsack budget constraints.
    
    Supports:
    - Minoux (1978) Accelerated / Lazy Greedy with priority queue (exact equivalence).
    - Cost exponent scaling r in [0, 1] from Lin & Bilmes (2011) Section 4.2.
    - Modified greedy knapsack guarantee (Sviridenko 2004).
    - Fractional / benchmark budget fill (DUC official evaluation cut-off).
    """

    def __init__(
        self,
        budget_limit: int = 100,
        budget_type: Literal["words", "bytes"] = "words",
        sort_order: Literal["document_order", "appearance", "selection_order"] = "document_order",
        cost_power: float = 1.0,
        use_lazy_greedy: bool = True,
        fill_budget_fractional: bool = False,
        modified_greedy_guarantee: bool = False,
    ) -> None:
        self.budget_limit = budget_limit
        self.budget_type = budget_type
        self.sort_order = sort_order
        self.cost_power = cost_power
        self.use_lazy_greedy = use_lazy_greedy
        self.fill_budget_fractional = fill_budget_fractional
        self.modified_greedy_guarantee = modified_greedy_guarantee

    def get_costs(self, sentences: list[Sentence]) -> np.ndarray:
        """Extract item costs based on word or byte counts."""
        if self.budget_type == "words":
            costs = [max(1, s.word_count) for s in sentences]
        else:
            costs = [max(1, s.byte_count) for s in sentences]
        return np.array(costs, dtype=np.int64)

    def optimize(
        self,
        objective: CompositeSubmodularObjective,
        sentences: list[Sentence],
    ) -> SummaryResult:
        """Run cost-benefit greedy selection under budget limit B."""
        start_time = time.perf_counter()

        n = len(sentences)
        if n == 0 or self.budget_limit <= 0:
            return SummaryResult(
                selected_indices=[],
                ordered_sentences=[],
                summary_text="",
                total_cost=0,
                budget_limit=self.budget_limit,
                num_sentences=0,
                budget_type=self.budget_type,
                breakdown=objective.get_breakdown([]),
                runtime_seconds=0.0,
            )

        costs = self.get_costs(sentences)
        objective.reset_state()

        if self.use_lazy_greedy:
            selected, current_cost = self._optimize_lazy_greedy(objective, sentences, costs)
        else:
            selected, current_cost = self._optimize_standard_greedy(objective, sentences, costs)

        # Optional modified greedy knapsack guarantee check:
        # compare S_greedy with best feasible singleton {v}
        if self.modified_greedy_guarantee:
            f_greedy = objective.evaluate(selected)
            best_single = -1
            best_single_val = -1.0
            for j in range(n):
                if costs[j] <= self.budget_limit:
                    val = objective.evaluate([j])
                    if val > best_single_val:
                        best_single_val = val
                        best_single = j
            if best_single != -1 and best_single_val > f_greedy:
                selected = [best_single]
                current_cost = int(costs[best_single])

        # Sort selected sentences according to configuration
        if self.sort_order in ("document_order", "appearance"):
            ordered_indices = sorted(selected, key=lambda idx: sentences[idx].sentence_id)
        else:
            ordered_indices = list(selected)

        ordered_sentences = [sentences[idx] for idx in ordered_indices]

        # Optional benchmark budget fill (DUC evaluation standard)
        if self.fill_budget_fractional and current_cost < self.budget_limit:
            remaining_budget = self.budget_limit - current_cost
            best_extra_cand = -1
            best_extra_gain = 0.0
            for j in range(n):
                if j in selected:
                    continue
                gain = objective.marginal_gain(j)
                if gain > best_extra_gain:
                    best_extra_gain = gain
                    best_extra_cand = j

            if best_extra_cand != -1:
                cand_s = sentences[best_extra_cand]
                if self.budget_type == "words":
                    words = cand_s.original_text.split()
                    truncated_text = " ".join(words[:remaining_budget])
                    extra_cost = len(truncated_text.split())
                else:
                    encoded = cand_s.original_text.encode("utf-8")[:remaining_budget]
                    truncated_text = encoded.decode("utf-8", errors="ignore")
                    extra_cost = len(truncated_text.encode("utf-8"))

                if truncated_text.strip():
                    trunc_sentence = Sentence(
                        sentence_id=cand_s.sentence_id,
                        document_id=cand_s.document_id,
                        original_text=truncated_text,
                        normalized_text=cand_s.normalized_text,
                        word_count=len(truncated_text.split()),
                        byte_count=len(truncated_text.encode("utf-8")),
                    )
                    ordered_sentences.append(trunc_sentence)
                    ordered_indices.append(best_extra_cand)
                    current_cost += extra_cost

        summary_text = " ".join(s.original_text for s in ordered_sentences)
        runtime = time.perf_counter() - start_time
        breakdown = objective.get_breakdown(selected)

        return SummaryResult(
            selected_indices=ordered_indices,
            ordered_sentences=ordered_sentences,
            summary_text=summary_text,
            total_cost=current_cost,
            budget_limit=self.budget_limit,
            num_sentences=len(ordered_sentences),
            budget_type=self.budget_type,
            breakdown=breakdown,
            runtime_seconds=runtime,
        )

    def _optimize_standard_greedy(
        self,
        objective: CompositeSubmodularObjective,
        sentences: list[Sentence],
        costs: np.ndarray,
    ) -> tuple[list[int], int]:
        """Standard O(K * N) linear scan greedy algorithm."""
        n = len(sentences)
        selected: list[int] = []
        selected_set: set[int] = set()
        current_cost = 0

        while True:
            best_cand = -1
            best_density = -1.0
            best_gain = 0.0

            for j in range(n):
                if j in selected_set:
                    continue

                item_cost = int(costs[j])
                if current_cost + item_cost > self.budget_limit:
                    continue

                marginal_gain = objective.marginal_gain(j)
                if marginal_gain <= 0:
                    continue

                denom = float(item_cost) ** self.cost_power if self.cost_power > 0 else 1.0
                density = marginal_gain / max(denom, 1e-9)
                if density > best_density:
                    best_density = density
                    best_cand = j
                    best_gain = marginal_gain

            if best_cand == -1 or best_gain <= 0:
                break

            selected.append(best_cand)
            selected_set.add(best_cand)
            current_cost += int(costs[best_cand])
            objective.update_state_after_select(best_cand)

        return selected, current_cost

    def _optimize_lazy_greedy(
        self,
        objective: CompositeSubmodularObjective,
        sentences: list[Sentence],
        costs: np.ndarray,
    ) -> tuple[list[int], int]:
        """Minoux (1978) Accelerated / Lazy Greedy using a priority queue.
        
        Submodularity guarantees Delta(j | S_t) <= Delta(j | S_{t-1}).
        By caching upper bounds on density in a max-heap, we avoid recalculating
        marginal gains for candidates that cannot beat the current top candidate.
        """
        n = len(sentences)
        # Heap elements: (-density_upper_bound, candidate_id, round_evaluated)
        heap: list[tuple[float, int, int]] = []

        # Initial pass: compute marginal gain on empty set S = {}
        for j in range(n):
            item_cost = int(costs[j])
            if item_cost <= self.budget_limit:
                gain = objective.marginal_gain(j)
                if gain > 0:
                    denom = float(item_cost) ** self.cost_power if self.cost_power > 0 else 1.0
                    density = gain / max(denom, 1e-9)
                    heapq.heappush(heap, (-density, j, 0))

        selected: list[int] = []
        selected_set: set[int] = set()
        current_cost = 0
        round_idx = 0

        while heap:
            neg_density, cand, eval_round = heapq.heappop(heap)

            if cand in selected_set:
                continue

            item_cost = int(costs[cand])
            if current_cost + item_cost > self.budget_limit:
                continue

            if eval_round == round_idx:
                # Top candidate is fresh at current state S_t; guaranteed to be optimal!
                selected.append(cand)
                selected_set.add(cand)
                current_cost += item_cost
                objective.update_state_after_select(cand)
                round_idx += 1
            else:
                # Re-evaluate marginal gain at current state S_t
                gain = objective.marginal_gain(cand)
                if gain > 0:
                    denom = float(item_cost) ** self.cost_power if self.cost_power > 0 else 1.0
                    new_density = gain / max(denom, 1e-9)
                    heapq.heappush(heap, (-new_density, cand, round_idx))

        return selected, current_cost
        runtime = time.perf_counter() - start_time
        breakdown = objective.get_breakdown(selected)

        return SummaryResult(
            selected_indices=ordered_indices,
            ordered_sentences=ordered_sentences,
            summary_text=summary_text,
            total_cost=current_cost,
            budget_limit=self.budget_limit,
            num_sentences=len(ordered_sentences),
            budget_type=self.budget_type,
            breakdown=breakdown,
            runtime_seconds=runtime,
        )


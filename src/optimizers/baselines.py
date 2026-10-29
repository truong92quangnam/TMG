"""Baseline summarizers for empirical comparison.

Implements classic heuristic and competitive summarization baselines:
1. Lead Baseline (first sentences of cluster documents up to budget)
2. Random Baseline (stochastically chosen sentences with fixed seed)
3. Top-K Query Relevance (highest query term overlap density)
4. Maximal Marginal Relevance (MMR - Carbonell & Goldstein, 1998)
"""

from __future__ import annotations

import time
from typing import Literal
import numpy as np

from src.data.duc_loader import Sentence
from src.optimizers.greedy_knapsack import SummaryResult


class LeadBaselineSummarizer:
    """Selects leading sentences in document presentation order until budget is exhausted."""

    def __init__(
        self,
        budget_limit: int = 100,
        budget_type: Literal["words", "bytes"] = "words",
    ) -> None:
        self.budget_limit = budget_limit
        self.budget_type = budget_type

    def summarize(self, sentences: list[Sentence]) -> SummaryResult:
        start_time = time.perf_counter()
        selected: list[int] = []
        current_cost = 0

        for idx, s in enumerate(sentences):
            cost = s.word_count if self.budget_type == "words" else s.byte_count
            if current_cost + cost <= self.budget_limit:
                selected.append(idx)
                current_cost += cost

        ordered_sentences = [sentences[idx] for idx in selected]
        summary_text = " ".join(s.original_text for s in ordered_sentences)

        return SummaryResult(
            selected_indices=selected,
            ordered_sentences=ordered_sentences,
            summary_text=summary_text,
            total_cost=current_cost,
            budget_limit=self.budget_limit,
            num_sentences=len(ordered_sentences),
            budget_type=self.budget_type,
            breakdown={"method": "lead"},
            runtime_seconds=time.perf_counter() - start_time,
        )


class RandomBaselineSummarizer:
    """Stochastically samples sentences up to budget limit with a fixed seed."""

    def __init__(
        self,
        budget_limit: int = 100,
        budget_type: Literal["words", "bytes"] = "words",
        seed: int = 42,
    ) -> None:
        self.budget_limit = budget_limit
        self.budget_type = budget_type
        self.seed = seed

    def summarize(self, sentences: list[Sentence]) -> SummaryResult:
        start_time = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        n = len(sentences)

        indices = np.arange(n)
        rng.shuffle(indices)

        selected: list[int] = []
        current_cost = 0

        for idx in indices:
            s = sentences[idx]
            cost = s.word_count if self.budget_type == "words" else s.byte_count
            if current_cost + cost <= self.budget_limit:
                selected.append(int(idx))
                current_cost += cost

        ordered_indices = sorted(selected, key=lambda i: sentences[i].sentence_id)
        ordered_sentences = [sentences[idx] for idx in ordered_indices]
        summary_text = " ".join(s.original_text for s in ordered_sentences)

        return SummaryResult(
            selected_indices=ordered_indices,
            ordered_sentences=ordered_sentences,
            summary_text=summary_text,
            total_cost=current_cost,
            budget_limit=self.budget_limit,
            num_sentences=len(ordered_sentences),
            budget_type=self.budget_type,
            breakdown={"method": "random"},
            runtime_seconds=time.perf_counter() - start_time,
        )


class TopKQuerySummarizer:
    """Greedily selects sentences with highest query overlap relevance density."""

    def __init__(
        self,
        budget_limit: int = 100,
        budget_type: Literal["words", "bytes"] = "words",
    ) -> None:
        self.budget_limit = budget_limit
        self.budget_type = budget_type

    def summarize(
        self, sentences: list[Sentence], relevance_scores: np.ndarray
    ) -> SummaryResult:
        start_time = time.perf_counter()
        n = len(sentences)
        costs = np.array(
            [s.word_count if self.budget_type == "words" else s.byte_count for s in sentences],
            dtype=np.float64,
        )

        densities = relevance_scores / np.maximum(costs, 1.0)
        ranked_indices = np.argsort(densities)[::-1]

        selected: list[int] = []
        current_cost = 0

        for idx in ranked_indices:
            cost = int(costs[idx])
            if current_cost + cost <= self.budget_limit and relevance_scores[idx] > 0:
                selected.append(int(idx))
                current_cost += cost

        ordered_indices = sorted(selected, key=lambda i: sentences[i].sentence_id)
        ordered_sentences = [sentences[idx] for idx in ordered_indices]
        summary_text = " ".join(s.original_text for s in ordered_sentences)

        return SummaryResult(
            selected_indices=ordered_indices,
            ordered_sentences=ordered_sentences,
            summary_text=summary_text,
            total_cost=current_cost,
            budget_limit=self.budget_limit,
            num_sentences=len(ordered_sentences),
            budget_type=self.budget_type,
            breakdown={"method": "top_k_query"},
            runtime_seconds=time.perf_counter() - start_time,
        )


class MMRSummarizer:
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998) baseline.

    score(j) = lambda_mmr * sim(j, Q) - (1 - lambda_mmr) * max_{i in S} sim(i, j)
    """

    def __init__(
        self,
        budget_limit: int = 100,
        budget_type: Literal["words", "bytes"] = "words",
        lambda_mmr: float = 0.70,
    ) -> None:
        self.budget_limit = budget_limit
        self.budget_type = budget_type
        self.lambda_mmr = float(lambda_mmr)

    def summarize(
        self,
        sentences: list[Sentence],
        W: np.ndarray,
        query_sim: np.ndarray,
    ) -> SummaryResult:
        start_time = time.perf_counter()
        n = len(sentences)
        costs = [
            s.word_count if self.budget_type == "words" else s.byte_count for s in sentences
        ]

        selected: list[int] = []
        selected_set: set[int] = set()
        current_cost = 0

        while True:
            best_cand = -1
            best_score = -1e9

            for j in range(n):
                if j in selected_set:
                    continue
                cost_j = costs[j]
                if current_cost + cost_j > self.budget_limit:
                    continue

                rel = query_sim[j]
                if selected:
                    redundancy = np.max([W[j, i] for i in selected])
                else:
                    redundancy = 0.0

                mmr_val = self.lambda_mmr * rel - (1.0 - self.lambda_mmr) * redundancy
                density = mmr_val / float(max(1, cost_j))

                if density > best_score:
                    best_score = density
                    best_cand = j

            if best_cand == -1:
                break

            selected.append(best_cand)
            selected_set.add(best_cand)
            current_cost += costs[best_cand]

        ordered_indices = sorted(selected, key=lambda i: sentences[i].sentence_id)
        ordered_sentences = [sentences[idx] for idx in ordered_indices]
        summary_text = " ".join(s.original_text for s in ordered_sentences)

        return SummaryResult(
            selected_indices=ordered_indices,
            ordered_sentences=ordered_sentences,
            summary_text=summary_text,
            total_cost=current_cost,
            budget_limit=self.budget_limit,
            num_sentences=len(ordered_sentences),
            budget_type=self.budget_type,
            breakdown={"method": "mmr", "lambda_mmr": self.lambda_mmr},
            runtime_seconds=time.perf_counter() - start_time,
        )


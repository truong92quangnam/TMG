"""ROUGE evaluation module supporting single and multi-reference setups.

Calculates ROUGE-1, ROUGE-2, and ROUGE-L Precision, Recall, and F1 scores
using the official Google rouge-score implementation.
In multi-reference DUC evaluation, takes the maximum score across reference models
following standard NIST / DUC evaluation practices.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from rouge_score import rouge_scorer


class RougeEvaluator:
    """Computes ROUGE metrics against one or multiple reference summaries."""

    def __init__(
        self,
        metrics: list[str] | None = None,
        use_stemmer: bool = True,
    ) -> None:
        self.metrics = metrics or ["rouge1", "rouge2", "rougeL"]
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=self.metrics,
            use_stemmer=use_stemmer,
        )

    def evaluate_single_pair(
        self, summary: str, reference: str
    ) -> dict[str, float]:
        """Compute scores against a single reference summary."""
        if not summary.strip() or not reference.strip():
            results: dict[str, float] = {}
            for m in self.metrics:
                results[f"{m}_precision"] = 0.0
                results[f"{m}_recall"] = 0.0
                results[f"{m}_f1"] = 0.0
            return results

        scores = self.scorer.score(reference, summary)
        results: dict[str, float] = {}
        for m in self.metrics:
            if m in scores:
                results[f"{m}_precision"] = float(scores[m].precision)
                results[f"{m}_recall"] = float(scores[m].recall)
                results[f"{m}_f1"] = float(scores[m].fmeasure)
        return results

    def evaluate(
        self, summary: str, references: list[str]
    ) -> dict[str, float]:
        """Evaluate summary against multiple reference summaries (taking max recall/f1)."""
        if not references:
            return self.evaluate_single_pair(summary, "")

        all_pair_scores = [self.evaluate_single_pair(summary, ref) for ref in references]

        # Standard DUC / TAC multi-reference aggregation: maximum score across human models
        aggregated: dict[str, float] = {}
        for key in all_pair_scores[0].keys():
            aggregated[key] = float(max(score_dict[key] for score_dict in all_pair_scores))

        return aggregated


"""Evaluation metrics aggregation and CSV reporting module.

Aggregates ROUGE scores, redundancy metrics, query relevance, length budgets,
and runtime telemetry, exporting cluster-level CSVs and formatted markdown summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.evaluation.statistics import StatisticalTester


class EvaluationReporter:
    """Computes cluster telemetry and generates aggregated evaluation reports."""

    @staticmethod
    def compute_redundancy(selected_indices: list[int], W: np.ndarray) -> float:
        """Compute average pairwise similarity among selected sentences.

        Lower redundancy indicates better summary diversity.
        """
        k = len(selected_indices)
        if k <= 1:
            return 0.0

        pair_sims: list[float] = []
        for i in range(k):
            for j in range(i + 1, k):
                pair_sims.append(float(W[selected_indices[i], selected_indices[j]]))

        return float(np.mean(pair_sims)) if pair_sims else 0.0

    @staticmethod
    def compute_query_relevance(
        selected_indices: list[int], relevance_scores: np.ndarray | None
    ) -> float:
        """Compute mean query relevance r_jQ of selected sentences."""
        if not selected_indices or relevance_scores is None or len(relevance_scores) == 0:
            return 0.0
        return float(np.mean([relevance_scores[i] for i in selected_indices]))

    @staticmethod
    def compile_cluster_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert list of cluster execution metrics to pandas DataFrame."""
        return pd.DataFrame(records)

    @staticmethod
    def export_csv(df: pd.DataFrame, output_path: str | Path) -> None:
        """Save evaluation DataFrame to CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @classmethod
    def generate_summary_table(
        cls, df: pd.DataFrame, group_col: str = "method"
    ) -> pd.DataFrame:
        """Aggregate performance across all clusters by method."""
        metrics_to_agg = [
            "rouge1_recall", "rouge1_f1",
            "rouge2_recall", "rouge2_f1",
            "rougeL_recall", "rougeL_f1",
            "redundancy", "query_relevance",
            "word_count", "sentence_count", "runtime_seconds"
        ]
        available_metrics = [m for m in metrics_to_agg if m in df.columns]

        grouped = df.groupby(group_col)[available_metrics].agg(["mean", "std"])
        return grouped


"""Query-Aware Cluster Diversity Objective R_Q(S).

Implementation of the query-focused diversity reward function proposed in Lin & Bilmes (2011):
    q_j = beta * r_j + (1 - beta) * r_jQ
    R_Q(S) = sum_{k=1}^K sqrt( sum_{j in S intersection P_k} q_j )

Automatically falls back to generic R_1(S) when no query is provided (relevance is None).
"""

from __future__ import annotations

import logging
from typing import Literal
import numpy as np

from src.features.relevance import QueryRelevanceCalculator
from src.objectives.diversity import ClusterDiversityObjective

logger = logging.getLogger(__name__)


class QueryAwareClusterDiversityObjective(ClusterDiversityObjective):
    """Query-focused submodular diversity objective combining centrality and query relevance."""

    def __init__(
        self,
        cluster_labels: np.ndarray,
        num_clusters: int,
        centrality_r: np.ndarray,
        relevance_rQ: np.ndarray | None = None,
        beta: float = 0.50,
        concave_func: Literal["sqrt", "log"] = "sqrt",
        sentence_texts: list[str] | None = None,
    ) -> None:
        """Initialize query-aware diversity objective.

        Args:
            cluster_labels: Partition labels of shape [N].
            num_clusters: Number of clusters K.
            centrality_r: Centrality array r_j of shape [N].
            relevance_rQ: Query overlap relevance r_jQ in [0, 1] of shape [N] or None.
            beta: Balance parameter in [0, 1].
            concave_func: Concave function name ("sqrt" or "log").
            sentence_texts: Optional original texts for logging top sentences.
        """
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.centrality_r = centrality_r
        self.relevance_rQ = relevance_rQ

        # Compute combined reward q_j
        combined_q = QueryRelevanceCalculator.combine_rewards(
            centrality_r=centrality_r,
            relevance_rQ=relevance_rQ,
            beta=self.beta,
        )

        if sentence_texts is not None and len(sentence_texts) > 0:
            QueryRelevanceCalculator.log_top_sentences(
                sentence_texts=sentence_texts,
                r_centrality=centrality_r,
                r_relevance=relevance_rQ,
                q_combined=combined_q,
                top_k=10,
            )

        super().__init__(
            cluster_labels=cluster_labels,
            num_clusters=num_clusters,
            rewards=combined_q,
            concave_func=concave_func,
        )


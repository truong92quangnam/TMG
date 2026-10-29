"""Query relevance and combined centrality-relevance computation.

Calculates query term overlap r_jQ and the combined reward:
    q_j = beta * r_j + (1 - beta) * r_jQ
supporting unigram/bigram overlap and logging top-ranked sentences.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class QueryRelevanceCalculator:
    """Calculates query overlap relevance and combined query-centrality rewards."""

    @staticmethod
    def compute_overlap_relevance(
        sentence_terms_list: list[set[str]],
        query_terms: set[str],
    ) -> np.ndarray:
        """Compute normalized term overlap between each sentence and the query.

        Args:
            sentence_terms_list: List of sets of unigrams/bigrams for each sentence.
            query_terms: Set of unigrams/bigrams in the preprocessed query.

        Returns:
            np.ndarray: Relevance scores in [0, 1] of shape [N].
        """
        n = len(sentence_terms_list)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        if not query_terms:
            return np.zeros(n, dtype=np.float64)

        scores = np.zeros(n, dtype=np.float64)
        for i, s_terms in enumerate(sentence_terms_list):
            if not s_terms:
                continue
            # Intersection count normalized by query size (or min terms)
            overlap = len(s_terms & query_terms)
            scores[i] = overlap

        # Normalize to [0, 1]
        max_score = np.max(scores)
        if max_score > 0:
            scores = scores / max_score

        return scores

    @staticmethod
    def combine_rewards(
        centrality_r: np.ndarray,
        relevance_rQ: np.ndarray | None,
        beta: float = 0.5,
    ) -> np.ndarray:
        """Combine generic centrality r_j and query relevance r_jQ.

        q_j = beta * r_j + (1 - beta) * r_jQ
        """
        if relevance_rQ is None or len(relevance_rQ) == 0:
            return centrality_r.copy()

        beta = float(np.clip(beta, 0.0, 1.0))
        # Ensure centrality is normalized to comparable scale [0, 1]
        max_r = np.max(centrality_r) if np.max(centrality_r) > 0 else 1.0
        norm_r = centrality_r / max_r

        q = beta * norm_r + (1.0 - beta) * relevance_rQ
        return np.maximum(q, 0.0)

    @staticmethod
    def log_top_sentences(
        sentence_texts: list[str],
        r_centrality: np.ndarray,
        r_relevance: np.ndarray | None,
        q_combined: np.ndarray,
        top_k: int = 10,
    ) -> None:
        """Log top sentences ranked by centrality, query relevance, and combined reward."""
        k = min(top_k, len(sentence_texts))
        if k == 0:
            return

        logger.info(f"--- Top {k} sentences by Centrality r_j ---")
        top_cent = np.argsort(r_centrality)[::-1][:k]
        for rank, idx in enumerate(top_cent, 1):
            logger.info(
                f"  [{rank}] Score: {r_centrality[idx]:.4f} | {sentence_texts[idx][:90]}..."
            )

        if r_relevance is not None:
            logger.info(f"--- Top {k} sentences by Query Relevance r_jQ ---")
            top_rel = np.argsort(r_relevance)[::-1][:k]
            for rank, idx in enumerate(top_rel, 1):
                logger.info(
                    f"  [{rank}] Score: {r_relevance[idx]:.4f} | {sentence_texts[idx][:90]}..."
                )

        logger.info(f"--- Top {k} sentences by Combined Reward q_j ---")
        top_comb = np.argsort(q_combined)[::-1][:k]
        for rank, idx in enumerate(top_comb, 1):
            logger.info(
                f"  [{rank}] Score: {q_combined[idx]:.4f} | {sentence_texts[idx][:90]}..."
            )


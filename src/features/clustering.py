"""Sentence clustering via K-Means.

Partitions candidate sentences into semantic clusters P_1, ..., P_K
to evaluate submodular diversity functions as described in Lin & Bilmes (2011).
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class SentenceClusterer:
    """Partitions sentences into K clusters using K-Means with reproducible seeding."""

    def __init__(
        self,
        k_ratio: float = 0.10,
        random_seed: int = 42,
    ) -> None:
        self.k_ratio = k_ratio
        self.random_seed = random_seed

    def determine_k(self, n_samples: int) -> int:
        """Determine number of clusters: K = max(2, min(n_samples, round(k_ratio * N)))."""
        if n_samples <= 2:
            return max(1, n_samples)
        k = max(2, int(round(self.k_ratio * n_samples)))
        return min(k, n_samples)

    def cluster(
        self,
        features: csr_matrix | np.ndarray,
        k: int | None = None,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Cluster sentence representations.

        Args:
            features: TF-IDF feature matrix of shape [N, V]
            k: Explicit number of clusters, or None to use k_ratio.

        Returns:
            tuple:
                - labels: array of cluster index for each sentence of shape [N]
                - partitions: list of length K, where element k contains sentence indices in cluster k
        """
        n_samples = features.shape[0]
        if n_samples == 0:
            return np.zeros(0, dtype=int), []
        if n_samples == 1:
            return np.zeros(1, dtype=int), [[0]]

        num_clusters = k if k is not None else self.determine_k(n_samples)

        kmeans = KMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=self.random_seed,
        )

        labels = kmeans.fit_predict(features)

        # Build explicit partition index lists
        partitions: list[list[int]] = [[] for _ in range(num_clusters)]
        for idx, cluster_label in enumerate(labels):
            partitions[cluster_label].append(idx)

        logger.debug(
            f"Clustered {n_samples} sentences into {num_clusters} clusters (k_ratio={self.k_ratio})."
        )
        return labels, partitions


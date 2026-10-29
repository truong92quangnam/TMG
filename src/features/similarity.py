"""Pairwise sentence similarity and semantic similarity computation.

Provides:
1. Lexical Cosine Similarity on TF-IDF features.
2. Latent Semantic Analysis (LSA) Cosine Similarity via TruncatedSVD:
   Projects sentence representations into a dense latent semantic concept space,
   capturing semantic relatedness and synonymy without requiring heavy external models or LLMs.
3. Hybrid Similarity: Convex combination of lexical and latent semantic similarities.
4. Symmetric, non-negative matrix validation with diagonal zeroing.
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityMatrixBuilder:
    """Constructs and validates pairwise sentence similarity matrix W and query similarity vector."""

    @staticmethod
    def compute(
        feature_matrix: csr_matrix | np.ndarray,
        keep_diagonal: bool = False,
        clip_negative: bool = True,
        method: Literal["tfidf", "lsa", "hybrid"] = "tfidf",
        n_components: int = 50,
        hybrid_weight: float = 0.5,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Compute symmetric similarity matrix W from feature vectors.

        Args:
            feature_matrix: Feature matrix of shape [N, vocab_size]
            keep_diagonal: If False, set W[i, i] = 0.0
            clip_negative: If True, enforce W[i, j] >= 0.0
            method: "tfidf" (lexical cosine), "lsa" (latent semantic analysis),
                    or "hybrid" (convex combination)
            n_components: Number of latent semantic dimensions for LSA
            hybrid_weight: Weight for LSA similarity in hybrid mode (1 - weight for TF-IDF)
            random_seed: Random state for TruncatedSVD

        Returns:
            np.ndarray: [N, N] symmetric non-negative similarity matrix
        """
        n_samples = feature_matrix.shape[0]
        if n_samples == 0:
            return np.zeros((0, 0), dtype=np.float64)

        if method == "tfidf":
            sim_matrix = cosine_similarity(feature_matrix, dense_output=True)

        elif method == "lsa":
            sim_matrix = SimilarityMatrixBuilder.compute_lsa_similarity(
                feature_matrix=feature_matrix,
                n_components=n_components,
                random_seed=random_seed,
            )

        elif method == "hybrid":
            w_tfidf = cosine_similarity(feature_matrix, dense_output=True)
            w_lsa = SimilarityMatrixBuilder.compute_lsa_similarity(
                feature_matrix=feature_matrix,
                n_components=n_components,
                random_seed=random_seed,
            )
            alpha_sem = float(np.clip(hybrid_weight, 0.0, 1.0))
            sim_matrix = alpha_sem * w_lsa + (1.0 - alpha_sem) * w_tfidf

        else:
            raise ValueError(f"Unknown similarity method: {method}. Choose from 'tfidf', 'lsa', 'hybrid'.")

        # Clip negative numerical noise
        if clip_negative:
            np.maximum(sim_matrix, 0.0, out=sim_matrix)

        # Enforce exact symmetry
        sim_matrix = 0.5 * (sim_matrix + sim_matrix.T)

        # Handle diagonal self-similarity
        if not keep_diagonal:
            np.fill_diagonal(sim_matrix, 0.0)

        return sim_matrix

    @staticmethod
    def compute_lsa_similarity(
        feature_matrix: csr_matrix | np.ndarray,
        n_components: int = 50,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Compute semantic similarity via Latent Semantic Analysis (TruncatedSVD).

        LSA maps high-dimensional sparse lexical features into a compact latent semantic space,
        where sentences sharing concepts or co-occurring terms have high cosine similarity
        even if they do not share identical surface vocabulary.
        """
        n_samples, n_features = feature_matrix.shape
        if n_samples <= 1:
            return np.ones((n_samples, n_samples), dtype=np.float64)

        # Number of components must be strictly less than min(n_samples, n_features)
        max_valid_k = min(n_samples - 1, n_features - 1)
        k = max(1, min(n_components, max_valid_k))

        svd = TruncatedSVD(n_components=k, random_state=random_seed)
        latent_vectors = svd.fit_transform(feature_matrix)

        # L2-normalize latent semantic representations
        norms = np.linalg.norm(latent_vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized_latent = latent_vectors / norms

        sim_matrix = np.dot(normalized_latent, normalized_latent.T)
        return sim_matrix

    @staticmethod
    def compute_query_similarity(
        feature_matrix: csr_matrix | np.ndarray,
        query_vector: csr_matrix | np.ndarray | None,
        method: Literal["tfidf", "lsa", "hybrid"] = "tfidf",
        n_components: int = 50,
        hybrid_weight: float = 0.5,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Compute semantic similarity vector between each candidate sentence and the query.

        Returns:
            np.ndarray: [N] non-negative similarity vector in [0, 1].
        """
        n_samples = feature_matrix.shape[0]
        if query_vector is None or n_samples == 0:
            return np.zeros(n_samples, dtype=np.float64)

        if method == "tfidf":
            sim = cosine_similarity(feature_matrix, query_vector).ravel()

        elif method in ("lsa", "hybrid"):
            n_samples, n_features = feature_matrix.shape
            max_valid_k = min(n_samples - 1, n_features - 1)
            k = max(1, min(n_components, max_valid_k)) if max_valid_k >= 1 else 1

            svd = TruncatedSVD(n_components=k, random_state=random_seed)
            svd.fit(feature_matrix)

            doc_latent = svd.transform(feature_matrix)
            query_latent = svd.transform(query_vector)

            # L2 normalize
            d_norm = doc_latent / np.maximum(np.linalg.norm(doc_latent, axis=1, keepdims=True), 1e-9)
            q_norm = query_latent / np.maximum(np.linalg.norm(query_latent, axis=1, keepdims=True), 1e-9)
            lsa_sim = np.dot(d_norm, q_norm.T).ravel()

            if method == "lsa":
                sim = lsa_sim
            else:
                tfidf_sim = cosine_similarity(feature_matrix, query_vector).ravel()
                alpha_sem = float(np.clip(hybrid_weight, 0.0, 1.0))
                sim = alpha_sem * lsa_sim + (1.0 - alpha_sem) * tfidf_sim
        else:
            sim = cosine_similarity(feature_matrix, query_vector).ravel()

        sim = np.maximum(sim, 0.0)
        # Normalize to max 1.0 if not all zeros
        max_val = np.max(sim)
        if max_val > 1.0:
            sim = sim / max_val
        return sim

    @staticmethod
    def validate_properties(W: np.ndarray, atol: float = 1e-6) -> bool:
        """Validate that W is square, symmetric, and non-negative."""
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            return False
        if np.any(W < -atol):
            return False
        if not np.allclose(W, W.T, atol=atol):
            return False
        return True

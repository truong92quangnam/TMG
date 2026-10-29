"""Unit tests for Semantic Similarity (LSA and Hybrid) without external AI models."""

import numpy as np
from scipy.sparse import csr_matrix
from src.features.similarity import SimilarityMatrixBuilder


def test_lsa_similarity_properties():
    """Verify LSA similarity produces symmetric, non-negative, zero-diagonal matrix."""
    np.random.seed(42)
    # 8 sentences with 15 vocabulary terms
    feature_matrix = np.random.uniform(0.0, 1.0, size=(8, 15))
    # sparsify
    feature_matrix[feature_matrix < 0.6] = 0.0
    csr_X = csr_matrix(feature_matrix)

    W = SimilarityMatrixBuilder.compute(
        feature_matrix=csr_X,
        keep_diagonal=False,
        clip_negative=True,
        method="lsa",
        n_components=4,
        random_seed=42,
    )

    assert W.shape == (8, 8)
    assert np.allclose(W, W.T, atol=1e-7), "LSA similarity must be symmetric."
    assert np.all(W >= 0.0), "LSA similarity must be non-negative."
    assert np.all(np.diag(W) == 0.0), "Diagonal must be 0."
    assert SimilarityMatrixBuilder.validate_properties(W)


def test_hybrid_similarity():
    """Verify hybrid similarity combines LSA and TF-IDF."""
    feature_matrix = np.array([
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.1],
        [0.0, 0.1, 0.9],
        [0.1, 0.0, 0.8],
    ])

    W_hybrid = SimilarityMatrixBuilder.compute(
        feature_matrix=feature_matrix,
        keep_diagonal=False,
        clip_negative=True,
        method="hybrid",
        n_components=2,
        hybrid_weight=0.5,
    )

    assert W_hybrid.shape == (4, 4)
    assert np.allclose(W_hybrid, W_hybrid.T, atol=1e-7)
    assert np.all(W_hybrid >= 0.0)


def test_query_semantic_similarity():
    """Verify query-to-sentence semantic similarity vector."""
    feature_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    query_vector = np.array([[0.9, 0.1, 0.0]])

    sim = SimilarityMatrixBuilder.compute_query_similarity(
        feature_matrix=feature_matrix,
        query_vector=query_vector,
        method="lsa",
        n_components=2,
    )

    assert len(sim) == 3
    assert np.all(sim >= 0.0)
    assert np.all(sim <= 1.0001)
    # Sentence 0 has highest overlap with query
    assert sim[0] >= sim[1]


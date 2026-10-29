"""Unit tests for sentence similarity computation."""

import numpy as np
from scipy.sparse import csr_matrix
from src.features.similarity import SimilarityMatrixBuilder


def test_similarity_matrix_properties():
    """Test symmetry, non-negativity, and diagonal zeroing of similarity matrix W."""
    # Synthetic TF-IDF feature vectors
    X = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.7, 0.3],
        [0.8, 0.1, 0.1],
        [0.0, 0.0, 1.0],
    ])
    csr_X = csr_matrix(X)

    W = SimilarityMatrixBuilder.compute(csr_X, keep_diagonal=False, clip_negative=True)

    # 1. Check shape
    assert W.shape == (4, 4)

    # 2. Check exact symmetry
    assert np.allclose(W, W.T, atol=1e-7)

    # 3. Check non-negativity
    assert np.all(W >= 0.0)

    # 4. Check diagonal is zero
    assert np.all(np.diag(W) == 0.0)

    # 5. Check validator function
    assert SimilarityMatrixBuilder.validate_properties(W)


def test_similarity_keep_diagonal():
    """Test keep_diagonal=True preserves self-similarity."""
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    W = SimilarityMatrixBuilder.compute(X, keep_diagonal=True)
    assert np.allclose(np.diag(W), 1.0)


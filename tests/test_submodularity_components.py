"""Unit tests verifying mathematical submodularity and incremental updates."""

import numpy as np
from src.features.similarity import SimilarityMatrixBuilder
from src.features.centrality import CentralityCalculator
from src.objectives.coverage import TruncatedCoverageObjective
from src.objectives.diversity import ClusterDiversityObjective
from src.objectives.composite import CompositeSubmodularObjective


def build_test_environment():
    """Build reproducible test data with 6 sentences and 2 clusters."""
    np.random.seed(42)
    # 6 items in 4D space
    features = np.random.uniform(0.1, 1.0, size=(6, 4))
    W = SimilarityMatrixBuilder.compute(features, keep_diagonal=False)
    r = CentralityCalculator.compute(W)

    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    num_clusters = 2
    return W, r, labels, num_clusters


def test_coverage_submodularity_diminishing_returns():
    """Verify diminishing returns for L_1(S):

    For A subset B and e not in B:
        L_1(A union {e}) - L_1(A) >= L_1(B union {e}) - L_1(B)
    """
    W, _, _, _ = build_test_environment()
    cov_obj = TruncatedCoverageObjective(W=W, alpha=0.30)

    # A = {0}, B = {0, 1}, candidate e = 2
    A = {0}
    B = {0, 1}
    e = 2

    gain_A = cov_obj.evaluate(A | {e}) - cov_obj.evaluate(A)
    gain_B = cov_obj.evaluate(B | {e}) - cov_obj.evaluate(B)

    # Submodularity implies gain_A >= gain_B
    assert gain_A >= gain_B - 1e-9


def test_coverage_incremental_consistency():
    """Verify that incremental marginal_gain matches full evaluate() difference."""
    W, _, _, _ = build_test_environment()
    cov_obj = TruncatedCoverageObjective(W=W, alpha=0.30)

    # Select item 0
    cov_obj.update_state_after_select(0)
    # Select item 1
    cov_obj.update_state_after_select(1)

    # Check candidate 2
    inc_gain = cov_obj.marginal_gain(2)
    full_gain = cov_obj.evaluate({0, 1, 2}) - cov_obj.evaluate({0, 1})

    assert np.isclose(inc_gain, full_gain, atol=1e-7)


def test_diversity_submodularity_diminishing_returns():
    """Verify diminishing returns for R_1(S)."""
    _, r, labels, num_clusters = build_test_environment()
    div_obj = ClusterDiversityObjective(
        cluster_labels=labels,
        num_clusters=num_clusters,
        rewards=r,
        concave_func="sqrt",
    )

    A = {0}
    B = {0, 1}
    e = 2  # item 2 belongs to same cluster 0

    gain_A = div_obj.evaluate(A | {e}) - div_obj.evaluate(A)
    gain_B = div_obj.evaluate(B | {e}) - div_obj.evaluate(B)

    # Concave function on same cluster must exhibit diminishing returns
    assert gain_A >= gain_B - 1e-9


def test_diversity_incremental_consistency():
    """Verify incremental marginal gain matches evaluate() difference."""
    _, r, labels, num_clusters = build_test_environment()
    div_obj = ClusterDiversityObjective(
        cluster_labels=labels,
        num_clusters=num_clusters,
        rewards=r,
        concave_func="sqrt",
    )

    div_obj.update_state_after_select(0)
    inc_gain = div_obj.marginal_gain(1)
    full_gain = div_obj.evaluate({0, 1}) - div_obj.evaluate({0})

    assert np.isclose(inc_gain, full_gain, atol=1e-7)


def test_composite_monotonicity():
    """Verify monotonicity: F(S union {e}) >= F(S) for non-negative weights."""
    W, r, labels, num_clusters = build_test_environment()
    cov_obj = TruncatedCoverageObjective(W=W, alpha=0.25)
    div_obj = ClusterDiversityObjective(labels, num_clusters, r)

    comp_obj = CompositeSubmodularObjective(cov_obj, [(3.0, div_obj)])

    S = {0, 3}
    e = 1
    val_S = comp_obj.evaluate(S)
    val_Se = comp_obj.evaluate(S | {e})

    assert val_Se >= val_S - 1e-9


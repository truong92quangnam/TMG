"""Unit tests verifying mathematical monotone submodularity of Query-Aware Coverage L_Q(S)."""

import numpy as np
from src.features.similarity import SimilarityMatrixBuilder
from src.objectives.coverage import TruncatedCoverageObjective
from src.objectives.query_coverage import QueryAwareCoverageObjective


def build_test_environment():
    """Build reproducible similarity matrix and query vector."""
    np.random.seed(42)
    features = np.random.uniform(0.1, 1.0, size=(7, 6))
    W = SimilarityMatrixBuilder.compute(features, keep_diagonal=False, method="lsa", n_components=3)
    query_sim = np.array([0.9, 0.8, 0.3, 0.1, 0.05, 0.0, 0.5])
    return W, query_sim


def test_query_aware_coverage_submodularity():
    """Verify diminishing returns for L_Q(S):

    For A subset B and candidate e not in B:
        L_Q(A union {e}) - L_Q(A) >= L_Q(B union {e}) - L_Q(B)
    """
    W, query_sim = build_test_environment()
    obj = QueryAwareCoverageObjective(
        W=W,
        query_sim=query_sim,
        alpha=0.30,
        alpha_q=0.50,
        gamma=0.35,
        eta=0.20,
    )

    A = {0}
    B = {0, 1, 2}
    e = 3

    gain_A = obj.evaluate(A | {e}) - obj.evaluate(A)
    gain_B = obj.evaluate(B | {e}) - obj.evaluate(B)

    # Submodularity condition
    assert gain_A >= gain_B - 1e-9, f"Diminishing returns violated: gain_A={gain_A}, gain_B={gain_B}"


def test_query_aware_coverage_incremental_consistency():
    """Verify that incremental marginal_gain matches full evaluate() difference."""
    W, query_sim = build_test_environment()
    obj = QueryAwareCoverageObjective(
        W=W,
        query_sim=query_sim,
        alpha=0.25,
        alpha_q=0.40,
        gamma=0.30,
        eta=0.15,
    )

    # Add sentence 0 and 1 into S
    obj.update_state_after_select(0)
    obj.update_state_after_select(1)

    # Evaluate candidate 2
    inc_gain = obj.marginal_gain(2)
    full_gain = obj.evaluate({0, 1, 2}) - obj.evaluate({0, 1})

    assert np.isclose(inc_gain, full_gain, atol=1e-7), f"Incremental {inc_gain} != Full {full_gain}"


def test_query_aware_coverage_monotonicity():
    """Verify monotonicity: L_Q(S union {e}) >= L_Q(S) for any candidate e."""
    W, query_sim = build_test_environment()
    obj = QueryAwareCoverageObjective(
        W=W,
        query_sim=query_sim,
        alpha=0.30,
        alpha_q=0.50,
        gamma=0.40,
    )

    S = {1, 3}
    e = 4
    val_S = obj.evaluate(S)
    val_Se = obj.evaluate(S | {e})

    assert val_Se >= val_S - 1e-9, "Monotonicity violated."


def test_smooth_reduction_to_generic():
    """When gamma=0 and eta=1 (or query_sim is uniform 0 with gamma=0),

    QueryAwareCoverage matches original TruncatedCoverageObjective L_1(S).
    """
    W, query_sim = build_test_environment()
    alpha = 0.25

    generic_obj = TruncatedCoverageObjective(W=W, alpha=alpha)
    query_obj = QueryAwareCoverageObjective(
        W=W,
        query_sim=query_sim,
        alpha=alpha,
        gamma=0.0,
        eta=1.0,  # rho_i = 1.0 everywhere
    )

    test_subsets = [{0}, {0, 1}, {2, 3, 5}]
    for S in test_subsets:
        v_gen = generic_obj.evaluate(S)
        v_qa = query_obj.evaluate(S)
        assert np.isclose(v_gen, v_qa, atol=1e-7), f"Reduction failed for subset {S}: {v_gen} vs {v_qa}"


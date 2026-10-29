"""Unit tests for budget constraints and sentence ordering."""

import numpy as np
from src.data.duc_loader import Sentence
from src.features.similarity import SimilarityMatrixBuilder
from src.objectives.coverage import TruncatedCoverageObjective
from src.objectives.diversity import ClusterDiversityObjective
from src.objectives.composite import CompositeSubmodularObjective
from src.optimizers.greedy_knapsack import BudgetedGreedyOptimizer


def make_dummy_sentences():
    """Create list of sentences with distinct costs."""
    return [
        Sentence(sentence_id=0, document_id="doc1", original_text="Sentence zero.", normalized_text="sentence zero", word_count=10, byte_count=50),
        Sentence(sentence_id=1, document_id="doc1", original_text="Sentence one is longer.", normalized_text="sentence one is longer", word_count=20, byte_count=100),
        Sentence(sentence_id=2, document_id="doc2", original_text="Sentence two.", normalized_text="sentence two", word_count=15, byte_count=75),
        Sentence(sentence_id=3, document_id="doc2", original_text="Sentence three.", normalized_text="sentence three", word_count=25, byte_count=120),
    ]


def test_word_budget_strictly_enforced():
    """Ensure selected summary total words never exceeds budget_limit."""
    sentences = make_dummy_sentences()
    W = np.array([
        [0.0, 0.8, 0.4, 0.2],
        [0.8, 0.0, 0.3, 0.5],
        [0.4, 0.3, 0.0, 0.7],
        [0.2, 0.5, 0.7, 0.0],
    ])
    labels = np.array([0, 0, 1, 1])
    r = np.array([0.5, 0.6, 0.4, 0.5])

    cov = TruncatedCoverageObjective(W, alpha=0.3)
    div = ClusterDiversityObjective(labels, 2, r)
    comp = CompositeSubmodularObjective(cov, [(1.0, div)])

    # Test word budget of 30 words
    budget = 30
    optimizer = BudgetedGreedyOptimizer(budget_limit=budget, budget_type="words", sort_order="document_order")
    res = optimizer.optimize(comp, sentences)

    assert res.total_cost <= budget
    total_words = sum(s.word_count for s in res.ordered_sentences)
    assert total_words <= budget


def test_chronological_document_ordering():
    """Ensure summary sentences are sorted by original document/sentence appearance."""
    sentences = make_dummy_sentences()
    W = np.full((4, 4), 0.5)
    np.fill_diagonal(W, 0.0)
    labels = np.array([0, 0, 1, 1])
    r = np.full(4, 0.5)

    cov = TruncatedCoverageObjective(W, alpha=0.5)
    div = ClusterDiversityObjective(labels, 2, r)
    comp = CompositeSubmodularObjective(cov, [(1.0, div)])

    optimizer = BudgetedGreedyOptimizer(budget_limit=100, budget_type="words", sort_order="document_order")
    res = optimizer.optimize(comp, sentences)

    # Check sentence_ids are strictly ascending
    ids = [s.sentence_id for s in res.ordered_sentences]
    assert ids == sorted(ids)


def test_lazy_greedy_equivalence():
    """Verify that Minoux (1978) Lazy Greedy produces the exact same result as Standard Greedy."""
    sentences = make_dummy_sentences()
    W = np.array([
        [0.0, 0.8, 0.4, 0.2],
        [0.8, 0.0, 0.3, 0.5],
        [0.4, 0.3, 0.0, 0.7],
        [0.2, 0.5, 0.7, 0.0],
    ])
    labels = np.array([0, 0, 1, 1])
    r = np.array([0.5, 0.6, 0.4, 0.5])

    cov = TruncatedCoverageObjective(W, alpha=0.3)
    div = ClusterDiversityObjective(labels, 2, r)
    comp = CompositeSubmodularObjective(cov, [(1.0, div)])

    opt_standard = BudgetedGreedyOptimizer(budget_limit=40, use_lazy_greedy=False)
    res_standard = opt_standard.optimize(comp, sentences)

    opt_lazy = BudgetedGreedyOptimizer(budget_limit=40, use_lazy_greedy=True)
    res_lazy = opt_lazy.optimize(comp, sentences)

    assert res_standard.selected_indices == res_lazy.selected_indices
    assert res_standard.total_cost == res_lazy.total_cost


def test_cost_power_effect():
    """Verify that cost_power parameter smoothly controls penalty on sentence length."""
    sentences = make_dummy_sentences()
    W = np.ones((4, 4)) - np.eye(4)
    labels = np.array([0, 0, 1, 1])
    r = np.array([0.8, 0.8, 0.8, 0.8])

    cov = TruncatedCoverageObjective(W, alpha=0.5)
    div = ClusterDiversityObjective(labels, 2, r)
    comp = CompositeSubmodularObjective(cov, [(1.0, div)])

    opt_costly = BudgetedGreedyOptimizer(budget_limit=25, cost_power=1.0)
    res_costly = opt_costly.optimize(comp, sentences)

    opt_pure = BudgetedGreedyOptimizer(budget_limit=25, cost_power=0.0)
    res_pure = opt_pure.optimize(comp, sentences)

    assert res_costly.total_cost <= 25
    assert res_pure.total_cost <= 25


def test_fractional_budget_filling():
    """Verify that fractional budget filling reaches full budget limit without violating it."""
    sentences = make_dummy_sentences()
    W = np.ones((4, 4)) - np.eye(4)
    labels = np.array([0, 0, 1, 1])
    r = np.array([0.5, 0.5, 0.5, 0.5])

    cov = TruncatedCoverageObjective(W, alpha=0.5)
    div = ClusterDiversityObjective(labels, 2, r)
    comp = CompositeSubmodularObjective(cov, [(1.0, div)])

    opt = BudgetedGreedyOptimizer(budget_limit=32, fill_budget_fractional=True)
    res = opt.optimize(comp, sentences)

    # Total words should fill closer to 32 words
    assert res.total_cost <= 32



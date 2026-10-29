"""Generic multi-document summarization experiment runner.

Executes generic summarization on DUC document clusters using monotone submodular
maximization:
    F(S) = L_1(S) + lambda * R_1(S)
Evaluates against Lead, Random, Coverage-only, and Diversity-only baselines.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data.duc_loader import DUCDatasetLoader
from src.preprocessing.sentence_splitter import SentenceSplitter
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.vectorizer import ClusterVectorizer
from src.features.similarity import SimilarityMatrixBuilder
from src.features.centrality import CentralityCalculator
from src.features.clustering import SentenceClusterer
from src.objectives.coverage import TruncatedCoverageObjective
from src.objectives.diversity import ClusterDiversityObjective
from src.objectives.composite import CompositeSubmodularObjective
from src.optimizers.greedy_knapsack import BudgetedGreedyOptimizer
from src.optimizers.baselines import LeadBaselineSummarizer, RandomBaselineSummarizer
from src.evaluation.rouge_eval import RougeEvaluator
from src.evaluation.statistics import StatisticalTester
from src.evaluation.reports import EvaluationReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_generic")


def run_generic_cluster(
    cluster,
    config: ExperimentConfig,
    splitter: SentenceSplitter,
    normalizer: TextNormalizer,
    evaluator: RougeEvaluator,
) -> list[dict]:
    """Execute proposed method and baselines on a single cluster."""
    sentences = cluster.sentences
    n = len(sentences)
    if n == 0:
        logger.warning(f"Cluster {cluster.cluster_id} has no valid sentences.")
        return []

    # 1. TF-IDF vectorization
    vectorizer = ClusterVectorizer(
        normalizer=normalizer,
        ngram_range=config.preprocessing.ngram_range,
        sublinear_tf=config.preprocessing.sublinear_tf,
    )
    raw_texts = [s.original_text for s in sentences]
    tfidf_matrix = vectorizer.fit_transform(raw_texts)

    # 2. Similarity matrix W (supports lexical TF-IDF, latent semantic LSA, or hybrid)
    W = SimilarityMatrixBuilder.compute(
        feature_matrix=tfidf_matrix,
        keep_diagonal=config.similarity.keep_diagonal,
        clip_negative=config.similarity.clip_negative,
        method=config.similarity.method,
        n_components=config.similarity.lsa_components,
        hybrid_weight=config.similarity.hybrid_weight,
        random_seed=config.seed,
    )

    # 3. Singleton centrality r_j
    r = CentralityCalculator.compute(W)

    # 4. K-means sentence clustering
    clusterer = SentenceClusterer(
        k_ratio=config.objective.k_ratio,
        random_seed=config.seed,
    )
    labels, partitions = clusterer.cluster(tfidf_matrix)
    num_clusters = len(partitions)

    # 5. Build Submodular Objectives
    cov_obj = TruncatedCoverageObjective(W=W, alpha=config.objective.alpha)
    div_obj = ClusterDiversityObjective(
        cluster_labels=labels,
        num_clusters=num_clusters,
        rewards=r,
        concave_func=config.objective.concave_func,
    )

    # Setup Greedy Optimizer
    optimizer = BudgetedGreedyOptimizer(
        budget_limit=config.optimizer.budget_limit,
        budget_type=config.optimizer.budget_type,
        sort_order=config.optimizer.sort_order,
        cost_power=config.optimizer.cost_power,
        use_lazy_greedy=config.optimizer.use_lazy_greedy,
        fill_budget_fractional=config.optimizer.fill_budget_fractional,
        modified_greedy_guarantee=config.optimizer.modified_greedy_guarantee,
    )

    cluster_records = []

    # Method 1: Proposed Lin & Bilmes F(S) = L_1(S) + lambda * R_1(S)
    prop_obj = CompositeSubmodularObjective(
        coverage_obj=cov_obj,
        diversity_objs=[(config.objective.lambda_weight, div_obj)],
    )
    res_prop = optimizer.optimize(prop_obj, sentences)
    score_prop = evaluator.evaluate(res_prop.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "proposed_lin_bilmes",
        "summary_text": res_prop.summary_text,
        "word_count": len(res_prop.summary_text.split()),
        "sentence_count": res_prop.num_sentences,
        "runtime_seconds": res_prop.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_prop.selected_indices, W),
        "query_relevance": 0.0,
        **score_prop,
    })

    # Method 2: Coverage Only L_1(S) (lambda = 0)
    cov_only_obj = CompositeSubmodularObjective(
        coverage_obj=cov_obj,
        diversity_objs=[(0.0, div_obj)],
    )
    res_cov = optimizer.optimize(cov_only_obj, sentences)
    score_cov = evaluator.evaluate(res_cov.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "coverage_only",
        "summary_text": res_cov.summary_text,
        "word_count": len(res_cov.summary_text.split()),
        "sentence_count": res_cov.num_sentences,
        "runtime_seconds": res_cov.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_cov.selected_indices, W),
        "query_relevance": 0.0,
        **score_cov,
    })

    # Method 3: Diversity Only R_1(S) (alpha = 0 => cov=0, weight=1.0)
    zero_cov_obj = TruncatedCoverageObjective(W=W, alpha=0.0)
    div_only_obj = CompositeSubmodularObjective(
        coverage_obj=zero_cov_obj,
        diversity_objs=[(1.0, div_obj)],
    )
    res_div = optimizer.optimize(div_only_obj, sentences)
    score_div = evaluator.evaluate(res_div.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "diversity_only",
        "summary_text": res_div.summary_text,
        "word_count": len(res_div.summary_text.split()),
        "sentence_count": res_div.num_sentences,
        "runtime_seconds": res_div.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_div.selected_indices, W),
        "query_relevance": 0.0,
        **score_div,
    })

    # Method 4: Lead Baseline
    lead_summarizer = LeadBaselineSummarizer(
        budget_limit=config.optimizer.budget_limit,
        budget_type=config.optimizer.budget_type,
    )
    res_lead = lead_summarizer.summarize(sentences)
    score_lead = evaluator.evaluate(res_lead.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "lead_baseline",
        "summary_text": res_lead.summary_text,
        "word_count": len(res_lead.summary_text.split()),
        "sentence_count": res_lead.num_sentences,
        "runtime_seconds": res_lead.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_lead.selected_indices, W),
        "query_relevance": 0.0,
        **score_lead,
    })

    # Method 5: Random Baseline
    rand_summarizer = RandomBaselineSummarizer(
        budget_limit=config.optimizer.budget_limit,
        budget_type=config.optimizer.budget_type,
        seed=config.seed,
    )
    res_rand = rand_summarizer.summarize(sentences)
    score_rand = evaluator.evaluate(res_rand.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "random_baseline",
        "summary_text": res_rand.summary_text,
        "word_count": len(res_rand.summary_text.split()),
        "sentence_count": res_rand.num_sentences,
        "runtime_seconds": res_rand.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_rand.selected_indices, W),
        "query_relevance": 0.0,
        **score_rand,
    })

    return cluster_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generic multi-document summarization.")
    parser.add_argument("--config", type=str, default="configs/generic_duc03_duc04.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clusters")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config) if Path(args.config).exists() else ExperimentConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
    logger.info(f"Loaded config: alpha={config.objective.alpha}, lambda={config.objective.lambda_weight}, budget={config.optimizer.budget_limit} {config.optimizer.budget_type}")

    splitter = SentenceSplitter(
        min_chars=config.preprocessing.min_sentence_chars,
        min_words=config.preprocessing.min_sentence_words,
    )
    normalizer = TextNormalizer(
        lowercase=config.preprocessing.lowercase,
        remove_punctuation=config.preprocessing.remove_punctuation,
        remove_stopwords=config.preprocessing.remove_stopwords,
        use_stemming=config.preprocessing.use_stemming,
    )
    evaluator = RougeEvaluator(metrics=config.evaluation.metrics)

    loader = DUCDatasetLoader(
        base_dir=config.data_dir,
        segmenter_fn=splitter.split,
        normalizer_fn=normalizer.normalize,
    )

    clusters = loader.load_dataset()
    if args.limit:
        clusters = clusters[: args.limit]
    logger.info(f"Loaded {len(clusters)} clusters for evaluation.")

    all_records = []
    for idx, cluster in enumerate(clusters, 1):
        logger.info(f"[{idx}/{len(clusters)}] Processing cluster: {cluster.cluster_id} ({len(cluster.sentences)} sentences)")
        cluster_records = run_generic_cluster(
            cluster=cluster,
            config=config,
            splitter=splitter,
            normalizer=normalizer,
            evaluator=evaluator,
        )
        all_records.extend(cluster_records)

    df = pd.DataFrame(all_records)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics" / f"generic_{config.dataset_name}_cluster_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved cluster-level results to {csv_path}")

    # Summary table
    grouped = df.groupby("method")[["rouge1_recall", "rouge1_f1", "rouge2_recall", "rouge2_f1", "redundancy", "word_count", "runtime_seconds"]].mean()
    print("\n================== AGGREGATED GENERIC PERFORMANCE ==================")
    print(grouped.to_string())
    print("====================================================================\n")

    # Bootstrap significance vs Lead baseline
    prop_r1 = df[df["method"] == "proposed_lin_bilmes"]["rouge1_recall"].values
    lead_r1 = df[df["method"] == "lead_baseline"]["rouge1_recall"].values
    test_res = StatisticalTester.paired_bootstrap_test(prop_r1, lead_r1)
    logger.info(
        f"Paired Bootstrap Test (Proposed vs Lead on ROUGE-1 Recall): "
        f"Mean Diff = {test_res['mean_diff']:.4f}, p-value = {test_res['p_value']:.4f}, "
        f"Significant (p < 0.05) = {bool(test_res['significant_05'])}"
    )


if __name__ == "__main__":
    main()


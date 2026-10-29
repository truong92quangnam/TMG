"""Query-focused multi-document summarization experiment runner.

Implements query-focused summarization on DUC 2005-2007:
    F(S; Q) = L_1(S) + lambda * R_Q(S)
and multi-resolution extension:
    F(S; Q) = L_1(S) + sum_{ell=1}^L lambda_ell * R_{Q, ell}(S)
Evaluates against Lead, Random, Top-K Query Overlap, and MMR baselines.
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
from src.preprocessing.query_processor import QueryProcessor
from src.preprocessing.vectorizer import ClusterVectorizer
from src.features.similarity import SimilarityMatrixBuilder
from src.features.centrality import CentralityCalculator
from src.features.relevance import QueryRelevanceCalculator
from src.features.clustering import SentenceClusterer
from src.objectives.coverage import TruncatedCoverageObjective
from src.objectives.query_coverage import QueryAwareCoverageObjective
from src.objectives.query_diversity import QueryAwareClusterDiversityObjective
from src.objectives.composite import CompositeSubmodularObjective
from src.optimizers.greedy_knapsack import BudgetedGreedyOptimizer
from src.optimizers.baselines import (
    LeadBaselineSummarizer,
    RandomBaselineSummarizer,
    TopKQuerySummarizer,
    MMRSummarizer,
)
from src.evaluation.rouge_eval import RougeEvaluator
from src.evaluation.statistics import StatisticalTester
from src.evaluation.reports import EvaluationReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_query_focused")


def run_query_focused_cluster(
    cluster,
    config: ExperimentConfig,
    splitter: SentenceSplitter,
    normalizer: TextNormalizer,
    query_processor: QueryProcessor,
    evaluator: RougeEvaluator,
) -> list[dict]:
    """Execute query-focused methods and baselines on a single cluster."""
    sentences = cluster.sentences
    n = len(sentences)
    if n == 0:
        logger.warning(f"Cluster {cluster.cluster_id} has no sentences.")
        return []

    # 1. Process query
    q_data = query_processor.process_query(cluster.query or "")
    query_terms = q_data["all_terms"]

    # 2. Extract sentence terms for overlap
    sentence_terms_list = []
    for s in sentences:
        tokens = normalizer.tokenize(s.original_text)
        bigrams = normalizer.get_ngrams(tokens, n=2)
        sentence_terms_list.append(set(tokens) | set(bigrams))

    # 3. TF-IDF vectorization
    vectorizer = ClusterVectorizer(
        normalizer=normalizer,
        ngram_range=config.preprocessing.ngram_range,
        sublinear_tf=config.preprocessing.sublinear_tf,
    )
    raw_texts = [s.original_text for s in sentences]
    tfidf_matrix = vectorizer.fit_transform(raw_texts)

    # 4. Pairwise similarity matrix W (supports lexical TF-IDF, latent semantic LSA, or hybrid)
    W = SimilarityMatrixBuilder.compute(
        feature_matrix=tfidf_matrix,
        keep_diagonal=config.similarity.keep_diagonal,
        clip_negative=config.similarity.clip_negative,
        method=config.similarity.method,
        n_components=config.similarity.lsa_components,
        hybrid_weight=config.similarity.hybrid_weight,
        random_seed=config.seed,
    )

    # 5. Singleton centrality r_j
    r_centrality = CentralityCalculator.compute(W)

    # 6. Query relevance r_jQ (term overlap + semantic similarity)
    r_overlap = QueryRelevanceCalculator.compute_overlap_relevance(
        sentence_terms_list=sentence_terms_list,
        query_terms=query_terms,
    )

    # Semantic query similarity vector w_jQ
    q_vec = vectorizer.transform_query(cluster.query or "")
    if q_vec is not None:
        query_sem_sim = SimilarityMatrixBuilder.compute_query_similarity(
            feature_matrix=tfidf_matrix,
            query_vector=q_vec,
            method=config.similarity.method,
            n_components=config.similarity.lsa_components,
            hybrid_weight=config.similarity.hybrid_weight,
            random_seed=config.seed,
        )
        # Combined query relevance score
        r_relevance = 0.5 * r_overlap + 0.5 * query_sem_sim
        query_sim = query_sem_sim
    else:
        r_relevance = r_overlap
        query_sim = r_overlap.copy()

    # 7. Setup Submodular Objectives
    # (A) Classic generic coverage L_1(S) (Lin & Bilmes 2011)
    generic_cov_obj = TruncatedCoverageObjective(W=W, alpha=config.objective.alpha)

    # (B) User's Novel Extension: Query-Aware Coverage L_Q(S)
    query_aware_cov_obj = QueryAwareCoverageObjective(
        W=W,
        query_sim=query_sim,
        alpha=config.objective.alpha,
        alpha_q=config.objective.query_coverage_alpha_q,
        gamma=config.objective.query_coverage_gamma,
        eta=config.objective.query_coverage_eta,
    )

    # Single-Resolution Diversity R_Q(S)
    clusterer = SentenceClusterer(k_ratio=config.objective.k_ratio, random_seed=config.seed)
    labels, partitions = clusterer.cluster(tfidf_matrix)
    div_obj_single = QueryAwareClusterDiversityObjective(
        cluster_labels=labels,
        num_clusters=len(partitions),
        centrality_r=r_centrality,
        relevance_rQ=r_relevance,
        beta=config.objective.beta,
        concave_func=config.objective.concave_func,
        sentence_texts=raw_texts,
    )

    # Multi-Resolution Diversity: L levels
    multi_div_objs = []
    for k_ratio, lam_val in zip(
        config.objective.multi_resolution_ratios,
        config.objective.multi_resolution_lambdas,
    ):
        m_clusterer = SentenceClusterer(k_ratio=k_ratio, random_seed=config.seed)
        m_labels, m_partitions = m_clusterer.cluster(tfidf_matrix)
        m_div = QueryAwareClusterDiversityObjective(
            cluster_labels=m_labels,
            num_clusters=len(m_partitions),
            centrality_r=r_centrality,
            relevance_rQ=r_relevance,
            beta=config.objective.beta,
            concave_func=config.objective.concave_func,
        )
        multi_div_objs.append((lam_val, m_div))

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

    # Method 1: Original Lin & Bilmes 2011: Generic L_1(S) + lambda * R_Q(S)
    lb_obj = CompositeSubmodularObjective(
        coverage_obj=generic_cov_obj,
        diversity_objs=[(config.objective.lambda_weight, div_obj_single)],
    )
    res_lb = optimizer.optimize(lb_obj, sentences)
    score_lb = evaluator.evaluate(res_lb.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "lin_bilmes_2011_generic_cov",
        "summary_text": res_lb.summary_text,
        "word_count": len(res_lb.summary_text.split()),
        "sentence_count": res_lb.num_sentences,
        "runtime_seconds": res_lb.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_lb.selected_indices, W),
        "query_relevance": EvaluationReporter.compute_query_relevance(res_lb.selected_indices, r_relevance),
        **score_lb,
    })

    # Method 2: User's Proposed Approach: Query-Aware Coverage L_Q(S) + lambda * R_Q(S)
    prop_obj = CompositeSubmodularObjective(
        coverage_obj=query_aware_cov_obj,
        diversity_objs=[(config.objective.lambda_weight, div_obj_single)],
    )
    res_prop = optimizer.optimize(prop_obj, sentences)
    score_prop = evaluator.evaluate(res_prop.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "user_query_aware_coverage_single_res",
        "summary_text": res_prop.summary_text,
        "word_count": len(res_prop.summary_text.split()),
        "sentence_count": res_prop.num_sentences,
        "runtime_seconds": res_prop.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_prop.selected_indices, W),
        "query_relevance": EvaluationReporter.compute_query_relevance(res_prop.selected_indices, r_relevance),
        **score_prop,
    })

    # Method 3: User's Multi-Resolution Approach: L_Q(S) + sum lambda_ell * R_Q,ell(S)
    multires_obj = CompositeSubmodularObjective(
        coverage_obj=query_aware_cov_obj,
        diversity_objs=multi_div_objs,
    )
    res_multi = optimizer.optimize(multires_obj, sentences)
    score_multi = evaluator.evaluate(res_multi.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "user_query_aware_coverage_multi_res",
        "summary_text": res_multi.summary_text,
        "word_count": len(res_multi.summary_text.split()),
        "sentence_count": res_multi.num_sentences,
        "runtime_seconds": res_multi.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_multi.selected_indices, W),
        "query_relevance": EvaluationReporter.compute_query_relevance(res_multi.selected_indices, r_relevance),
        **score_multi,
    })

    # Method 3: MMR Baseline
    mmr = MMRSummarizer(
        budget_limit=config.optimizer.budget_limit,
        budget_type=config.optimizer.budget_type,
        lambda_mmr=config.optimizer.mmr_lambda,
    )
    res_mmr = mmr.summarize(sentences, W, query_sim)
    score_mmr = evaluator.evaluate(res_mmr.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "mmr_baseline",
        "summary_text": res_mmr.summary_text,
        "word_count": len(res_mmr.summary_text.split()),
        "sentence_count": res_mmr.num_sentences,
        "runtime_seconds": res_mmr.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_mmr.selected_indices, W),
        "query_relevance": EvaluationReporter.compute_query_relevance(res_mmr.selected_indices, r_relevance),
        **score_mmr,
    })

    # Method 4: Top-K Query Relevance Density
    topk = TopKQuerySummarizer(
        budget_limit=config.optimizer.budget_limit,
        budget_type=config.optimizer.budget_type,
    )
    res_topk = topk.summarize(sentences, r_relevance)
    score_topk = evaluator.evaluate(res_topk.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "top_k_query_baseline",
        "summary_text": res_topk.summary_text,
        "word_count": len(res_topk.summary_text.split()),
        "sentence_count": res_topk.num_sentences,
        "runtime_seconds": res_topk.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_topk.selected_indices, W),
        "query_relevance": EvaluationReporter.compute_query_relevance(res_topk.selected_indices, r_relevance),
        **score_topk,
    })

    # Method 5: Lead Baseline
    lead = LeadBaselineSummarizer(
        budget_limit=config.optimizer.budget_limit,
        budget_type=config.optimizer.budget_type,
    )
    res_lead = lead.summarize(sentences)
    score_lead = evaluator.evaluate(res_lead.summary_text, cluster.references)
    cluster_records.append({
        "cluster_id": cluster.cluster_id,
        "method": "lead_baseline",
        "summary_text": res_lead.summary_text,
        "word_count": len(res_lead.summary_text.split()),
        "sentence_count": res_lead.num_sentences,
        "runtime_seconds": res_lead.runtime_seconds,
        "redundancy": EvaluationReporter.compute_redundancy(res_lead.selected_indices, W),
        "query_relevance": EvaluationReporter.compute_query_relevance(res_lead.selected_indices, r_relevance),
        **score_lead,
    })

    return cluster_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query-focused summarization.")
    parser.add_argument("--config", type=str, default="configs/query_duc05_duc06.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config) if Path(args.config).exists() else ExperimentConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
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
    query_processor = QueryProcessor(
        normalizer=normalizer,
        instruction_words=config.preprocessing.instruction_words,
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
    logger.info(f"Loaded {len(clusters)} query-focused clusters.")

    all_records = []
    for idx, cluster in enumerate(clusters, 1):
        logger.info(f"[{idx}/{len(clusters)}] Processing query cluster {cluster.cluster_id}")
        records = run_query_focused_cluster(
            cluster=cluster,
            config=config,
            splitter=splitter,
            normalizer=normalizer,
            query_processor=query_processor,
            evaluator=evaluator,
        )
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    out_dir = Path(config.output_dir)
    csv_path = out_dir / "metrics" / f"query_{config.dataset_name}_cluster_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved cluster results to {csv_path}")

    grouped = df.groupby("method")[["rouge2_recall", "rouge2_f1", "rouge1_recall", "rouge1_f1", "redundancy", "query_relevance", "word_count", "runtime_seconds"]].mean()
    print("\n================== AGGREGATED QUERY-FOCUSED PERFORMANCE ==================")
    print(grouped.to_string())
    print("==========================================================================\n")


if __name__ == "__main__":
    main()


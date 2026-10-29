"""Ablation study and visualization module.

Evaluates the contribution of each component:
A. Raw similarity objective: sum_{i in V} sum_{j in S} w_ij
B. Coverage only: L_1(S)
C. Diversity only: R_1(S)
D. Proposed generic: L_1(S) + lambda * R_1(S)
E. Query-aware: L_1(S) + lambda * R_Q(S)
F. Multi-resolution: L_1(S) + sum lambda_ell * R_{Q, ell}(S)

Generates analysis plots:
1. ROUGE vs alpha saturation
2. ROUGE vs lambda diversity tradeoff
3. ROUGE vs K_ratio
4. Redundancy vs ROUGE
5. Runtime vs sentence count N
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from src.objectives.diversity import ClusterDiversityObjective
from src.objectives.query_diversity import QueryAwareClusterDiversityObjective
from src.objectives.composite import CompositeSubmodularObjective
from src.optimizers.greedy_knapsack import BudgetedGreedyOptimizer
from src.evaluation.rouge_eval import RougeEvaluator
from src.evaluation.reports import EvaluationReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ablation")


def run_ablation_on_clusters(
    clusters,
    config: ExperimentConfig,
    splitter: SentenceSplitter,
    normalizer: TextNormalizer,
    query_processor: QueryProcessor,
    evaluator: RougeEvaluator,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run ablation variants A through F across clusters."""
    optimizer = BudgetedGreedyOptimizer(
        budget_limit=config.optimizer.budget_limit,
        budget_type=config.optimizer.budget_type,
        cost_power=config.optimizer.cost_power,
        use_lazy_greedy=config.optimizer.use_lazy_greedy,
        fill_budget_fractional=config.optimizer.fill_budget_fractional,
    )

    all_records = []
    telemetry_data = []

    for idx, cluster in enumerate(clusters, 1):
        sentences = cluster.sentences
        n = len(sentences)
        if n == 0:
            continue

        raw_texts = [s.original_text for s in sentences]
        vectorizer = ClusterVectorizer(
            normalizer=normalizer,
            ngram_range=config.preprocessing.ngram_range,
            sublinear_tf=config.preprocessing.sublinear_tf,
        )
        tfidf = vectorizer.fit_transform(raw_texts)
        W = SimilarityMatrixBuilder.compute(tfidf)
        r = CentralityCalculator.compute(W)

        # Query features
        q_data = query_processor.process_query(cluster.query or cluster.documents[0].text[:120])
        sentence_terms_list = [
            set(normalizer.tokenize(s.original_text)) | set(normalizer.get_ngrams(normalizer.tokenize(s.original_text), 2))
            for s in sentences
        ]
        r_relevance = QueryRelevanceCalculator.compute_overlap_relevance(
            sentence_terms_list, q_data["all_terms"]
        )

        clusterer = SentenceClusterer(k_ratio=config.objective.k_ratio, random_seed=config.seed)
        labels, partitions = clusterer.cluster(tfidf)

        # Component objectives
        raw_cov_obj = TruncatedCoverageObjective(W=W, alpha=1.0)  # alpha=1.0: no saturation clipping
        cov_obj = TruncatedCoverageObjective(W=W, alpha=config.objective.alpha)
        zero_cov_obj = TruncatedCoverageObjective(W=W, alpha=0.0)

        div_obj = ClusterDiversityObjective(
            cluster_labels=labels,
            num_clusters=len(partitions),
            rewards=r,
            concave_func=config.objective.concave_func,
        )
        query_div_obj = QueryAwareClusterDiversityObjective(
            cluster_labels=labels,
            num_clusters=len(partitions),
            centrality_r=r,
            relevance_rQ=r_relevance,
            beta=config.objective.beta,
            concave_func=config.objective.concave_func,
        )

        # Multi-resolution
        multi_div_objs = []
        for k_ratio, lam_val in zip(
            config.objective.multi_resolution_ratios,
            config.objective.multi_resolution_lambdas,
        ):
            m_clusterer = SentenceClusterer(k_ratio=k_ratio, random_seed=config.seed)
            m_labels, m_parts = m_clusterer.cluster(tfidf)
            m_div = QueryAwareClusterDiversityObjective(
                cluster_labels=m_labels,
                num_clusters=len(m_parts),
                centrality_r=r,
                relevance_rQ=r_relevance,
                beta=config.objective.beta,
                concave_func=config.objective.concave_func,
            )
            multi_div_objs.append((lam_val, m_div))

        # A. Raw Similarity Objective (no saturation, no diversity)
        obj_A = CompositeSubmodularObjective(raw_cov_obj, [(0.0, div_obj)])
        res_A = optimizer.optimize(obj_A, sentences)
        sc_A = evaluator.evaluate(res_A.summary_text, cluster.references)
        all_records.append({
            "cluster_id": cluster.cluster_id,
            "ablation": "A_Raw_Similarity",
            "redundancy": EvaluationReporter.compute_redundancy(res_A.selected_indices, W),
            "runtime_seconds": res_A.runtime_seconds,
            "sentence_count_N": n,
            **sc_A,
        })

        # B. Coverage Only (with saturation alpha)
        obj_B = CompositeSubmodularObjective(cov_obj, [(0.0, div_obj)])
        res_B = optimizer.optimize(obj_B, sentences)
        sc_B = evaluator.evaluate(res_B.summary_text, cluster.references)
        all_records.append({
            "cluster_id": cluster.cluster_id,
            "ablation": "B_Coverage_Only",
            "redundancy": EvaluationReporter.compute_redundancy(res_B.selected_indices, W),
            "runtime_seconds": res_B.runtime_seconds,
            "sentence_count_N": n,
            **sc_B,
        })

        # C. Diversity Only
        obj_C = CompositeSubmodularObjective(zero_cov_obj, [(1.0, div_obj)])
        res_C = optimizer.optimize(obj_C, sentences)
        sc_C = evaluator.evaluate(res_C.summary_text, cluster.references)
        all_records.append({
            "cluster_id": cluster.cluster_id,
            "ablation": "C_Diversity_Only",
            "redundancy": EvaluationReporter.compute_redundancy(res_C.selected_indices, W),
            "runtime_seconds": res_C.runtime_seconds,
            "sentence_count_N": n,
            **sc_C,
        })

        # D. Coverage + Diversity (Proposed Generic)
        obj_D = CompositeSubmodularObjective(cov_obj, [(config.objective.lambda_weight, div_obj)])
        res_D = optimizer.optimize(obj_D, sentences)
        sc_D = evaluator.evaluate(res_D.summary_text, cluster.references)
        all_records.append({
            "cluster_id": cluster.cluster_id,
            "ablation": "D_Coverage_Plus_Diversity",
            "redundancy": EvaluationReporter.compute_redundancy(res_D.selected_indices, W),
            "runtime_seconds": res_D.runtime_seconds,
            "sentence_count_N": n,
            **sc_D,
        })

        # E. Query-Aware Diversity
        obj_E = CompositeSubmodularObjective(cov_obj, [(config.objective.lambda_weight, query_div_obj)])
        res_E = optimizer.optimize(obj_E, sentences)
        sc_E = evaluator.evaluate(res_E.summary_text, cluster.references)
        all_records.append({
            "cluster_id": cluster.cluster_id,
            "ablation": "E_Query_Aware",
            "redundancy": EvaluationReporter.compute_redundancy(res_E.selected_indices, W),
            "runtime_seconds": res_E.runtime_seconds,
            "sentence_count_N": n,
            **sc_E,
        })

        # F. Multi-Resolution
        obj_F = CompositeSubmodularObjective(cov_obj, multi_div_objs)
        res_F = optimizer.optimize(obj_F, sentences)
        sc_F = evaluator.evaluate(res_F.summary_text, cluster.references)
        all_records.append({
            "cluster_id": cluster.cluster_id,
            "ablation": "F_Multi_Resolution",
            "redundancy": EvaluationReporter.compute_redundancy(res_F.selected_indices, W),
            "runtime_seconds": res_F.runtime_seconds,
            "sentence_count_N": n,
            **sc_F,
        })

        # G. User Novel Extension: Query-Aware Coverage L_Q(S) + lambda * R_Q(S)
        q_vec = vectorizer.transform_query(cluster.query or cluster.documents[0].text[:120])
        query_sem_sim = SimilarityMatrixBuilder.compute_query_similarity(
            feature_matrix=tfidf,
            query_vector=q_vec,
            method="lsa",
            n_components=config.similarity.lsa_components,
        )
        qa_cov_obj = QueryAwareCoverageObjective(
            W=W,
            query_sim=query_sem_sim,
            alpha=config.objective.alpha,
            alpha_q=config.objective.query_coverage_alpha_q,
            gamma=config.objective.query_coverage_gamma,
            eta=config.objective.query_coverage_eta,
        )
        obj_G = CompositeSubmodularObjective(qa_cov_obj, [(config.objective.lambda_weight, query_div_obj)])
        res_G = optimizer.optimize(obj_G, sentences)
        sc_G = evaluator.evaluate(res_G.summary_text, cluster.references)
        all_records.append({
            "cluster_id": cluster.cluster_id,
            "ablation": "G_User_Query_Aware_Coverage",
            "redundancy": EvaluationReporter.compute_redundancy(res_G.selected_indices, W),
            "runtime_seconds": res_G.runtime_seconds,
            "sentence_count_N": n,
            **sc_G,
        })

        # H. Semantic Similarity (LSA Latent Semantic Space)
        W_lsa = SimilarityMatrixBuilder.compute(
            feature_matrix=tfidf,
            method="lsa",
            n_components=config.similarity.lsa_components,
            keep_diagonal=config.similarity.keep_diagonal,
            clip_negative=config.similarity.clip_negative,
            random_seed=config.seed,
        )
        r_lsa = CentralityCalculator.compute(W_lsa)
        cov_lsa = TruncatedCoverageObjective(W=W_lsa, alpha=config.objective.alpha)
        div_lsa = ClusterDiversityObjective(
            cluster_labels=labels,
            num_clusters=len(partitions),
            rewards=r_lsa,
            concave_func=config.objective.concave_func,
        )
        obj_H = CompositeSubmodularObjective(cov_lsa, [(config.objective.lambda_weight, div_lsa)])
        res_H = optimizer.optimize(obj_H, sentences)
        sc_H = evaluator.evaluate(res_H.summary_text, cluster.references)
        all_records.append({
            "cluster_id": cluster.cluster_id,
            "ablation": "H_Semantic_LSA_Similarity",
            "redundancy": EvaluationReporter.compute_redundancy(res_H.selected_indices, W_lsa),
            "runtime_seconds": res_H.runtime_seconds,
            "sentence_count_N": n,
            **sc_H,
        })

        telemetry_data.append({
            "sentence_count_N": n,
            "runtime_seconds": res_D.runtime_seconds,
            "redundancy": EvaluationReporter.compute_redundancy(res_D.selected_indices, W),
            "rouge1_recall": sc_D["rouge1_recall"],
            "rouge2_recall": sc_D["rouge2_recall"],
        })

    df_ablation = pd.DataFrame(all_records)
    return df_ablation, telemetry_data


def generate_plots(
    df_ablation: pd.DataFrame,
    telemetry_data: list[dict],
    clusters,
    config: ExperimentConfig,
    normalizer: TextNormalizer,
    evaluator: RougeEvaluator,
    output_dir: Path,
) -> None:
    """Generate and save the 5 required analytical plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: ROUGE vs Alpha
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    alpha_r1 = []
    optimizer = BudgetedGreedyOptimizer(budget_limit=config.optimizer.budget_limit)
    sample_clusters = clusters[: min(10, len(clusters))]

    for alpha_val in alphas:
        scores = []
        for c in sample_clusters:
            raw_texts = [s.original_text for s in c.sentences]
            if not raw_texts:
                continue
            vec = ClusterVectorizer(normalizer=normalizer)
            tfidf = vec.fit_transform(raw_texts)
            W = SimilarityMatrixBuilder.compute(tfidf)
            r = CentralityCalculator.compute(W)
            clusterer = SentenceClusterer(k_ratio=config.objective.k_ratio)
            labels, parts = clusterer.cluster(tfidf)
            cov = TruncatedCoverageObjective(W, alpha=alpha_val)
            div = ClusterDiversityObjective(labels, len(parts), r)
            comp = CompositeSubmodularObjective(cov, [(config.objective.lambda_weight, div)])
            res = optimizer.optimize(comp, c.sentences)
            sc = evaluator.evaluate(res.summary_text, c.references)
            scores.append(sc["rouge1_recall"])
        alpha_r1.append(float(np.mean(scores)) if scores else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, alpha_r1, marker="o", color="#1f77b4", linewidth=2)
    ax.set_title("ROUGE-1 Recall vs. Saturation Threshold (alpha)", fontsize=13)
    ax.set_xlabel("Saturation Parameter alpha", fontsize=11)
    ax.set_ylabel("ROUGE-1 Recall", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "rouge_vs_alpha.png", dpi=200)
    plt.close(fig)

    # Plot 2: ROUGE vs Lambda
    lambdas = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
    lam_r1 = []
    for lam_val in lambdas:
        scores = []
        for c in sample_clusters:
            raw_texts = [s.original_text for s in c.sentences]
            if not raw_texts:
                continue
            vec = ClusterVectorizer(normalizer=normalizer)
            tfidf = vec.fit_transform(raw_texts)
            W = SimilarityMatrixBuilder.compute(tfidf)
            r = CentralityCalculator.compute(W)
            clusterer = SentenceClusterer(k_ratio=config.objective.k_ratio)
            labels, parts = clusterer.cluster(tfidf)
            cov = TruncatedCoverageObjective(W, alpha=config.objective.alpha)
            div = ClusterDiversityObjective(labels, len(parts), r)
            comp = CompositeSubmodularObjective(cov, [(lam_val, div)])
            res = optimizer.optimize(comp, c.sentences)
            sc = evaluator.evaluate(res.summary_text, c.references)
            scores.append(sc["rouge1_recall"])
        lam_r1.append(float(np.mean(scores)) if scores else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lambdas, lam_r1, marker="s", color="#2ca02c", linewidth=2)
    ax.set_title("ROUGE-1 Recall vs. Diversity Weight (lambda)", fontsize=13)
    ax.set_xlabel("Diversity Weight lambda", fontsize=11)
    ax.set_ylabel("ROUGE-1 Recall", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "rouge_vs_lambda.png", dpi=200)
    plt.close(fig)

    # Plot 3: ROUGE vs K_ratio
    k_ratios = [0.05, 0.10, 0.15, 0.20, 0.25]
    k_r1 = []
    for kr in k_ratios:
        scores = []
        for c in sample_clusters:
            raw_texts = [s.original_text for s in c.sentences]
            if not raw_texts:
                continue
            vec = ClusterVectorizer(normalizer=normalizer)
            tfidf = vec.fit_transform(raw_texts)
            W = SimilarityMatrixBuilder.compute(tfidf)
            r = CentralityCalculator.compute(W)
            clusterer = SentenceClusterer(k_ratio=kr)
            labels, parts = clusterer.cluster(tfidf)
            cov = TruncatedCoverageObjective(W, alpha=config.objective.alpha)
            div = ClusterDiversityObjective(labels, len(parts), r)
            comp = CompositeSubmodularObjective(cov, [(config.objective.lambda_weight, div)])
            res = optimizer.optimize(comp, c.sentences)
            sc = evaluator.evaluate(res.summary_text, c.references)
            scores.append(sc["rouge1_recall"])
        k_r1.append(float(np.mean(scores)) if scores else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_ratios, k_r1, marker="^", color="#d62728", linewidth=2)
    ax.set_title("ROUGE-1 Recall vs. Cluster Ratio (K_ratio)", fontsize=13)
    ax.set_xlabel("Cluster Ratio K_ratio (K = max(2, round(K_ratio * N)))", fontsize=11)
    ax.set_ylabel("ROUGE-1 Recall", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "rouge_vs_k_ratio.png", dpi=200)
    plt.close(fig)

    # Plot 4: Redundancy vs ROUGE
    df_tel = pd.DataFrame(telemetry_data)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not df_tel.empty:
        ax.scatter(df_tel["redundancy"], df_tel["rouge1_recall"], color="#9467bd", alpha=0.75, edgecolors="none", s=50)
        # Add trend line
        z = np.polyfit(df_tel["redundancy"], df_tel["rouge1_recall"], 1)
        p = np.poly1d(z)
        x_vals = np.linspace(df_tel["redundancy"].min(), df_tel["redundancy"].max(), 50)
        ax.plot(x_vals, p(x_vals), "r--", linewidth=1.5, label="Trendline")
        ax.legend()
    ax.set_title("Summary Redundancy vs. ROUGE-1 Recall", fontsize=13)
    ax.set_xlabel("Average Pairwise Similarity (Redundancy)", fontsize=11)
    ax.set_ylabel("ROUGE-1 Recall", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "redundancy_vs_rouge.png", dpi=200)
    plt.close(fig)

    # Plot 5: Runtime vs Sentence Count N
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not df_tel.empty:
        sorted_tel = df_tel.sort_values(by="sentence_count_N")
        ax.scatter(sorted_tel["sentence_count_N"], sorted_tel["runtime_seconds"], color="#ff7f0e", alpha=0.75, s=50)
        z = np.polyfit(sorted_tel["sentence_count_N"], sorted_tel["runtime_seconds"], 2)
        p = np.poly1d(z)
        x_vals = np.linspace(sorted_tel["sentence_count_N"].min(), sorted_tel["sentence_count_N"].max(), 50)
        ax.plot(x_vals, p(x_vals), "b--", linewidth=1.5, label="O(N^2) fit")
        ax.legend()
    ax.set_title("Runtime Scaling vs. Number of Sentences N", fontsize=13)
    ax.set_xlabel("Number of Sentences in Cluster (N)", fontsize=11)
    ax.set_ylabel("Execution Time (seconds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "runtime_vs_sentence_count.png", dpi=200)
    plt.close(fig)

    logger.info(f"Generated and saved all 5 ablation plots to {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study across components.")
    parser.add_argument("--config", type=str, default="configs/generic_duc03_duc04.yaml")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config) if Path(args.config).exists() else ExperimentConfig()
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
    query_processor = QueryProcessor(normalizer=normalizer)
    evaluator = RougeEvaluator(metrics=["rouge1", "rouge2", "rougeL"])

    loader = DUCDatasetLoader(
        base_dir=config.data_dir,
        segmenter_fn=splitter.split,
        normalizer_fn=normalizer.normalize,
    )
    clusters = loader.load_dataset()
    if args.limit:
        clusters = clusters[: args.limit]

    logger.info(f"Running ablation study on {len(clusters)} clusters...")
    df_ablation, telemetry_data = run_ablation_on_clusters(
        clusters=clusters,
        config=config,
        splitter=splitter,
        normalizer=normalizer,
        query_processor=query_processor,
        evaluator=evaluator,
    )

    out_dir = Path(config.output_dir)
    csv_path = out_dir / "metrics" / "ablation_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_ablation.to_csv(csv_path, index=False)
    logger.info(f"Saved ablation table to {csv_path}")

    # Generate plots
    generate_plots(
        df_ablation=df_ablation,
        telemetry_data=telemetry_data,
        clusters=clusters,
        config=config,
        normalizer=normalizer,
        evaluator=evaluator,
        output_dir=out_dir,
    )

    grouped = df_ablation.groupby("ablation")[
        ["rouge1_recall", "rouge1_f1", "rouge2_recall", "rouge2_f1", "rougeL_f1", "redundancy", "runtime_seconds"]
    ].mean()
    print("\n======================= ABLATION STUDY RESULTS =======================")
    print(grouped.to_string())
    print("======================================================================\n")


if __name__ == "__main__":
    main()


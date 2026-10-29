"""Hyperparameter tuning and grid search module.

Performs parameter exploration over alpha, lambda, beta, and K_ratio
following the train/test split protocol specified in the experimental design.
"""

from __future__ import annotations

import argparse
import itertools
import json
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
from src.objectives.diversity import ClusterDiversityObjective
from src.objectives.query_diversity import QueryAwareClusterDiversityObjective
from src.objectives.composite import CompositeSubmodularObjective
from src.optimizers.greedy_knapsack import BudgetedGreedyOptimizer
from src.evaluation.rouge_eval import RougeEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tune")


def tune_generic_grid(
    clusters,
    config: ExperimentConfig,
    splitter: SentenceSplitter,
    normalizer: TextNormalizer,
    evaluator: RougeEvaluator,
    alpha_grid: list[float] | None = None,
    lambda_grid: list[float] | None = None,
    k_ratio_grid: list[float] | None = None,
) -> pd.DataFrame:
    """Run grid search for generic summarization over train/dev clusters."""
    alpha_list = alpha_grid or [0.10, 0.20, 0.30]
    lambda_list = lambda_grid or [0.5, 1.0, 2.0, 4.0]
    k_ratio_list = k_ratio_grid or [0.05, 0.10, 0.20]

    combinations = list(itertools.product(alpha_list, lambda_list, k_ratio_list))
    logger.info(f"Running grid search over {len(combinations)} parameter configurations on {len(clusters)} clusters...")

    # Precompute TF-IDF and similarity matrices for efficiency
    cluster_cache = []
    for c in clusters:
        raw_texts = [s.original_text for s in c.sentences]
        if not raw_texts:
            continue
        vec = ClusterVectorizer(
            normalizer=normalizer,
            ngram_range=config.preprocessing.ngram_range,
            sublinear_tf=config.preprocessing.sublinear_tf,
        )
        tfidf = vec.fit_transform(raw_texts)
        W = SimilarityMatrixBuilder.compute(
            tfidf_matrix=tfidf,
            keep_diagonal=config.similarity.keep_diagonal,
            clip_negative=config.similarity.clip_negative,
        )
        r = CentralityCalculator.compute(W)
        cluster_cache.append({
            "cluster": c,
            "tfidf": tfidf,
            "W": W,
            "r": r,
        })

    results = []
    for alpha, lam, k_ratio in combinations:
        cluster_scores = []
        for item in cluster_cache:
            c = item["cluster"]
            W = item["W"]
            r = item["r"]
            tfidf = item["tfidf"]

            clusterer = SentenceClusterer(k_ratio=k_ratio, random_seed=config.seed)
            labels, partitions = clusterer.cluster(tfidf)

            cov_obj = TruncatedCoverageObjective(W=W, alpha=alpha)
            div_obj = ClusterDiversityObjective(
                cluster_labels=labels,
                num_clusters=len(partitions),
                rewards=r,
                concave_func=config.objective.concave_func,
            )
            comp_obj = CompositeSubmodularObjective(
                coverage_obj=cov_obj,
                diversity_objs=[(lam, div_obj)],
            )

            optimizer = BudgetedGreedyOptimizer(
                budget_limit=config.optimizer.budget_limit,
                budget_type=config.optimizer.budget_type,
            )
            res = optimizer.optimize(comp_obj, c.sentences)
            score = evaluator.evaluate(res.summary_text, c.references)
            cluster_scores.append(score["rouge1_recall"])

        mean_rouge1_r = float(np.mean(cluster_scores)) if cluster_scores else 0.0
        results.append({
            "alpha": alpha,
            "lambda": lam,
            "k_ratio": k_ratio,
            "mean_rouge1_recall": mean_rouge1_r,
        })
        logger.info(f"alpha={alpha:.2f}, lambda={lam:.2f}, k_ratio={k_ratio:.2f} -> ROUGE-1 Recall: {mean_rouge1_r:.4f}")

    df = pd.DataFrame(results).sort_values(by="mean_rouge1_recall", ascending=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune submodular summarization hyperparameters.")
    parser.add_argument("--config", type=str, default="configs/generic_duc03_duc04.yaml")
    parser.add_argument("--num_train_clusters", type=int, default=10)
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
    evaluator = RougeEvaluator(metrics=["rouge1", "rouge2"])

    loader = DUCDatasetLoader(
        base_dir=config.data_dir,
        segmenter_fn=splitter.split,
        normalizer_fn=normalizer.normalize,
    )
    all_clusters = loader.load_dataset()
    train_clusters = all_clusters[: args.num_train_clusters]

    df_grid = tune_generic_grid(
        clusters=train_clusters,
        config=config,
        splitter=splitter,
        normalizer=normalizer,
        evaluator=evaluator,
    )

    out_path = Path(config.output_dir) / "metrics" / "tuning_grid_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(out_path, index=False)
    logger.info(f"Saved tuning results to {out_path}")
    print("\nTop 5 Parameter Configurations:")
    print(df_grid.head(5).to_string(index=False))


if __name__ == "__main__":
    main()


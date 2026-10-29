"""Configuration module for submodular document summarization.

Defines typed configuration classes for preprocessing, feature extraction,
submodular objectives, greedy optimization, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import yaml


@dataclass
class PreprocessingConfig:
    """Configuration for sentence splitting, tokenization, and vectorization."""
    min_sentence_chars: int = 15
    min_sentence_words: int = 3
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    use_stemming: bool = True
    ngram_range: tuple[int, int] = (1, 2)
    sublinear_tf: bool = True
    max_vocab_size: int | None = None
    instruction_words: list[str] = field(
        default_factory=lambda: [
            "describe", "explain", "identify", "discuss", "provide",
            "background", "information", "detail", "details", "give",
            "overview", "summarize", "summary"
        ]
    )


@dataclass
class SimilarityConfig:
    """Configuration for pairwise sentence similarity matrix W and semantic similarity."""
    metric: Literal["cosine"] = "cosine"
    method: Literal["tfidf", "lsa", "hybrid"] = "lsa"  # Semantic similarity via LSA or lexical TF-IDF
    lsa_components: int = 50  # Number of latent semantic dimensions
    hybrid_weight: float = 0.5  # Weight of LSA in hybrid mode
    keep_diagonal: bool = False
    clip_negative: bool = True
    use_sparse_threshold: int = 2000  # use sparse matrix if N > threshold


@dataclass
class ObjectiveConfig:
    """Hyperparameters for Lin & Bilmes (2011) and Query-Aware submodular objectives."""
    alpha: float = 0.20  # Coverage saturation threshold in [0, 1]
    lambda_weight: float = 4.0  # Diversity tradeoff parameter lambda >= 0
    beta: float = 0.50  # Query-focus balance: beta * centrality + (1 - beta) * relevance
    k_ratio: float = 0.10  # K-means cluster ratio: K = max(2, round(k_ratio * N))
    concave_func: Literal["sqrt", "log"] = "sqrt"  # Diminishing return function
    
    # Query-Aware Coverage settings:
    use_query_aware_coverage: bool = True  # Enable user extension: Query-Aware Coverage L_Q(S)
    query_coverage_gamma: float = 0.30  # Weight for direct query coverage vs query-weighted doc coverage
    query_coverage_eta: float = 0.20  # Background context smoothing factor
    query_coverage_alpha_q: float = 0.50  # Saturation threshold for direct query coverage

    multi_resolution_ratios: list[float] = field(
        default_factory=lambda: [0.30, 0.15, 0.05]
    )
    multi_resolution_lambdas: list[float] = field(
        default_factory=lambda: [2.0, 1.5, 1.0]
    )


@dataclass
class OptimizerConfig:
    """Budget and search strategy configurations."""
    budget_type: Literal["words", "bytes"] = "words"
    budget_limit: int = 100  # DUC 2004 generic is ~100 words / 665 bytes; DUC 2005-2007 is 250 words
    enforce_strict_budget: bool = True
    fill_budget_fractional: bool = False  # Truncate last sentence to strictly fill budget limit (DUC benchmark style)
    cost_power: float = 1.0  # Cost scaling exponent r in [0, 1] from Lin & Bilmes (2011) Section 4.2
    use_lazy_greedy: bool = True  # Minoux (1978) accelerated greedy with priority queue
    modified_greedy_guarantee: bool = False  # consider max(S_greedy, argmax_{v} F({v}))
    sort_order: Literal["document_order", "appearance", "selection_order"] = "document_order"
    random_seed: int = 42
    mmr_lambda: float = 0.70  # For MMR baseline


@dataclass
class EvaluationConfig:
    """Metrics and statistical test settings."""
    metrics: list[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Root configuration for an end-to-end experiment."""
    task_type: Literal["generic", "query_focused", "multi_resolution"] = "generic"
    dataset_name: str = "duc2004"
    data_dir: str = "data/duc2004"
    output_dir: str = "outputs"
    seed: int = 42
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Create ExperimentConfig from a dictionary with nested dataclasses."""
        prep_data = data.get("preprocessing", {})
        if "ngram_range" in prep_data and isinstance(prep_data["ngram_range"], list):
            prep_data["ngram_range"] = tuple(prep_data["ngram_range"])

        return cls(
            task_type=data.get("task_type", "generic"),
            dataset_name=data.get("dataset_name", "duc2004"),
            data_dir=data.get("data_dir", "data/duc2004"),
            output_dir=data.get("output_dir", "outputs"),
            seed=data.get("seed", 42),
            preprocessing=PreprocessingConfig(**prep_data),
            similarity=SimilarityConfig(**data.get("similarity", {})),
            objective=ObjectiveConfig(**data.get("objective", {})),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load experiment configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save current configuration to YAML file."""
        import dataclasses
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)

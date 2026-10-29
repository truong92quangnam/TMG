"""TF-IDF vectorizer module.

Extracts unigram and bigram TF-IDF sparse matrices across all sentences within a cluster,
with support for sublinear TF scaling and shared query representation mapping.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.normalizer import TextNormalizer


class ClusterVectorizer:
    """Computes TF-IDF representations for sentences and queries within a cluster."""

    def __init__(
        self,
        normalizer: TextNormalizer | None = None,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        max_vocab_size: int | None = None,
    ) -> None:
        self.normalizer = normalizer or TextNormalizer()
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.max_vocab_size = max_vocab_size

        self.vectorizer = TfidfVectorizer(
            tokenizer=self.normalizer.tokenize,
            token_pattern=None,  # Use custom tokenizer
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            norm="l2",
            max_features=self.max_vocab_size,
        )
        self.is_fitted = False

    def fit_transform(
        self, sentence_texts: list[str]
    ) -> csr_matrix:
        """Fit vectorizer on cluster sentences and return sparse TF-IDF matrix [N, V]."""
        if not sentence_texts:
            raise ValueError("Cannot fit TF-IDF vectorizer on empty sentence list.")

        matrix = self.vectorizer.fit_transform(sentence_texts)
        self.is_fitted = True
        return matrix

    def transform_query(self, query_text: str | None) -> csr_matrix | None:
        """Project query into the sentence TF-IDF vocabulary space."""
        if not self.is_fitted or not query_text or not query_text.strip():
            return None
        return self.vectorizer.transform([query_text])

    @property
    def vocabulary(self) -> dict[str, int]:
        """Return vocabulary mapping."""
        return self.vectorizer.vocabulary_


"""Query processor for DUC 2005-2007 query-focused summarization tasks.

Filters generic task instructions (e.g. 'describe', 'identify', 'discuss')
and normalizes queries using the same tokenization/stemming pipeline as candidate sentences.
"""

from __future__ import annotations

import re
from src.preprocessing.normalizer import TextNormalizer


class QueryProcessor:
    """Processes narrative queries into task-relevant semantic tokens."""

    def __init__(
        self,
        normalizer: TextNormalizer | None = None,
        instruction_words: list[str] | None = None,
    ) -> None:
        self.normalizer = normalizer or TextNormalizer()
        self.instruction_words = set(
            instruction_words
            or [
                "describe", "explain", "identify", "discuss", "provide",
                "background", "information", "detail", "details", "give",
                "overview", "summarize", "summary", "report", "account"
            ]
        )
        if self.normalizer.use_stemming and self.normalizer.stemmer:
            self.stemmed_instructions = {
                self.normalizer.stemmer.stem(w.lower()) for w in self.instruction_words
            }
        else:
            self.stemmed_instructions = {w.lower() for w in self.instruction_words}

    def process_query(self, raw_query: str) -> dict[str, Any]:
        """Normalize raw query text and extract content tokens and n-grams.

        Returns:
            dict containing:
                - normalized_text: stemmed space-separated words
                - tokens: list of unigrams without instruction words
                - bigrams: list of bigrams
                - all_terms: union of unigrams and bigrams
        """
        if not raw_query:
            return {
                "normalized_text": "",
                "tokens": [],
                "bigrams": [],
                "all_terms": set(),
            }

        tokens = self.normalizer.tokenize(raw_query)
        # Filter instruction words
        filtered_tokens = [t for t in tokens if t not in self.stemmed_instructions]
        bigrams = self.normalizer.get_ngrams(filtered_tokens, n=2)

        return {
            "normalized_text": " ".join(filtered_tokens),
            "tokens": filtered_tokens,
            "bigrams": bigrams,
            "all_terms": set(filtered_tokens) | set(bigrams),
        }


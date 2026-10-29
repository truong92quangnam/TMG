"""Text normalization and tokenization module.

Performs lowercase transformation, punctuation stripping, English stopword removal,
and Porter stemming to reproduce the classic IR pipeline described in Lin & Bilmes (2011).
"""

from __future__ import annotations

import re
import string
import nltk
from nltk.stem.porter import PorterStemmer


class TextNormalizer:
    """Normalizes raw text into stemmed tokens and cleaned representations."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        use_stemming: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming

        self.stemmer = PorterStemmer() if use_stemming else None

        try:
            self.stopwords = set(nltk.corpus.stopwords.words("english"))
        except Exception:
            # Fallback stopword set if nltk data not found
            self.stopwords = {
                "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
                "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
                "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
                "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for",
                "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
                "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
                "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
                "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "ought",
                "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
                "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
                "themselves", "then", "there", "these", "they", "this", "those", "through",
                "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
                "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you"
            }

        self.token_pattern = re.compile(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b")

    def tokenize(self, text: str) -> list[str]:
        """Extract individual word tokens, applying lowercase and filtering."""
        if not text:
            return []

        processed = text.lower() if self.lowercase else text
        tokens = self.token_pattern.findall(processed)

        result: list[str] = []
        for t in tokens:
            if self.remove_stopwords and t in self.stopwords:
                continue
            if len(t) <= 1 and not t.isdigit():
                continue
            if self.use_stemming and self.stemmer:
                t = self.stemmer.stem(t)
            result.append(t)

        return result

    def get_ngrams(self, tokens: list[str], n: int = 2) -> list[str]:
        """Extract character or word n-grams from a list of tokens."""
        if len(tokens) < n:
            return []
        return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def normalize(self, text: str) -> str:
        """Return space-delimited normalized and stemmed tokens."""
        tokens = self.tokenize(text)
        return " ".join(tokens)


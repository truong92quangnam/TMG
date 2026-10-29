"""Sentence segmentation module.

Splits document text into clean, well-formed sentences using NLTK Punkt
with robust regex fallback, filtering out noisy, too short, or punctuation-only fragments.
"""

from __future__ import annotations

import re
import nltk


class SentenceSplitter:
    """Segments multi-document text into candidate sentences."""

    def __init__(self, min_chars: int = 15, min_words: int = 3) -> None:
        self.min_chars = min_chars
        self.min_words = min_words
        self._punct_regex = re.compile(r"^[\s\W_]+$")

        # Try to ensure nltk punkt is available
        try:
            self.tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        except Exception:
            try:
                self.tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
            except Exception:
                self.tokenizer = None

    def is_valid_sentence(self, text: str) -> bool:
        """Check if a candidate sentence satisfies length and quality criteria."""
        s = text.strip()
        if len(s) < self.min_chars:
            return False
        words = s.split()
        if len(words) < self.min_words:
            return False
        # Discard if string is purely punctuation/symbols
        if self._punct_regex.match(s):
            return False
        return True

    def split(self, text: str) -> list[str]:
        """Split text into a list of valid sentences."""
        if not text or not text.strip():
            return []

        # Normalize linebreaks and remove weird carriage returns
        cleaned_text = text.replace("\r\n", "\n").replace("\r", "\n")

        if self.tokenizer is not None:
            raw_sentences = self.tokenizer.tokenize(cleaned_text)
        else:
            # Fallback regex sentence splitter
            raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", cleaned_text)

        valid_sentences: list[str] = []
        for raw_s in raw_sentences:
            s = re.sub(r"\s+", " ", raw_s).strip()
            if self.is_valid_sentence(s):
                valid_sentences.append(s)

        return valid_sentences


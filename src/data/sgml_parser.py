"""SGML and XML parser for raw DUC documents.

DUC documents from NIST are historically formatted in SGML/XML format:
<DOC>
  <DOCNO> APW19981016.0240 </DOCNO>
  <DOCTYPE> NEWS STORY </DOCTYPE>
  <TXTTYPE> ... </TXTTYPE>
  <TEXT>
    <P> ... paragraph 1 ... </P>
    <P> ... paragraph 2 ... </P>
  </TEXT>
</DOC>

This module extracts the document ID and cleaned body text from both SGML
and plain-text formatted documents.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


class SGMLDocumentParser:
    """Robust parser for DUC / TREC SGML documents."""

    DOC_REGEX = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL | re.IGNORECASE)
    DOCNO_REGEX = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>", re.DOTALL | re.IGNORECASE)
    TEXT_REGEX = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL | re.IGNORECASE)
    HEADLINE_REGEX = re.compile(r"<HEADLINE>(.*?)</HEADLINE>", re.DOTALL | re.IGNORECASE)
    TAG_REGEX = re.compile(r"<[^>]+>")

    @classmethod
    def clean_text(cls, raw: str) -> str:
        """Strip tags, unescape HTML entities, and normalize whitespace."""
        # Unescape XML entities like &amp;, &quot;, &lt;, &gt;
        text = html.unescape(raw)
        # Strip all tags like <P>, </P>, etc.
        text = cls.TAG_REGEX.sub(" ", text)
        # Normalize whitespace while preserving standard paragraph flow
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    @classmethod
    def parse_file(cls, file_path: str | Path) -> tuple[str, str]:
        """Parse a document file.

        Returns:
            tuple[str, str]: (document_id, cleaned_text)
        """
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="replace").strip()

        # Check if file has SGML tags
        docno_match = cls.DOCNO_REGEX.search(content)
        doc_id = docno_match.group(1).strip() if docno_match else path.stem

        text_matches = cls.TEXT_REGEX.findall(content)
        headline_matches = cls.HEADLINE_REGEX.findall(content)

        if text_matches:
            body_parts: list[str] = []
            if headline_matches:
                body_parts.extend(headline_matches)
            body_parts.extend(text_matches)
            cleaned_text = cls.clean_text("\n".join(body_parts))
        else:
            # File might be plain text without XML tags (or already extracted)
            cleaned_text = cls.clean_text(content)

        return doc_id, cleaned_text


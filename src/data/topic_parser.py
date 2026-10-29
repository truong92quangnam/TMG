"""Topic and query parser for DUC 2005-2007 query-focused summarization.

Supports both JSON files and standard NIST XML/SGML topic definitions:
<topic>
  <num> d301i </num>
  <title> International Red Cross </title>
  <narr> Describe the activities of the International Red Cross. </narr>
</topic>
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class TopicParser:
    """Parser for query topics in JSON or XML/SGML format."""

    TOPIC_BLOCK_REGEX = re.compile(r"<topic>(.*?)</topic>", re.DOTALL | re.IGNORECASE)
    NUM_REGEX = re.compile(r"<num>\s*(.*?)\s*</num>", re.DOTALL | re.IGNORECASE)
    TITLE_REGEX = re.compile(r"<title>\s*(.*?)\s*</title>", re.DOTALL | re.IGNORECASE)
    NARR_REGEX = re.compile(r"<narr>\s*(.*?)\s*</narr>", re.DOTALL | re.IGNORECASE)

    @classmethod
    def parse_json(cls, file_path: str | Path) -> dict[str, str]:
        """Parse topic from JSON format."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cluster_id = str(data.get("cluster_id", path.stem))
        title = str(data.get("title", "")).strip()
        narrative = str(data.get("narrative", "")).strip()
        query = str(data.get("query", f"{title} {narrative}".strip())).strip()

        return {
            "cluster_id": cluster_id,
            "title": title,
            "narrative": narrative,
            "query": query,
        }

    @classmethod
    def parse_xml_topics(cls, file_path: str | Path) -> dict[str, dict[str, str]]:
        """Parse multiple topics from a single XML/SGML topicset file.

        Returns:
            dict[cluster_id, {cluster_id, title, narrative, query}]
        """
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="replace")

        topics: dict[str, dict[str, str]] = {}
        for block in cls.TOPIC_BLOCK_REGEX.findall(content):
            num_m = cls.NUM_REGEX.search(block)
            title_m = cls.TITLE_REGEX.search(block)
            narr_m = cls.NARR_REGEX.search(block)

            cluster_id = num_m.group(1).strip() if num_m else "unknown"
            title = title_m.group(1).strip() if title_m else ""
            narrative = narr_m.group(1).strip() if narr_m else ""

            # Normalize spaces
            title = re.sub(r"\s+", " ", title)
            narrative = re.sub(r"\s+", " ", narrative)
            query = f"{title} {narrative}".strip()

            topics[cluster_id] = {
                "cluster_id": cluster_id,
                "title": title,
                "narrative": narrative,
                "query": query,
            }

        return topics


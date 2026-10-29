"""DUC Dataset Loader module.

Provides unified abstractions for Document, Sentence, and ClusterExample,
with flexible discovery across various DUC directory structures:
- Standard dataset layout: data/{duc2003, duc2004, duc2005, duc2006, duc2007}
- Native / uncompressed DUC distributions (e.g. DUC2004 testdata with reference/)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.data.sgml_parser import SGMLDocumentParser
from src.data.topic_parser import TopicParser

logger = logging.getLogger(__name__)


@dataclass
class Sentence:
    """Represents an extracted sentence with length and content metadata."""
    sentence_id: int
    document_id: str
    original_text: str
    normalized_text: str
    word_count: int
    byte_count: int


@dataclass
class Document:
    """Represents a source document within a cluster."""
    document_id: str
    text: str


@dataclass
class ClusterExample:
    """Represents an evaluation document cluster with documents, sentences, and references."""
    cluster_id: str
    documents: list[Document]
    sentences: list[Sentence] = field(default_factory=list)
    query: str | None = None
    references: list[str] = field(default_factory=list)


class DUCDatasetLoader:
    """Dataset loader supporting DUC 2003-2007 standard and native structures."""

    def __init__(
        self,
        base_dir: str | Path,
        segmenter_fn: Callable[[str], list[str]] | None = None,
        normalizer_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.segmenter_fn = segmenter_fn
        self.normalizer_fn = normalizer_fn

    def _split_into_sentences(
        self, text: str, document_id: str, start_id: int
    ) -> list[Sentence]:
        """Split document text into Sentence objects."""
        if self.segmenter_fn is not None:
            raw_sentences = self.segmenter_fn(text)
        else:
            # Fallback simple sentence splitter
            import re
            raw_sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", text)
                if len(s.strip()) > 5
            ]

        sentences: list[Sentence] = []
        cur_id = start_id
        for raw_s in raw_sentences:
            cleaned_s = raw_s.strip()
            if not cleaned_s:
                continue

            words = cleaned_s.split()
            word_count = len(words)
            byte_count = len(cleaned_s.encode("utf-8"))

            norm_text = (
                self.normalizer_fn(cleaned_s)
                if self.normalizer_fn is not None
                else cleaned_s.lower()
            )

            sentences.append(
                Sentence(
                    sentence_id=cur_id,
                    document_id=document_id,
                    original_text=cleaned_s,
                    normalized_text=norm_text,
                    word_count=word_count,
                    byte_count=byte_count,
                )
            )
            cur_id += 1

        return sentences

    def load_cluster(
        self,
        cluster_id: str,
        doc_paths: list[Path],
        ref_paths: list[Path],
        query: str | None = None,
    ) -> ClusterExample:
        """Load documents and references for a specific cluster."""
        documents: list[Document] = []
        all_sentences: list[Sentence] = []

        sent_counter = 0
        for doc_path in sorted(doc_paths):
            doc_id, text = SGMLDocumentParser.parse_file(doc_path)
            doc = Document(document_id=doc_id, text=text)
            documents.append(doc)

            doc_sentences = self._split_into_sentences(text, doc_id, sent_counter)
            all_sentences.extend(doc_sentences)
            sent_counter += len(doc_sentences)

        references: list[str] = []
        for ref_path in sorted(ref_paths):
            _, ref_text = SGMLDocumentParser.parse_file(ref_path)
            if ref_text.strip():
                references.append(ref_text.strip())

        return ClusterExample(
            cluster_id=cluster_id,
            documents=documents,
            sentences=all_sentences,
            query=query,
            references=references,
        )

    def load_dataset(self) -> list[ClusterExample]:
        """Detect and load all clusters in the configured directory."""
        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: '{self.base_dir}'. "
                f"This year's DUC dataset has not been provided in the workspace yet. "
                f"Please place the NIST data into '{self.base_dir}' or update the config YAML. "
                f"See 'data/README.md' for expected layout."
            )

        # Check for standard directory structure: docs/, [topics/], references/
        docs_dir = self.base_dir / "docs"
        refs_dir = self.base_dir / "references"
        topics_dir = self.base_dir / "topics"

        if docs_dir.exists():
            return self._load_standard_layout(docs_dir, refs_dir, topics_dir)

        # Check for native DUC2004 distribution layout
        # e.g., DUC2004/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs
        native_docs = list(self.base_dir.glob("**/tasks1and2/**/docs"))
        if native_docs:
            native_refs = self.base_dir / "reference"
            if not native_refs.exists():
                native_refs = self.base_dir.parent / "reference"
            return self._load_native_duc2004(native_docs[0], native_refs)

        # Direct search if base_dir itself contains cluster directories
        subdirs = [p for p in self.base_dir.iterdir() if p.is_dir()]
        if subdirs:
            # Check if subdirs contain doc files directly
            clusters: list[ClusterExample] = []
            for subdir in sorted(subdirs):
                doc_files = [f for f in subdir.iterdir() if f.is_file() and not f.name.startswith(".")]
                if doc_files:
                    clusters.append(self.load_cluster(subdir.name, doc_files, []))
            if clusters:
                return clusters

        raise ValueError(f"Could not automatically detect dataset format in {self.base_dir}")

    def _load_standard_layout(
        self, docs_dir: Path, refs_dir: Path, topics_dir: Path
    ) -> list[ClusterExample]:
        """Load dataset from standard data/ducXXXX layout."""
        clusters: list[ClusterExample] = []

        # Load topics if available (for query-focused DUC2005-2007)
        topics: dict[str, str] = {}
        if topics_dir.exists():
            for topic_file in topics_dir.glob("*.json"):
                t_data = TopicParser.parse_json(topic_file)
                topics[t_data["cluster_id"]] = t_data["query"]
            for xml_file in list(topics_dir.glob("*.xml")) + list(topics_dir.glob("*.sgml")):
                x_topics = TopicParser.parse_xml_topics(xml_file)
                for cid, cval in x_topics.items():
                    topics[cid] = cval["query"]

        cluster_dirs = sorted([d for d in docs_dir.iterdir() if d.is_dir()])
        for c_dir in cluster_dirs:
            cid = c_dir.name
            doc_files = sorted([f for f in c_dir.iterdir() if f.is_file() and not f.name.startswith(".")])

            ref_files: list[Path] = []
            ref_cluster_dir = refs_dir / cid
            if ref_cluster_dir.exists():
                ref_files = sorted([f for f in ref_cluster_dir.iterdir() if f.is_file() and not f.name.startswith(".")])
            else:
                # Try matching Task{cid}_* or {cid}_* in refs_dir directly
                ref_files = sorted(
                    list(refs_dir.glob(f"*{cid}*")) + list(refs_dir.glob(f"Task{cid}*"))
                )

            query = topics.get(cid)
            cluster = self.load_cluster(cid, doc_files, ref_files, query)
            clusters.append(cluster)

        logger.info(f"Loaded {len(clusters)} clusters from standard layout at {self.base_dir}")
        return clusters

    def _load_native_duc2004(self, docs_dir: Path, refs_dir: Path) -> list[ClusterExample]:
        """Load dataset from native DUC2004 directory structure."""
        clusters: list[ClusterExample] = []
        cluster_dirs = sorted([d for d in docs_dir.iterdir() if d.is_dir()])

        for c_dir in cluster_dirs:
            cid = c_dir.name
            doc_files = sorted([f for f in c_dir.iterdir() if f.is_file() and not f.name.startswith(".")])

            ref_files: list[Path] = []
            if refs_dir.exists():
                # DUC 2004 format: Task1001_reference1.txt, Task1001_reference2.txt ...
                ref_files = sorted(list(refs_dir.glob(f"Task{cid}_reference*.txt")))
                if not ref_files:
                    ref_files = sorted(list(refs_dir.glob(f"*{cid}*")))

            cluster = self.load_cluster(cid, doc_files, ref_files, query=None)
            clusters.append(cluster)

        logger.info(f"Loaded {len(clusters)} clusters from native DUC 2004 layout at {docs_dir}")
        return clusters


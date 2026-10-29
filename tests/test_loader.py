"""Unit tests for DUC dataset loader and parsers."""

from pathlib import Path
from src.data.duc_loader import DUCDatasetLoader
from src.data.sgml_parser import SGMLDocumentParser
from src.data.topic_parser import TopicParser


def test_sgml_cleaner():
    """Test SGML cleaner removes tags and handles entities."""
    raw = "<DOC><HEADLINE>Breaking News &amp; Events</HEADLINE><TEXT><P>First paragraph.</P><P>Second paragraph.</P></TEXT></DOC>"
    cleaned = SGMLDocumentParser.clean_text(raw)
    assert "<" not in cleaned
    assert "&amp;" not in cleaned
    assert "Breaking News & Events" in cleaned
    assert "First paragraph." in cleaned


def test_loader_real_cluster():
    """Test loading real clusters from data/duc2004."""
    data_dir = Path("data/duc2004")
    if not data_dir.exists():
        return

    loader = DUCDatasetLoader(base_dir=data_dir)
    clusters = loader.load_dataset()

    assert len(clusters) > 0
    c0 = clusters[0]
    assert c0.cluster_id is not None
    assert len(c0.documents) > 0
    assert len(c0.sentences) > 0
    assert c0.sentences[0].word_count > 0
    assert c0.sentences[0].byte_count > 0

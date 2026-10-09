import pytest

from Chunker import chunk_documents
from Loaders import Document


def test_chunks_inherit_metadata():
    doc = Document("a" * 1000, {"source": "x.pdf", "page": 3})
    chunks = chunk_documents([doc])

    for chunk in chunks:
        assert chunk.metadata["source"] == "x.pdf"
        assert chunk.metadata["page"] == 3


def test_overlap_is_real():
    doc = Document("abcdefghijklmnopqrstuvwxyz", {})
    chunks = chunk_documents([doc], chunk_size=10, overlap=3)

    assert chunks[0].page_content[-3:] == chunks[1].page_content[:3]


def test_bad_settings_raise():
    doc = Document("hello world", {})

    with pytest.raises(ValueError):
        chunk_documents([doc], chunk_size=10, overlap=10)


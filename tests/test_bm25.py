import pytest

from BM25Index import BM25Index, tokenize
from Loaders import Document


def test_tokenize_lowercase_and_strip_punctuation():
    tokens = tokenize("Hello, WORLD! Python.")
    assert tokens == ["hello", "world", "python"]


def test_tokenize_hindi_punctuation():
    tokens = tokenize("नमस्ते। आप कैसे हैं?")
    assert tokens == ["नमस्ते", "आप", "कैसे", "हैं"]


def test_tokenize_removes_empty_tokens():
    tokens = tokenize("!!! Hello ... ???")
    assert tokens == ["hello"]


def test_bm25_search_returns_matching_chunks():
    chunks = [
        Document("Python is a programming language", {"page": 1}),
        Document("Machine learning uses algorithms", {"page": 2}),
        Document("Python is used for machine learning", {"page": 3}),
        Document("Gardening guide", {"page": 4}),
        Document("Travel tips", {"page": 5}),
    ]

    index = BM25Index()
    index.add(chunks)

    results = index.search("Python")

    assert len(results) == 2
    assert all(score > 0 for _, score in results)
    assert all(isinstance(score, float) for _, score in results)


def test_bm25_search_ranks_best_match_first():
    chunks = [
        Document("Python programming", {"page": 1}),
        Document("Python Python Python programming", {"page": 2}),
        Document("Cooking recipes", {"page": 3}),
        Document("Travel tips", {"page": 4}),
        Document("Gardening guide", {"page": 5}),
    ]

    index = BM25Index()
    index.add(chunks)

    results = index.search("Python")

    assert results[0][0].metadata["page"] == 2


def test_bm25_add_rebuilds_index():
    index = BM25Index()

    first = Document("Python programming", {"page": 1})
    second = Document("Python machine learning", {"page": 2})

    index.add([first])
    index.add([second])
    index.add([
        Document("Travel tips", {"page": 3}),
        Document("Gardening guide", {"page": 4}),
        Document("Cooking recipes", {"page": 5}),
    ])

    assert len(index.chunks) == 5
    assert len(index.search("Python")) == 2

def test_bm25_empty_index_returns_empty_list():
    index = BM25Index()

    assert index.search("Python") == []


def test_bm25_no_matching_chunks_returns_empty_list():
    index = BM25Index()
    index.add([
        Document("Cooking recipes", {"page": 1}),
        Document("Travel and tourism", {"page": 2}),
    ])

    assert index.search("Python") == []


def test_bm25_respects_k():
    index = BM25Index()
    index.add([
        Document("Python programming", {"page": 1}),
        Document("Python machine learning", {"page": 2}),
        Document("Python development", {"page": 3}),
    ])

    results = index.search("Python", k=2)

    assert len(results) == 2

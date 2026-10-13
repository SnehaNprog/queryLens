from Hybrid import hybrid_search, reciprocal_rank_fusion
from BM25Index import BM25Index
from Loaders import Document
from VectorStore import VectorStore


def test_identical_lists_keep_order():
    doc1 = Document("First document", {"source": "a", "page": 1})
    doc2 = Document("Second document", {"source": "b", "page": 2})
    doc3 = Document("Third document", {"source": "c", "page": 3})

    ranked = [
        (doc1, 0.9),
        (doc2, 0.5),
        (doc3, 0.1)
    ]

    results = reciprocal_rank_fusion(ranked, ranked)

    assert [chunk for chunk, score in results] == [doc1, doc2, doc3]


def test_chunk_in_both_lists_beats_chunk_in_one():
    a = Document("Document A", {"source": "a", "page": 1})
    b = Document("Document B", {"source": "b", "page": 2})
    c = Document("Document C", {"source": "c", "page": 3})

    list_one = [(a, 1.0), (b, 0.9)]
    list_two = [(c, 1.0), (b, 0.9)]

    results = reciprocal_rank_fusion(list_one, list_two)

    assert results[0][0] == b


def test_hybrid_finds_exact_code():
    chunks = [
        Document(
            "Reference code XQ-9931 assigned to order batch seven",
            {"source": "doc", "page": 1, "chunk": 0}
        ),
        Document(
            "Order batch eight was assigned reference code AB-1234",
            {"source": "doc", "page": 2, "chunk": 0}
        ),
        Document(
            "The order batch processing system handles customer orders",
            {"source": "doc", "page": 3, "chunk": 0}
        ),
        Document(
            "Order references are generated during batch processing",
            {"source": "doc", "page": 4, "chunk": 0}
        ),
        Document(
            "Customer order batches are processed every morning",
            {"source": "doc", "page": 5, "chunk": 0}
        ),
    ]

    vector_store = VectorStore()
    bm25_index = BM25Index()

    vector_store.add(chunks)
    bm25_index.add(chunks)

    results = hybrid_search(
        "XQ-9931",
        vector_store,
        bm25_index,
        k=1
    )

    assert len(results) == 1
    assert "XQ-9931" in results[0][0].page_content
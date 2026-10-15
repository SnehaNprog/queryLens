from Loaders import Document
from Retriever import Retriever


DOCS = [
    Document(
        "Photosynthesis lets plants turn sunlight into chemical energy. "
        "Chlorophyll in the leaves absorbs the light.",
        {"source": "bio.pdf", "page": 1, "type": "pdf"}
    ),
    Document(
        "The stock market fell sharply after the central bank raised interest rates. "
        "Investors sold shares across all sectors.",
        {"source": "news.pdf", "page": 1, "type": "pdf"}
    ),
    Document(
        "A chocolate cake needs flour, cocoa, sugar and eggs. Bake it for forty minutes.",
        {"source": "food.pdf", "page": 1, "type": "pdf"}
    ),
]


def build_retriever():
    retriever = Retriever()
    retriever.add(DOCS)
    return retriever


def test_add_fills_both_indexes():
    retriever = build_retriever()

    vector_count = len(retriever.vector_store.chunks)
    bm25_count = len(retriever.bm25_index.chunks)

    assert vector_count > 0
    assert vector_count == bm25_count


def test_search_finds_right_source():
    retriever = build_retriever()

    results = retriever.search(
        "How do plants make energy from light?",
        k=1
    )

    assert len(results) == 1
    assert results[0][0].metadata["source"] == "bio.pdf"


def test_cross_lingual_end_to_end():
    retriever = build_retriever()

    results = retriever.search(
        "पौधे सूरज की रोशनी से ऊर्जा कैसे बनाते हैं?",
        k=1
    )

    assert len(results) == 1
    assert results[0][0].metadata["source"] == "bio.pdf"


def test_save_and_load_roundtrip(tmp_path):
    retriever = build_retriever()

    folder = tmp_path / "idx"
    retriever.save(folder)

    loaded = Retriever.load(folder)

    vector_count = len(loaded.vector_store.chunks)
    bm25_count = len(loaded.bm25_index.chunks)

    assert vector_count == len(retriever.vector_store.chunks)
    assert vector_count == bm25_count

    results = loaded.search("baking a cake", k=1)

    assert len(results) == 1
    assert results[0][0].metadata["source"] == "food.pdf"


def test_search_on_empty_retriever_returns_empty_list():
    retriever = Retriever()

    results = retriever.search("anything")

    assert results == []

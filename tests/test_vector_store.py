from querylens.ingestion.loaders import Document
from querylens.retrieval.vector_store import VectorStore

CHUNKS = [
    Document("Photosynthesis lets plants turn sunlight into chemical energy.", {"source": "bio.pdf", "page": 1}),
    Document("The stock market fell sharply after the interest rate announcement.", {"source": "news.pdf", "page": 4}),
    Document("A chocolate cake needs flour, cocoa, sugar and eggs.", {"source": "food.pdf", "page": 2}),
]


def build_store():
    store = VectorStore()
    store.add(CHUNKS)
    return store


def test_search_finds_relevant_chunk():
    results = build_store().search("How do plants make energy from light?", k=1)
    chunk, score = results[0]
    assert chunk.metadata["source"] == "bio.pdf"


def test_cross_lingual_search():
    # Hindi question, English documents
    results = build_store().search("पौधे सूरज की रोशनी से ऊर्जा कैसे बनाते हैं?", k=1)
    assert results[0][0].metadata["source"] == "bio.pdf"


def test_results_are_sorted_best_first():
    results = build_store().search("baking a cake", k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0].metadata["source"] == "food.pdf"


def test_save_and_load_roundtrip(tmp_path):
    original = build_store()
    original.save(tmp_path / "idx")

    loaded = VectorStore.load(tmp_path / "idx")
    assert len(loaded.chunks) == 3
    assert loaded.chunks[1].metadata == {"source": "news.pdf", "page": 4}

    query = "interest rates and the stock market"
    assert loaded.search(query, k=1)[0][0].page_content == original.search(query, k=1)[0][0].page_content


def test_k_larger_than_store_returns_all():
    assert len(build_store().search("anything", k=10)) == 3


def test_empty_store_search_returns_nothing():
    assert VectorStore().search("anything") == []

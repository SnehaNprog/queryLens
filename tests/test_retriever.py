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


def test_add_fills_the_index():
    retriever = build_retriever()

    assert len(retriever.vector_store.chunks) > 0


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

    assert len(loaded.vector_store.chunks) == len(retriever.vector_store.chunks)

    results = loaded.search("baking a cake", k=1)

    assert len(results) == 1
    assert results[0][0].metadata["source"] == "food.pdf"


def test_search_on_empty_retriever_returns_empty_list():
    retriever = Retriever()

    results = retriever.search("anything")

    assert results == []



def test_sources_lists_loaded_sources():
    retriever = build_retriever()

    assert retriever.sources() == [
        "bio.pdf",
        "food.pdf",
        "news.pdf",
    ]

    assert Retriever().sources() == []


def test_adding_same_source_twice_is_skipped():
    retriever = build_retriever()

    n = len(retriever.vector_store.chunks)

    added = retriever.add(DOCS)

    assert added == 0
    assert len(retriever.vector_store.chunks) == n



def test_add_returns_number_of_chunks_added():
    retriever = Retriever()

    added = retriever.add(DOCS)

    assert added == len(retriever.vector_store.chunks)
    assert added > 0


def test_adding_a_new_source_after_load_works(tmp_path):
    retriever = build_retriever()
    retriever.save(tmp_path / "idx")

    loaded = Retriever.load(tmp_path / "idx")
    added = loaded.add([Document("Mars is red.", {"source": "space.pdf", "page": 1})])

    assert added > 0
    assert loaded.sources() == ["bio.pdf", "food.pdf", "news.pdf", "space.pdf"]

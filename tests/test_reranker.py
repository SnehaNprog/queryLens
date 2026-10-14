from Loaders import Document
from Reranker import rerank


CANDIDATES = [
    (Document("Berlin is the capital of Germany.", {"page": 1}), 0.0),
    (Document("Paris is the capital and largest city of France.", {"page": 2}), 0.0),
    (Document("France is known for its wine and cheese.", {"page": 3}), 0.0),
]


def test_rerank_puts_best_answer_first():
    results = rerank("What is the capital of France?", CANDIDATES)

    assert results[0][0].metadata["page"] == 2


def test_rerank_respects_top_k():
    results = rerank("What is the capital of France?", CANDIDATES, top_k=2)

    assert len(results) == 2


def test_rerank_scores_are_sorted_best_first():
    results = rerank("What is the capital of France?", CANDIDATES)

    scores = [score for chunk, score in results]

    assert scores == sorted(scores, reverse=True)


def test_rerank_cross_lingual():
    results = rerank("फ्रांस की राजधानी क्या है?", CANDIDATES)

    assert results[0][0].metadata["page"] == 2


def test_rerank_empty_input_returns_empty_list():
    results = rerank("anything", [])

    assert results == []
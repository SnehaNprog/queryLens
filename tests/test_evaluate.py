import pytest

from Loaders import Document
from evaluate import first_hit_rank, summarize


def result(source, text):
    return (Document(text, {"source": source}), 0.0)


def test_first_hit_rank_finds_the_matching_chunk():
    results = [
        result("other", "The answer is 42"),       # right text, wrong document
        result("doc_a", "something unrelated"),     # right document, no evidence
        result("doc_a", "here the answer is 42"),   # both: rank 3
    ]
    assert first_hit_rank(results, "doc_a", "answer is 42") == 3


def test_first_hit_rank_is_none_when_nothing_matches():
    assert first_hit_rank([result("doc_a", "nope")], "doc_a", "answer") is None
    assert first_hit_rank([], "doc_a", "answer") is None


def test_summarize_computes_hits_and_mrr():
    records = [
        {"rank": 1, "ms": 10},
        {"rank": 2, "ms": 20},
        {"rank": 4, "ms": 30},
        {"rank": None, "ms": 40},
    ]
    s = summarize(records)

    assert s["n"] == 4
    assert s["hit@1"] == 0.25
    assert s["hit@3"] == 0.5
    assert s["hit@5"] == 0.75
    assert s["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4 + 0) / 4)
    assert s["median_ms"] == 25


def test_summarize_of_nothing_is_empty():
    assert summarize([]) == {"n": 0}

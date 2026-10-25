import json
from collections import Counter
from pathlib import Path

from Chunker import chunk_documents
from Loaders import Document

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def load_docs():
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted((EVAL_DIR / "docs").glob("*.txt"))
    }


def load_questions():
    return json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))


def doc_language(doc_id):
    """The document language is the suffix of its file name: obs_hi -> 'hi'."""
    return doc_id.rsplit("_", 1)[1]


def test_question_ids_are_unique():
    ids = [q["id"] for q in load_questions()]
    assert len(set(ids)) == len(ids)


def test_every_question_points_to_an_existing_document():
    docs = load_docs()
    for q in load_questions():
        assert q["doc"] in docs, f"{q['id']} points to unknown doc {q['doc']}"


def test_every_evidence_string_is_in_its_document():
    docs = load_docs()
    for q in load_questions():
        assert q["evidence"] in docs[q["doc"]], f"{q['id']}: evidence not in {q['doc']}"


def test_every_evidence_string_survives_chunking():
    """Chunking must not cut an evidence string in half, or the question is unanswerable."""
    docs = load_docs()
    for q in load_questions():
        chunks = chunk_documents(
            [Document(docs[q["doc"]], {"source": q["doc"]})], chunk_size=500, overlap=100
        )
        assert any(q["evidence"] in c.page_content for c in chunks), (
            f"{q['id']}: evidence is split across chunks"
        )


def test_most_questions_are_cross_lingual():
    questions = load_questions()
    cross = [q for q in questions if q["lang"] != doc_language(q["doc"])]
    assert len(cross) / len(questions) >= 0.8


def test_benchmark_covers_several_languages():
    assert {q["lang"] for q in load_questions()} >= {"en", "hi", "es", "fr", "de"}
    assert {doc_language(d) for d in load_docs()} >= {"en", "hi", "es", "fr", "de"}


def test_every_document_has_questions():
    asked = Counter(q["doc"] for q in load_questions())
    assert set(asked) == set(load_docs())

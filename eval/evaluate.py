"""Multilingual retrieval benchmark.

Compares two retrieval systems on eval/questions.json:
    vector          bge-m3 embeddings + FAISS
    vector+rerank   the same shortlist re-scored by bge-reranker-v2-m3

A question counts as a hit at rank r when the r-th retrieved chunk comes from the
correct document AND contains the question's evidence string.

Usage (from the project root):
    python -m eval.evaluate
    python -m eval.evaluate --distractors
    python -m eval.evaluate --distractors --distractor-url https://en.wikipedia.org/wiki/Photosynthesis
"""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from querylens.ingestion.loaders import Document, pdfloader, webloader
from querylens.retrieval.reranker import RERANKER_MODEL, rerank
from querylens.retrieval.embedder import MODEL_NAME
from querylens.retrieval.retriever import Retriever

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
TOP_N = 10  # how many results each system returns per question
CANDIDATES = 20  # shortlist size handed to the reranker


# ---------- pure metric functions (unit-tested) ----------

def first_hit_rank(results, doc_id, evidence):
    """1-based rank of the first chunk from `doc_id` containing `evidence`, else None.

    `results` is a list of (chunk, score) pairs, best first.
    """
    for rank, (chunk, _) in enumerate(results, start=1):
        if chunk.metadata.get("source") == doc_id and evidence in chunk.page_content:
            return rank
    return None


def summarize(records):
    """hit@1/3/5, MRR, and median latency for a list of {"rank", "ms"} records."""
    n = len(records)
    if n == 0:
        return {"n": 0}

    def hit_at(k):
        return sum(1 for r in records if r["rank"] is not None and r["rank"] <= k) / n

    return {
        "n": n,
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "mrr": sum(1 / r["rank"] for r in records if r["rank"] is not None) / n,
        "median_ms": statistics.median(r["ms"] for r in records),
    }


# ---------- loading ----------

def load_benchmark():
    docs = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted((EVAL_DIR / "docs").glob("*.txt"))
    }
    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    return docs, questions


def doc_language(doc_id):
    return doc_id.rsplit("_", 1)[1]


def load_distractors(urls):
    """Unrelated documents that make the haystack bigger (the answer must be found among them)."""
    docs = [Document((ROOT / "data" / "sample.txt").read_text(encoding="utf-8"),
                     {"source": "sample.txt", "type": "text"})]
    docs += pdfloader(ROOT / "data" / "sample.pdf")
    for url in urls:
        docs += webloader(url)
    return docs


# ---------- running ----------

def build_systems(retriever):
    """name -> search(query) returning [(chunk, score), ...]."""
    vs = retriever.vector_store

    return {
        "vector": lambda q: vs.search(q, k=TOP_N),
        "vector+rerank": lambda q: rerank(q, vs.search(q, k=CANDIDATES), top_k=TOP_N),
    }


def run_system(search, questions):
    search(questions[0]["question"])  # warm-up, so model loading is not timed
    records = []
    for q in questions:
        start = time.perf_counter()
        results = search(q["question"])
        ms = (time.perf_counter() - start) * 1000
        records.append({
            "id": q["id"],
            "lang": q["lang"],
            "cross_lingual": q["lang"] != doc_language(q["doc"]),
            "rank": first_hit_rank(results, q["doc"], q["evidence"]),
            "ms": ms,
        })
    return records


def format_table(title, rows):
    lines = [f"\n{title}", "| System | n | hit@1 | hit@3 | hit@5 | MRR | median ms |",
             "|---|---|---|---|---|---|---|"]
    for name, s in rows:
        lines.append(
            f"| {name} | {s['n']} | {s['hit@1']:.2f} | {s['hit@3']:.2f} | "
            f"{s['hit@5']:.2f} | {s['mrr']:.2f} | {s['median_ms']:.0f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--distractors", action="store_true",
                        help="also index data/sample.txt and data/sample.pdf as unrelated documents")
    parser.add_argument("--distractor-url", action="append", default=[],
                        help="also index this webpage as a distractor (repeatable; implies --distractors)")
    parser.add_argument("--out", default=str(EVAL_DIR / "results" / "latest.json"))
    args = parser.parse_args()

    docs, questions = load_benchmark()
    benchmark_docs = [Document(text, {"source": doc_id, "type": "benchmark"}) for doc_id, text in docs.items()]

    retriever = Retriever()
    retriever.add(benchmark_docs)
    benchmark_chunks = len(retriever.vector_store.chunks)

    distractors = args.distractors or bool(args.distractor_url)
    if distractors:
        retriever.add(load_distractors(args.distractor_url))
    total_chunks = len(retriever.vector_store.chunks)

    print(f"{len(docs)} documents, {len(questions)} questions, "
          f"{benchmark_chunks} benchmark chunks, {total_chunks} chunks indexed in total")

    results = {name: run_system(search, questions) for name, search in build_systems(retriever).items()}

    print(format_table("Overall", [(n, summarize(r)) for n, r in results.items()]))

    cross = [(n, summarize([x for x in r if x["cross_lingual"]])) for n, r in results.items()]
    same = [(n, summarize([x for x in r if not x["cross_lingual"]])) for n, r in results.items()]
    print(format_table("Cross-lingual questions (question language differs from document language)", cross))
    print(format_table("Same-language questions", same))

    for lang in sorted({q["lang"] for q in questions}):
        rows = [(n, summarize([x for x in r if x["lang"] == lang])) for n, r in results.items()]
        print(format_table(f"Questions asked in '{lang}'", rows))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": MODEL_NAME,
        "reranker_model": RERANKER_MODEL,
        "documents": len(docs),
        "questions": len(questions),
        "benchmark_chunks": benchmark_chunks,
        "total_chunks": total_chunks,
        "distractors": distractors,
        "distractor_urls": args.distractor_url,
        "summary": {name: summarize(r) for name, r in results.items()},
        "per_question": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder once and reuse it. First call downloads ~2 GB."""
    return CrossEncoder(RERANKER_MODEL)


def rerank(query, results, top_k=5):
    """Re-score (chunk, score) pairs by reading the query and each chunk together.

    A bi-encoder (bge-m3) embeds query and chunk separately; a cross-encoder
    reads them as one input, which is slower but much more accurate. So we run
    it only on the few candidates that hybrid search already shortlisted.

    Returns [(chunk, relevance_score), ...], best first, at most top_k.
    """
    if not results:
        return []
    chunks = [chunk for chunk, _ in results]
    pairs = [(query, chunk.page_content) for chunk in chunks]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
    return [(chunk, float(score)) for chunk, score in ranked[:top_k]]


if __name__ == "__main__":
    from querylens.ingestion.loaders import Document

    candidates = [
        (Document("Paris is the capital and largest city of France."), 0.0),
        (Document("The Eiffel Tower is a famous landmark in Paris."), 0.0),
        (Document("Berlin is the capital of Germany."), 0.0),
        (Document("France is known for its wine and cheese."), 0.0),
    ]
    for question in ["What is the capital of France?", "फ्रांस की राजधानी क्या है?"]:
        print(question)
        for chunk, score in rerank(question, candidates, top_k=3):
            print(f"  {score:.3f}  {chunk.page_content}")

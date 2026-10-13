RRF_K = 60


def chunk_key(chunk):
    return (
        chunk.metadata.get("source"),
        chunk.metadata.get("page"),
        chunk.metadata.get("chunk")
    )


def reciprocal_rank_fusion(*ranked_lists, k=RRF_K):
    totals = {}
    chunks = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, score) in enumerate(ranked_list, start=1):
            key = chunk_key(chunk)

            totals[key] = totals.get(key, 0) + 1 / (k + rank)
            chunks[key] = chunk

    return sorted(
        [(chunks[key], total) for key, total in totals.items()],
        key=lambda item: item[1],
        reverse=True
    )


def hybrid_search(query, vector_store, bm25_index, k=5, candidates=20):
    vector_results = vector_store.search(query, k=candidates)
    bm25_results = bm25_index.search(query, k=candidates)

    fused_results = reciprocal_rank_fusion(
        vector_results,
        bm25_results
    )

    return fused_results[:k]
from querylens.ingestion.chunker import chunk_documents
from querylens.retrieval.reranker import rerank
from querylens.retrieval.vector_store import VectorStore


class Retriever:
    """One object for the whole retrieval pipeline.

    add():    documents -> chunks -> FAISS index (bge-m3 embeddings)
    search(): vector search shortlist -> rerank (bge-reranker-v2-m3) -> top results
    """

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vector_store = VectorStore()

    def sources(self):
        """Sorted list of distinct sources currently loaded."""
        return sorted({
            chunk.metadata["source"]
            for chunk in self.vector_store.chunks
        })

    def add(self, docs):
        """Add new sources to the index. Skip already-loaded sources.

        Returns the number of chunks added.
        """
        # Snapshot existing sources BEFORE filtering.
        known = set(self.sources())

        docs = [
            doc for doc in docs
            if doc.metadata["source"] not in known
        ]

        if not docs:
            return 0

        chunks = chunk_documents(
            docs,
            self.chunk_size,
            self.overlap
        )

        self.vector_store.add(chunks)

        return len(chunks)

    def search(self, query, k=5, candidates=20):
        """Return the k best (chunk, relevance_score) pairs, best first."""
        shortlist = self.vector_store.search(query, k=candidates)

        return rerank(query, shortlist, top_k=k)

    def save(self, folder):
        """Persist the index so documents never need re-embedding."""
        self.vector_store.save(folder)

    @classmethod
    def load(cls, folder, chunk_size=500, overlap=100):
        retriever = cls(chunk_size, overlap)
        retriever.vector_store = VectorStore.load(folder)

        return retriever

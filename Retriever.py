from BM25Index import BM25Index
from Chunker import chunk_documents
from Hybrid import hybrid_search
from Reranker import rerank
from VectorStore import VectorStore


class Retriever:
    """One object for the whole retrieval pipeline.

    add():    documents -> chunks -> FAISS index + BM25 index
    search(): hybrid search (meaning + keywords) -> rerank -> top results
    """

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()

    def add(self, docs):
        """Chunk the documents and add the chunks to BOTH indexes."""
        chunks = chunk_documents(docs, self.chunk_size, self.overlap)

        self.vector_store.add(chunks)
        self.bm25_index.add(chunks)

    def search(self, query, k=5, candidates=20):
        """Return the k best (chunk, relevance_score) pairs, best first."""
        shortlist = hybrid_search(
            query,
            self.vector_store,
            self.bm25_index,
            k=candidates,
            candidates=candidates
        )

        return rerank(query, shortlist, top_k=k)

    def save(self, folder):
        """Persist the vector index. BM25 is rebuilt during loading."""
        self.vector_store.save(folder)

    @classmethod
    def load(cls, folder, chunk_size=500, overlap=100):
        retriever = cls(chunk_size, overlap)

        retriever.vector_store = VectorStore.load(folder)

        retriever.bm25_index.add(retriever.vector_store.chunks)

        return retriever

    
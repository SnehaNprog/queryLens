import json
from pathlib import Path

import faiss
import numpy as np

from querylens.retrieval.embedder import embed_texts
from querylens.ingestion.loaders import Document

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


class VectorStore:
    """FAISS index plus the chunks it points to.

    FAISS only stores vectors and returns row numbers. We keep the chunk
    texts and metadata in a parallel list: row i of the index is chunks[i].
    """

    def __init__(self):
        self.index = None
        self.chunks = []

    def add(self, chunks):
        """Embed chunks and add them to the index."""
        if not chunks:
            return
        vectors = np.asarray(
            embed_texts([chunk.page_content for chunk in chunks]), dtype="float32"
        )
        if self.index is None:
            # Inner product on unit-length vectors equals cosine similarity.
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query, k=5):
        """Return the k most similar chunks as (Document, score) pairs, best first."""
        if self.index is None:
            return []
        query_vector = np.asarray(embed_texts([query]), dtype="float32")
        scores, ids = self.index.search(query_vector, min(k, len(self.chunks)))
        return [
            (self.chunks[i], float(score))
            for score, i in zip(scores[0], ids[0])
            if i != -1  # FAISS pads with -1 when there are fewer than k results
        ]

    def save(self, folder):
        """Write the index and chunks to a folder so they never need re-embedding."""
        if self.index is None:
            raise ValueError("nothing to save: the store is empty")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / INDEX_FILE))
        data = [
            {"page_content": c.page_content, "metadata": c.metadata}
            for c in self.chunks
        ]
        (folder / CHUNKS_FILE).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, folder):
        """Read a saved store back from disk."""
        folder = Path(folder)
        store = cls()
        store.index = faiss.read_index(str(folder / INDEX_FILE))
        data = json.loads((folder / CHUNKS_FILE).read_text(encoding="utf-8"))
        store.chunks = [Document(d["page_content"], d["metadata"]) for d in data]
        return store

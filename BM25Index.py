import string

from rank_bm25 import BM25Okapi

# Punctuation to strip from word edges (includes the Hindi danda "।")
PUNCTUATION = string.punctuation + "।॥“”‘’«»¿¡"


def tokenize(text):
    """Lowercase, split on whitespace, strip punctuation from each word's edges."""
    return [
        word.strip(PUNCTUATION)
        for word in text.lower().split()
        if word.strip(PUNCTUATION)
    ]


class BM25Index:
    """Keyword search over chunks. Row i of the index is chunks[i], like VectorStore."""

    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def add(self, chunks):
        self.chunks.extend(chunks)

        if self.chunks:
            self.bm25 = BM25Okapi(
                [tokenize(c.page_content) for c in self.chunks]
            )

    def search(self, query, k=5):
        """Return the k best (Document, score) pairs, best first.
        Only return chunks that actually match (score > 0).
        """
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(tokenize(query))

        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = []

        for i in indices:
            if scores[i] > 0:
                results.append((self.chunks[i], float(scores[i])))

        return results
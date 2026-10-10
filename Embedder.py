from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def get_model():
    """Load bge-m3 once and reuse it. The first call downloads the weights (~2 GB)."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts, batch_size=16):
    """Turn a list of strings into a (len(texts), 1024) array of unit-length vectors.

    Vectors are normalized to length 1, so cosine similarity between two texts
    is just the dot product of their vectors.
    """
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


if __name__ == "__main__":
    sentences = [
        "What is artificial intelligence?",
        "कृत्रिम बुद्धिमत्ता क्या है?",
        "How do I bake a chocolate cake?",
    ]
    vectors = embed_texts(sentences)
    print("shape:", vectors.shape)
    print("English vs Hindi (same meaning):", float(vectors[0] @ vectors[1]))
    print("English vs cake (different meaning):", float(vectors[0] @ vectors[2]))

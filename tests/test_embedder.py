import numpy as np

from Embedder import embed_texts


def test_one_vector_per_text():
    vectors = embed_texts(["hello", "world", "again"])
    assert vectors.shape == (3, 1024)


def test_vectors_are_normalized():
    vectors = embed_texts(["hello"])
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-4)


def test_cross_lingual_similarity():
    english, hindi, unrelated = embed_texts(
        [
            "What is artificial intelligence?",
            "कृत्रिम बुद्धिमत्ता क्या है?",
            "How do I bake a chocolate cake?",
        ]
    )
    assert english @ hindi > english @ unrelated

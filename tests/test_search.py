import numpy as np
import pytest
from scipy import sparse

from similarity.encoders import ColorHistogramEncoder, TfidfEncoder
from similarity.search import normalize, scores, search, top_k


def test_normalize_gives_length_one():
    v = normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.linalg.norm(v, axis=1) == pytest.approx([1.0, 1.0])


def test_normalize_handles_zero_and_single_vector():
    v = normalize(np.array([0.0, 0.0]))
    assert v.shape == (1, 2)
    assert not np.isnan(v).any()


def test_top_k_order():
    index = normalize(np.array([[1, 0], [0, 1], [0.7, 0.7], [-1, 0]]))
    hits, sims = top_k(index, np.array([[1, 0.1]]), k=3)
    assert hits.tolist() == [0, 2, 1]
    assert list(sims) == sorted(sims, reverse=True)


def test_top_k_bigger_than_index():
    index = normalize(np.eye(3))
    hits, _ = top_k(index, np.array([[1, 0, 0]]), k=10)
    assert len(hits) == 3


def test_search_uses_any_encoder():
    class FakeEncoder:
        name = "fake"

        def encode(self, items):
            return np.array([[1.0, 0.0] if "cat" in item else [0.0, 1.0] for item in items])

    encoder = FakeEncoder()
    index = normalize(encoder.encode(["dog", "cat", "dog park"]))
    hits, _ = search(encoder, index, "a cat", k=1)
    assert hits[0] == 1


def test_tfidf_encoder_with_sparse_search():
    docs = ["rockets and space shuttles", "baseball pitchers and home runs", "space station orbit launch"]
    encoder = TfidfEncoder(docs * 2)
    index = normalize(encoder.encode(docs))
    assert sparse.issparse(index)
    hits, _ = top_k(index, encoder.encode(["space launch"]), k=1)
    assert hits[0] in (0, 2)
    assert len(scores(index, encoder.encode(["baseball"]))) == 3


def test_color_histogram_matches_same_color():
    from PIL import Image
    red = Image.new("RGB", (64, 64), (220, 20, 20))
    red2 = Image.new("RGB", (64, 64), (210, 30, 25))
    blue = Image.new("RGB", (64, 64), (20, 20, 220))
    encoder = ColorHistogramEncoder()
    index = normalize(encoder.encode([blue, red]))
    hits, _ = top_k(index, encoder.encode([red2]), k=1)
    assert hits[0] == 1

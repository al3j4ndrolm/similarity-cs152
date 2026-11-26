import numpy as np
from scipy import sparse


def normalize(vectors):
    if sparse.issparse(vectors):
        from sklearn.preprocessing import normalize as sparse_normalize
        return sparse_normalize(vectors).tocsr()
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1
    return vectors / lengths


def scores(index, query):
    query = normalize(query)
    s = index @ query[0].T if sparse.issparse(query) else index @ query[0]
    return s.toarray().ravel() if sparse.issparse(s) else np.asarray(s).ravel()


def top_k(index, query, k=10):
    s = scores(index, query)
    k = min(k, len(s))
    best = np.argpartition(-s, k - 1)[:k]
    best = best[np.argsort(-s[best])]
    return best, s[best]


def search(encoder, index, query, k=10):
    query_vector = encoder.encode([query])
    return top_k(index, query_vector, k)

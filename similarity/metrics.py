import numpy as np
from scipy import sparse


def similarity_matrix(query_vectors, db_vectors):
    sims = query_vectors @ db_vectors.T
    return sims.toarray() if sparse.issparse(sims) else np.asarray(sims)


def relevance(query_vectors, query_labels, db_vectors, db_labels):
    sims = similarity_matrix(query_vectors, db_vectors)
    order = np.argsort(-sims, axis=1, kind="stable")
    return db_labels[order] == query_labels[:, None]


def precision_at_k(rel, k):
    return rel[:, :k].mean(axis=1)


def recall_at_k(rel, k, total_relevant):
    return rel[:, :k].sum(axis=1) / np.maximum(total_relevant, 1)


def average_precision(rel, total_relevant):
    hits = np.cumsum(rel, axis=1)
    ranks = np.arange(1, rel.shape[1] + 1)
    return (hits / ranks * rel).sum(axis=1) / np.maximum(total_relevant, 1)


def evaluate(query_vectors, query_labels, db_vectors, db_labels, ks=(1, 5, 10, 20, 50, 100), chunk=512):
    query_labels = np.asarray(query_labels)
    db_labels = np.asarray(db_labels)
    label_counts = {label: count for label, count in zip(*np.unique(db_labels, return_counts=True))}
    total_relevant = np.array([label_counts.get(label, 0) for label in query_labels])

    p = {k: [] for k in ks}
    r = {k: [] for k in ks}
    ap = []
    for start in range(0, len(query_labels), chunk):
        end = start + chunk
        rel = relevance(query_vectors[start:end], query_labels[start:end], db_vectors, db_labels)
        for k in ks:
            p[k].append(precision_at_k(rel, k))
            r[k].append(recall_at_k(rel, k, total_relevant[start:end]))
        ap.append(average_precision(rel, total_relevant[start:end]))

    ap = np.concatenate(ap)
    result = {"queries": int(len(query_labels)), "database": int(len(db_labels)), "mAP": float(ap.mean())}
    for k in ks:
        result[f"P@{k}"] = float(np.concatenate(p[k]).mean())
        result[f"R@{k}"] = float(np.concatenate(r[k]).mean())
    return result, ap


def ap_by_class(ap, query_labels, names):
    query_labels = np.asarray(query_labels)
    rows = []
    for label in np.unique(query_labels):
        rows.append((names[label], float(ap[query_labels == label].mean()), int((query_labels == label).sum())))
    return sorted(rows, key=lambda row: row[1])


def top1_confusion(query_vectors, query_labels, db_vectors, db_labels, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for start in range(0, len(query_labels), 1024):
        sims = similarity_matrix(query_vectors[start:start + 1024], db_vectors)
        first = db_labels[np.argmax(sims, axis=1)]
        for true, found in zip(query_labels[start:start + 1024], first):
            matrix[true, found] += 1
    return matrix

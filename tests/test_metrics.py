import numpy as np
import pytest

from similarity.metrics import ap_by_class, average_precision, evaluate, precision_at_k, recall_at_k, top1_confusion


def clusters(per_class=4, classes=3, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    centers = np.eye(classes) * 10
    vectors = np.vstack([centers[c] + rng.normal(0, noise, classes) for c in range(classes) for _ in range(per_class)])
    labels = np.repeat(np.arange(classes), per_class)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype(np.float32), labels


def test_average_precision_by_hand():
    rel = np.array([[True, False, True, False]])
    assert average_precision(rel, np.array([2]))[0] == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_average_precision_counts_missed_items():
    rel = np.array([[True, False, False]])
    assert average_precision(rel, np.array([4]))[0] == pytest.approx(0.25)


def test_precision_and_recall():
    rel = np.array([[True, True, False, False], [False, True, False, True]])
    assert precision_at_k(rel, 2).tolist() == [1.0, 0.5]
    assert recall_at_k(rel, 2, np.array([2, 4])).tolist() == [1.0, 0.25]


def test_perfect_clusters_score_one():
    db, db_labels = clusters(seed=1)
    q, q_labels = clusters(seed=2)
    result, ap = evaluate(q, q_labels, db, db_labels, ks=(1, 4, 8))
    assert result["mAP"] == pytest.approx(1.0)
    assert result["P@4"] == pytest.approx(1.0)
    assert result["R@4"] == pytest.approx(1.0)
    assert result["P@8"] == pytest.approx(0.5)
    assert len(ap) == len(q_labels)


def test_chunks_do_not_change_results():
    rng = np.random.default_rng(0)
    db = rng.normal(size=(50, 8)).astype(np.float32)
    q = rng.normal(size=(23, 8)).astype(np.float32)
    db_labels = rng.integers(0, 5, 50)
    q_labels = rng.integers(0, 5, 23)
    small, _ = evaluate(q, q_labels, db, db_labels, ks=(1, 5), chunk=4)
    big, _ = evaluate(q, q_labels, db, db_labels, ks=(1, 5), chunk=1000)
    assert small == pytest.approx(big)


def test_random_vectors_score_near_chance():
    rng = np.random.default_rng(0)
    db = rng.normal(size=(2000, 32)).astype(np.float32)
    q = rng.normal(size=(300, 32)).astype(np.float32)
    db_labels = rng.integers(0, 10, 2000)
    q_labels = rng.integers(0, 10, 300)
    result, _ = evaluate(q, q_labels, db, db_labels, ks=(10,))
    assert result["P@10"] == pytest.approx(0.1, abs=0.03)


def test_ap_by_class_sorted():
    ap = np.array([0.9, 0.7, 0.2, 0.4])
    rows = ap_by_class(ap, np.array([0, 0, 1, 1]), ["a", "b"])
    assert rows[0][0] == "b"
    assert rows[0][1] == pytest.approx(0.3)
    assert rows[1][1] == pytest.approx(0.8)


def test_confusion_diagonal_for_clusters():
    db, db_labels = clusters(seed=1)
    q, q_labels = clusters(seed=2)
    matrix = top1_confusion(q, q_labels, db, db_labels, 3)
    assert np.trace(matrix) == len(q_labels)

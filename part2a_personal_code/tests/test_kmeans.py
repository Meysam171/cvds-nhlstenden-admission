"""Tests for the K-Means implementation."""

import numpy as np
import pytest

from algorithms.kmeans import KMeans


def _make_blobs(
    centers: np.ndarray,
    samples_per_cluster: int,
    std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pieces = [rng.normal(loc=c, scale=std, size=(samples_per_cluster, c.size)) for c in centers]
    x = np.vstack(pieces)
    y = np.repeat(np.arange(len(centers)), samples_per_cluster)
    perm = rng.permutation(x.shape[0])
    return x[perm], y[perm]


def test_recovers_well_separated_clusters() -> None:
    true_centers = np.array([[-5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    x, y_true = _make_blobs(true_centers, samples_per_cluster=80, std=0.3, seed=0)

    km = KMeans(n_clusters=3, random_state=0).fit(x)

    # Each predicted cluster must correspond 1-to-1 with a true cluster.
    mapping = {}
    for label in np.unique(km.labels_):
        mask = km.labels_ == label
        majority = np.bincount(y_true[mask]).argmax()
        mapping[label] = majority
    assert len(set(mapping.values())) == 3

    matched = np.array([mapping[l] for l in km.labels_])
    accuracy = float((matched == y_true).mean())
    assert accuracy > 0.99


def test_reproducible_with_random_state() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 4))
    a = KMeans(n_clusters=5, random_state=123).fit(x)
    b = KMeans(n_clusters=5, random_state=123).fit(x)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)
    assert a.inertia_ == pytest.approx(b.inertia_)


def test_predict_matches_labels_on_training_data() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(150, 3))
    km = KMeans(n_clusters=4, random_state=1).fit(x)
    np.testing.assert_array_equal(km.predict(x), km.labels_)


def test_inertia_decreases_with_more_clusters() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 2))
    inertias = [KMeans(n_clusters=k, random_state=0).fit(x).inertia_ for k in (1, 2, 4, 8)]
    assert inertias == sorted(inertias, reverse=True)


def test_single_cluster_centroid_equals_mean() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(loc=[10.0, -3.0], size=(50, 2))
    km = KMeans(n_clusters=1, random_state=0).fit(x)
    np.testing.assert_allclose(km.cluster_centers_[0], x.mean(axis=0), atol=1e-9)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        KMeans().predict(np.zeros((1, 2)))


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    with pytest.raises(ValueError):
        KMeans(max_iter=0)
    with pytest.raises(ValueError):
        KMeans().fit(np.zeros((0, 2)))
    with pytest.raises(ValueError):
        KMeans().fit(np.zeros(5))
    with pytest.raises(ValueError):
        KMeans(n_clusters=10).fit(np.zeros((3, 2)))

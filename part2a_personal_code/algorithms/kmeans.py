"""From-scratch K-Means clustering with k-means++ initialisation.

The implementation is fully vectorised with NumPy. It is small enough to read
in one sitting but covers the parts that matter in practice: a numerically
stable distance computation, the standard k-means++ seeding to avoid bad local
minima, an early-stopping criterion on centroid movement, and a deterministic
``random_state`` argument so experiments are reproducible.

The public API mirrors ``sklearn.cluster.KMeans`` for the methods that are
strictly necessary (``fit``, ``predict``, ``fit_predict``) - this makes the
class drop-in for the common case while remaining transparent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class KMeans:
    """K-Means clustering using Lloyd's algorithm with k-means++ seeding.

    Attributes are populated after :meth:`fit` is called. The convention with a
    trailing underscore (``cluster_centers_``, ``labels_``, ``inertia_``) is
    deliberate: it follows the scikit-learn idiom for fitted attributes.
    """

    n_clusters: int = 8
    max_iter: int = 300
    tol: float = 1e-4
    random_state: Optional[int] = None

    cluster_centers_: FloatArray = field(init=False, repr=False)
    labels_: IntArray = field(init=False, repr=False)
    inertia_: float = field(init=False, default=float("nan"))
    n_iter_: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    # ------------------------------------------------------------------ API

    def fit(self, x: FloatArray) -> "KMeans":
        """Fit the model on the data ``x`` of shape ``(n_samples, n_features)``."""
        x = self._validate(x)
        rng = np.random.default_rng(self.random_state)

        centers = _kmeans_pp_init(x, self.n_clusters, rng)
        labels = np.zeros(x.shape[0], dtype=np.int64)

        for it in range(1, self.max_iter + 1):
            labels = _assign(x, centers)
            new_centers = _update_centers(x, labels, self.n_clusters, centers, rng)
            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift <= self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = _inertia(x, labels, centers)
        self.n_iter_ = it
        return self

    def predict(self, x: FloatArray) -> IntArray:
        """Return the nearest cluster index for each sample in ``x``."""
        self._check_fitted()
        return _assign(self._validate(x), self.cluster_centers_)

    def fit_predict(self, x: FloatArray) -> IntArray:
        return self.fit(x).labels_

    # -------------------------------------------------------------- helpers

    def _check_fitted(self) -> None:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("KMeans instance is not fitted; call fit() first")

    @staticmethod
    def _validate(x: FloatArray) -> FloatArray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError("Input array must contain at least one sample")
        return arr


# --------------------------------------------------------------- internals


def _squared_distances(x: FloatArray, centers: FloatArray) -> FloatArray:
    """Return pairwise squared Euclidean distances ``(n_samples, n_clusters)``.

    Uses the algebraic identity ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b``,
    which is the standard vectorised form. The result is clipped at 0 because
    floating-point round-off can produce small negative values that would
    poison a subsequent ``sqrt``.
    """
    x_sq = np.einsum("ij,ij->i", x, x)[:, None]
    c_sq = np.einsum("ij,ij->i", centers, centers)[None, :]
    cross = x @ centers.T
    return np.maximum(x_sq + c_sq - 2.0 * cross, 0.0)


def _assign(x: FloatArray, centers: FloatArray) -> IntArray:
    return np.argmin(_squared_distances(x, centers), axis=1).astype(np.int64)


def _inertia(x: FloatArray, labels: IntArray, centers: FloatArray) -> float:
    diffs = x - centers[labels]
    return float(np.einsum("ij,ij->", diffs, diffs))


def _kmeans_pp_init(
    x: FloatArray,
    k: int,
    rng: np.random.Generator,
) -> FloatArray:
    """k-means++ centroid initialisation (Arthur & Vassilvitskii, 2007)."""
    n_samples = x.shape[0]
    if k > n_samples:
        raise ValueError("n_clusters cannot exceed number of samples")

    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    first = int(rng.integers(n_samples))
    centers[0] = x[first]

    closest_sq = _squared_distances(x, centers[:1]).ravel()
    for i in range(1, k):
        total = float(closest_sq.sum())
        if total == 0.0:
            # All points coincide with chosen centers; pick any remaining one.
            centers[i] = x[int(rng.integers(n_samples))]
        else:
            probs = closest_sq / total
            chosen = int(rng.choice(n_samples, p=probs))
            centers[i] = x[chosen]
        # Update running minimum of squared distances against new center only.
        new_dist = _squared_distances(x, centers[i : i + 1]).ravel()
        closest_sq = np.minimum(closest_sq, new_dist)
    return centers


def _update_centers(
    x: FloatArray,
    labels: IntArray,
    k: int,
    old_centers: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    """Compute new centroids; re-seed any cluster that lost all its points."""
    new_centers = np.empty_like(old_centers)
    counts = np.bincount(labels, minlength=k)
    sums = np.zeros_like(old_centers)
    np.add.at(sums, labels, x)

    for j in range(k):
        if counts[j] > 0:
            new_centers[j] = sums[j] / counts[j]
        else:
            # Re-seed empty cluster from a random point - standard recovery.
            new_centers[j] = x[int(rng.integers(x.shape[0]))]
    return new_centers

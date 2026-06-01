# Part 2a - Personal Programming Code Examples

Two small, self-contained algorithm modules. They are deliberately scoped:
the focus is on showing clean Python (typing, dataclasses, vectorised NumPy,
PEP 8, docstrings, unit tests) rather than maximum surface area.

## Files to review

| File | What it shows |
| ---- | ------------- |
| `algorithms/astar.py`  | A* shortest-path search on a 2D grid with eight-connectivity, octile heuristic, corner-cut prevention, and a heap-based open set. |
| `algorithms/kmeans.py` | K-Means clustering from scratch with k-means++ initialisation, vectorised distance computation, empty-cluster recovery, and `random_state` reproducibility. |

The accompanying tests live under `tests/`:

| File | What it covers |
| ---- | -------------- |
| `tests/test_astar.py`  | Path correctness, optimality vs. octile distance, corner-cut prevention, unreachable goals, input validation. |
| `tests/test_kmeans.py` | Cluster recovery on synthetic blobs, reproducibility, monotonically decreasing inertia in `k`, single-cluster centroid equals the mean, error paths. |

## Running the tests

From this directory:

```bash
python -m pytest -q
```

Expected: 17 tests pass.

## Design choices worth flagging

- **Pure functions where it makes sense.** `astar()` is a free function that
  takes an immutable `Grid`; `KMeans` is a small dataclass that keeps fitted
  state in `*_` attributes (the scikit-learn idiom).
- **No premature abstraction.** The grid is the simplest representation
  that allows corner-cut prevention; the K-Means class doesn't accept a
  pluggable distance metric because none of the tests need one.
- **Numerically careful K-Means.** Squared distances use the standard
  `||x||^2 + ||c||^2 - 2 x.c` form and are clipped at zero so that
  floating-point round-off cannot produce negative values that would later
  blow up under `sqrt`.
- **Determinism.** Both modules avoid hidden global state. `KMeans` takes
  `random_state` and seeds a local `np.random.default_rng`.

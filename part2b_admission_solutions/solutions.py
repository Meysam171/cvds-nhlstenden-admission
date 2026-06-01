"""Solutions to the CVDS NHL Stenden admission debugging exercises.

Each function below mirrors the signature of the buggy original supplied in
``debugging.ipynb``. The body has been corrected and short comments are placed
at the lines that contained the bug so a reviewer can audit the fix at a
glance.

The four exercises and their fixes:

1. :func:`id_to_fruit` - the original accepted a ``Set[str]`` and iterated it.
   Python's ``set`` has no defined order, so indexing it by insertion order is
   undefined behaviour. The fix is to take an ordered container (a list/tuple)
   or to sort the set deterministically before indexing.

2. :func:`swap` - two bugs. (a) The right-hand side of the assignment used
   ``coords[:, 1]`` twice instead of ``coords[:, 0]`` for the first slot.
   (b) NumPy slices are views, not copies, so the simultaneous tuple
   assignment that works for Python scalars overwrites values in place before
   the right-hand side is fully evaluated. The fix is to evaluate the desired
   columns into a fresh array (or call ``.copy()`` on the RHS).

3. :func:`plot_data` - ``csv.reader`` yields strings, so ``np.stack(results)``
   produces a string array and ``plt.plot`` plots them in their string order
   rather than numerically. The fix is to cast to ``float``. The axes were
   also swapped relative to the docstring; we put precision on the x-axis and
   recall on the y-axis as the docstring requires.

4. :func:`train_gan` - two bugs. (a) Structural: the last mini-batch of an
   epoch may be smaller than ``batch_size`` when ``len(dataset) % batch_size``
   is non-zero, so labels of shape ``(batch_size, 1)`` no longer line up with
   the discriminator output. The fix is to derive the label shape from
   ``real_samples.size(0)``. (b) Cosmetic: ``if n == batch_size - 1`` ties a
   logging condition to the *batch size* instead of to the *number of batches
   per epoch*, so changing ``batch_size`` silently changes how often progress
   is logged. The fix is to compare against ``len(train_loader) - 1`` (or any
   other deliberate logging cadence).
"""

from __future__ import annotations

import csv
from typing import Sequence

import numpy as np


# --------------------------------------------------------------- Exercise 1


def id_to_fruit(fruit_id: int, fruits: Sequence[str]) -> str:
    """Return the fruit at index ``fruit_id`` of an ordered fruit collection.

    The original signature used :class:`set`, which is unordered, so iterating
    it to find an index produced non-deterministic results. The fix is to use
    an ordered sequence (list/tuple). When a ``set`` is passed for backwards
    compatibility, we sort it to get deterministic indexing.
    """
    if isinstance(fruits, (set, frozenset)):
        # Determinism via lexicographic sort; documents the choice explicitly.
        fruits = sorted(fruits)
    if not 0 <= fruit_id < len(fruits):
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")
    return fruits[fruit_id]


# --------------------------------------------------------------- Exercise 2


def swap(coords: np.ndarray) -> np.ndarray:
    """Return ``coords`` with the (x1, y1) and (x2, y2) pairs flipped.

    Two bugs in the original implementation:

    * The RHS used ``coords[:, 1]`` twice and never read column 0, so x1 was
      lost. Correct RHS columns are ``(1, 0, 3, 2)``.
    * NumPy slices are views into the same buffer. Unlike scalar tuple
      assignment, by the time the second target is written the first
      assignment has already mutated the source. We therefore build a copy of
      the source columns *before* writing back.
    """
    out = coords.copy()
    out[:, 0] = coords[:, 1]
    out[:, 1] = coords[:, 0]
    out[:, 2] = coords[:, 3]
    out[:, 3] = coords[:, 2]
    return out


# --------------------------------------------------------------- Exercise 3


def plot_data(csv_file_path: str) -> None:
    """Plot a precision-recall curve from ``csv_file_path``.

    Two issues with the original:

    * ``csv.reader`` returns rows of strings, so ``np.stack(results)`` yields
      a string array. ``plt.plot`` then either crashes or interprets the
      values as categorical labels - either way it does not produce a sorted
      numeric line plot. The fix is to cast the array to ``float``.
    * The original swapped the precision and recall axes relative to the
      docstring. We follow the docstring: precision on x, recall on y.
    """
    # Imported lazily so the module is importable in environments where
    # matplotlib's backend cannot initialise (e.g. headless test runners).
    import matplotlib.pyplot as plt

    with open(csv_file_path, newline="") as fp:
        reader = csv.reader(fp, delimiter=",")
        next(reader)  # discard header
        rows = [row for row in reader]

    # Bug fix: cast strings to floats before stacking.
    data = np.array(rows, dtype=float)
    precision = data[:, 0]
    recall = data[:, 1]

    plt.plot(precision, recall)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()

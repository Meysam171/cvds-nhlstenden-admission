"""Unit tests that pin down the fixes for the admission debugging exercises."""

from __future__ import annotations

import csv
import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # headless backend for test runners

import numpy as np
import pytest

from solutions import id_to_fruit, plot_data, swap


# Exercise 1 ----------------------------------------------------------------


def test_id_to_fruit_with_list_is_order_preserving() -> None:
    fruits = ["apple", "orange", "melon", "kiwi", "strawberry"]
    assert id_to_fruit(1, fruits) == "orange"
    assert id_to_fruit(3, fruits) == "kiwi"
    assert id_to_fruit(4, fruits) == "strawberry"


def test_id_to_fruit_with_set_is_deterministic() -> None:
    fruits = {"apple", "orange", "melon", "kiwi", "strawberry"}
    # Sorted order: apple, kiwi, melon, orange, strawberry
    assert id_to_fruit(0, fruits) == "apple"
    assert id_to_fruit(4, fruits) == "strawberry"


def test_id_to_fruit_out_of_range_raises() -> None:
    with pytest.raises(RuntimeError):
        id_to_fruit(99, ["a", "b"])


# Exercise 2 ----------------------------------------------------------------


def test_swap_flips_x_and_y_pairs() -> None:
    coords = np.array(
        [
            [10, 5, 15, 6, 0],
            [11, 3, 13, 6, 0],
            [5, 3, 13, 6, 1],
            [4, 4, 13, 6, 1],
            [6, 5, 13, 16, 1],
        ]
    )
    expected = np.array(
        [
            [5, 10, 6, 15, 0],
            [3, 11, 6, 13, 0],
            [3, 5, 6, 13, 1],
            [4, 4, 6, 13, 1],
            [5, 6, 16, 13, 1],
        ]
    )
    np.testing.assert_array_equal(swap(coords), expected)


def test_swap_does_not_mutate_input() -> None:
    coords = np.array([[1, 2, 3, 4, 0]])
    original = coords.copy()
    swap(coords)
    np.testing.assert_array_equal(coords, original)


# Exercise 3 ----------------------------------------------------------------


def test_plot_data_runs_without_error_on_numeric_csv(tmp_path) -> None:
    path = tmp_path / "data_file.csv"
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["precision", "recall"])
        writer.writerows(
            [
                [0.013, 0.951],
                [0.376, 0.851],
                [0.441, 0.839],
                [0.570, 0.758],
                [0.635, 0.674],
                [0.721, 0.604],
                [0.837, 0.531],
                [0.860, 0.453],
                [0.962, 0.348],
                [0.982, 0.273],
                [1.0, 0.0],
            ]
        )
    plot_data(str(path))


def test_plot_data_rejects_non_numeric_payload(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["precision", "recall"])
        writer.writerow(["foo", "bar"])
    with pytest.raises(ValueError):
        plot_data(str(path))

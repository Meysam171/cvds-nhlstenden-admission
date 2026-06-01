"""Tests for the A* implementation."""

import math

import pytest

from algorithms.astar import Grid, astar, manhattan, octile, render_path


def test_trivial_path_same_cell() -> None:
    grid = Grid.from_rows([".."])
    assert astar(grid, (0, 0), (0, 0)) == [(0, 0)]


def test_straight_open_corridor() -> None:
    grid = Grid.from_rows(["....."])
    path = astar(grid, (0, 0), (0, 4))
    assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_must_go_around_wall() -> None:
    grid = Grid.from_rows(
        [
            "S....",
            "####.",
            ".....",
            ".####",
            "....G",
        ]
    )
    path = astar(grid, (0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (4, 4)
    # Path should not cross any wall cell.
    assert all(grid.cells[r][c] != "#" for r, c in path)


def test_no_path_when_completely_walled_off() -> None:
    grid = Grid.from_rows(
        [
            "S#.",
            "###",
            ".#G",
        ]
    )
    assert astar(grid, (0, 0), (2, 2)) is None


def test_rejects_unwalkable_endpoints() -> None:
    grid = Grid.from_rows(["#.."])
    assert astar(grid, (0, 0), (0, 2)) is None
    assert astar(grid, (0, 1), (0, 0)) is None


def test_diagonal_corner_cut_blocked() -> None:
    # Two diagonal walls meeting at a corner must NOT be cut by the path.
    grid = Grid.from_rows(
        [
            ".#",
            "#.",
        ]
    )
    assert astar(grid, (0, 0), (1, 1)) is None


def test_octile_heuristic_admissible_vs_manhattan() -> None:
    # On a fully open grid, optimal cost equals octile(start, goal).
    rows = ["." * 10 for _ in range(10)]
    grid = Grid.from_rows(rows)
    start, goal = (0, 0), (9, 9)

    path = astar(grid, start, goal, heuristic=octile)
    assert path is not None
    # Pure-diagonal optimal path has length n+1 on an n*n grid.
    assert len(path) == 10
    cost = sum(
        math.sqrt(2.0) if a[0] != b[0] and a[1] != b[1] else 1.0
        for a, b in zip(path, path[1:])
    )
    assert cost == pytest.approx(octile(start, goal))


def test_manhattan_heuristic_still_finds_path() -> None:
    grid = Grid.from_rows(["....", "....", "...."])
    assert astar(grid, (0, 0), (2, 3), heuristic=manhattan) is not None


def test_render_path_overlays_marker() -> None:
    grid = Grid.from_rows(["...", "...", "..."])
    path = astar(grid, (0, 0), (2, 2))
    rendered = render_path(grid, path or [], marker="*")
    assert rendered.count("*") == len(path or [])


def test_grid_validation_rejects_ragged_rows() -> None:
    with pytest.raises(ValueError):
        Grid.from_rows(["...", ".."])
    with pytest.raises(ValueError):
        Grid.from_rows([])

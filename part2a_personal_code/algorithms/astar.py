"""A* shortest-path search on a 2D grid.

The module provides a small, well-typed, PEP 8 compliant implementation of the
A* algorithm. The grid is represented as a sequence of rows of characters or
booleans, where any "truthy" cell except the wall character is walkable.

The implementation keeps the algorithm itself free of I/O and global state so
the same function can be reused for path planning on game maps, robot motion
planning, or simple computer-vision occupancy grids.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple


Point = Tuple[int, int]
Heuristic = Callable[[Point, Point], float]


# Eight-connectivity neighbourhood with pre-computed move costs. Diagonal moves
# cost sqrt(2); cardinal moves cost 1. Diagonals come last so that, for ties,
# cardinal moves are explored first - this yields nicer-looking paths.
_NEIGHBOURS: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def octile(a: Point, b: Point) -> float:
    """Octile distance - admissible heuristic for 8-connected grids."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)


def manhattan(a: Point, b: Point) -> float:
    """Manhattan distance - admissible heuristic for 4-connected grids."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


@dataclass(frozen=True)
class Grid:
    """Immutable rectangular grid where ``wall`` marks blocked cells."""

    cells: Tuple[str, ...]
    wall: str = "#"

    @classmethod
    def from_rows(cls, rows: Sequence[str], wall: str = "#") -> "Grid":
        """Build a :class:`Grid` from a sequence of equal-length rows."""
        if not rows:
            raise ValueError("Grid must contain at least one row")
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("All grid rows must have the same length")
        return cls(cells=tuple(rows), wall=wall)

    @property
    def height(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0])

    def in_bounds(self, p: Point) -> bool:
        r, c = p
        return 0 <= r < self.height and 0 <= c < self.width

    def walkable(self, p: Point) -> bool:
        r, c = p
        return self.in_bounds(p) and self.cells[r][c] != self.wall

    def neighbours(self, p: Point) -> Iterator[Tuple[Point, float]]:
        """Yield ``(neighbour, step_cost)`` for each walkable neighbour."""
        r, c = p
        for dr, dc, cost in _NEIGHBOURS:
            nb: Point = (r + dr, c + dc)
            if not self.walkable(nb):
                continue
            # Disallow diagonals that would "cut" a corner between two walls;
            # this is the standard fix that keeps A* paths physically sensible
            # for agents that cannot squeeze through diagonal gaps.
            if dr != 0 and dc != 0:
                if not self.walkable((r + dr, c)) or not self.walkable((r, c + dc)):
                    continue
            yield nb, cost


@dataclass(order=True)
class _Entry:
    """Priority-queue entry; ordering by ``(f, tie)`` only."""

    f: float
    tie: int
    point: Point = field(compare=False)


def astar(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic: Optional[Heuristic] = None,
) -> Optional[List[Point]]:
    """Return the shortest path from ``start`` to ``goal`` or ``None``.

    Args:
        grid: Walkable grid to search.
        start: Starting cell ``(row, col)``.
        goal: Target cell ``(row, col)``.
        heuristic: Admissible heuristic. Defaults to octile distance, which is
            admissible and consistent on 8-connected grids with unit/sqrt(2)
            costs and therefore guarantees optimal paths.

    Returns:
        A list of cells from ``start`` to ``goal`` inclusive, or ``None`` if no
        path exists.
    """
    if not grid.walkable(start) or not grid.walkable(goal):
        return None
    if start == goal:
        return [start]

    h: Heuristic = heuristic or octile

    came_from: dict[Point, Point] = {}
    g_score: dict[Point, float] = {start: 0.0}

    open_heap: List[_Entry] = []
    tie = 0
    heapq.heappush(open_heap, _Entry(h(start, goal), tie, start))
    # Closed set is implicit: a popped node is "closed" if its current g_score
    # is no longer reachable by a shorter path discovered later.
    closed: set[Point] = set()

    while open_heap:
        current = heapq.heappop(open_heap).point
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        closed.add(current)

        for nb, step in grid.neighbours(current):
            tentative = g_score[current] + step
            if tentative < g_score.get(nb, math.inf):
                g_score[nb] = tentative
                came_from[nb] = current
                tie += 1
                heapq.heappush(open_heap, _Entry(tentative + h(nb, goal), tie, nb))

    return None


def _reconstruct(came_from: dict[Point, Point], end: Point) -> List[Point]:
    path: List[Point] = [end]
    while end in came_from:
        end = came_from[end]
        path.append(end)
    path.reverse()
    return path


def render_path(grid: Grid, path: Iterable[Point], marker: str = "o") -> str:
    """Return a human-readable string of ``grid`` with ``path`` overlaid."""
    overlay = [list(row) for row in grid.cells]
    for r, c in path:
        if overlay[r][c] != grid.wall:
            overlay[r][c] = marker
    return "\n".join("".join(row) for row in overlay)

# thinning.py
"""
thinning.py

Implements two thinning strategies for point clouds that approximate reachability
sets:

1. Grid-based thinning: attach each point to a rectangular grid cell and keep
   at most one representative per cell. This approximates the "regular thinning"
   from the method notes and keeps the point spacing O(h).

2. Poisson Disk thinning: random greedy algorithm that enforces a minimum
   distance r between any two kept points, approximating Poisson Disk Sampling.
"""

from typing import Tuple, Dict

import numpy as np


def thin_grid(points: np.ndarray, h: float) -> np.ndarray:
    """
    Grid-based thinning.

    We overlay a regular grid with step h and keep at most one point per cell.
    This keeps the number of points manageable and approximates an almost
    uniform spacing of about h.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, 2).
    h : float
        Grid step size.

    Returns
    -------
    np.ndarray
        Thinned point cloud.
    """
    if points.size == 0:
        return points

    mins = points.min(axis=0)
    # Map each point to integer cell indices
    indices = np.floor((points - mins) / h).astype(int)

    cell_to_index: Dict[Tuple[int, int], int] = {}
    for idx_point, cell in enumerate(indices):
        key = (cell[0], cell[1])
        if key not in cell_to_index:
            cell_to_index[key] = idx_point
        else:
            # Optional: choose the point closer to the cell center
            prev_idx = cell_to_index[key]
            cell_center = mins + (np.array(key) + 0.5) * h
            d_prev = np.linalg.norm(points[prev_idx] - cell_center)
            d_new = np.linalg.norm(points[idx_point] - cell_center)
            if d_new < d_prev:
                cell_to_index[key] = idx_point

    kept_indices = np.fromiter(cell_to_index.values(), dtype=int)
    return points[kept_indices]


def thin_poisson(points: np.ndarray, r: float, random_state: int | None = None) -> np.ndarray:
    """
    Poisson Disk-like thinning via a simple greedy algorithm.

    Points are visited in random order. A point is accepted if it is at
    distance >= r from all previously accepted points. For moderate N this
    algorithm is sufficient and easy to implement.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N, 2).
    r : float
        Minimal allowed distance between any two kept points.
    random_state : int or None
        Seed for the RNG (for reproducibility).

    Returns
    -------
    np.ndarray
        Thinned point cloud.
    """
    if points.size == 0:
        return points

    rng = np.random.default_rng(random_state)
    order = rng.permutation(points.shape[0])
    kept = []

    r2 = r * r
    for idx in order:
        p = points[idx]
        accept = True
        for q in kept:
            if np.sum((p - q) ** 2) < r2:
                accept = False
                break
        if accept:
            kept.append(p)

    if not kept:
        # Fallback: keep at least one point
        kept.append(points[order[0]])

    return np.stack(kept, axis=0)

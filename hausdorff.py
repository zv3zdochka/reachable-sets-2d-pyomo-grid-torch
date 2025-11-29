# hausdorff.py
"""
hausdorff.py

Utilities to compute the (symmetric) Hausdorff distance between two finite
point clouds in R^2 and to identify one pair of points where this distance
is (approximately) attained.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree

    HAS_KDTREE = True
except ImportError:
    HAS_KDTREE = False


@dataclass
class HausdorffResult:
    """
    Container for Hausdorff distance computation results.
    """
    distance: float
    point_a: np.ndarray
    point_b: np.ndarray


def _directed_hausdorff(
        A: np.ndarray,
        B: np.ndarray,
) -> Tuple[float, int, int]:
    """
    Compute directed Hausdorff distance from A to B, and argmax indices.

    d(A,B) = max_{a in A} min_{b in B} ||a - b||_2.

    Returns
    -------
    d_max : float
        Directed distance from A to B.
    idx_a_max : int
        Index of point in A where max is attained.
    idx_b_near : int
        Index of closest point in B to that point.
    """
    if A.size == 0 or B.size == 0:
        return 0.0, -1, -1

    if HAS_KDTREE:
        tree = cKDTree(B)
        dists, idxs = tree.query(A)
        idx_a_max = int(np.argmax(dists))
        d_max = float(dists[idx_a_max])
        idx_b_near = int(idxs[idx_a_max])
        return d_max, idx_a_max, idx_b_near
    else:
        # Fallback: brute force via broadcasting
        diff = A[:, None, :] - B[None, :, :]  # (Na, Nb, 2)
        dists = np.linalg.norm(diff, axis=-1)  # (Na, Nb)
        min_dists = dists.min(axis=1)  # (Na,)
        idx_near = dists.argmin(axis=1)  # (Na,)
        idx_a_max = int(np.argmax(min_dists))
        d_max = float(min_dists[idx_a_max])
        idx_b_near = int(idx_near[idx_a_max])
        return d_max, idx_a_max, idx_b_near


def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> HausdorffResult:
    """
    Compute the symmetric Hausdorff distance between two point clouds.

        d_H(A,B) = max{ d(A,B), d(B,A) },

    where d is the directed Hausdorff distance.

    Parameters
    ----------
    A, B : np.ndarray
        Point clouds of shape (N_A, 2) and (N_B, 2).

    Returns
    -------
    HausdorffResult
        Struct containing the distance and the pair of points (one in A, one in B)
        where the maximum is attained.
    """
    d_ab, idx_a_ab, idx_b_ab = _directed_hausdorff(A, B)
    d_ba, idx_b_ba, idx_a_ba = _directed_hausdorff(B, A)

    if d_ab >= d_ba:
        distance = d_ab
        point_a = A[idx_a_ab]
        point_b = B[idx_b_ab]
    else:
        distance = d_ba
        point_a = A[idx_a_ba]
        point_b = B[idx_b_ba]

    return HausdorffResult(distance=distance, point_a=point_a, point_b=point_b)

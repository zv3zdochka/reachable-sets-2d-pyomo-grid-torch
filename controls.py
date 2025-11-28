# controls.py
"""
controls.py

Utility functions to generate discrete sets of admissible controls for
different shapes of the control set P (disk, box, ellipse). The primary
variant used in the project is a disk and its boundary (circle), sampled
uniformly in angle.
"""

import numpy as np


def generate_controls_disk(num_controls: int, u_max: float, on_circle: bool = True) -> np.ndarray:
    """
    Generate a finite set of controls for a disk P = {u : ||u|| <= u_max}.

    If on_circle=True, controls lie on the circle ||u|| = u_max (used for
    "perimeter sweep" heuristics). Otherwise, a small radial grid inside the
    disk is added.

    Parameters
    ----------
    num_controls : int
        Number of angular samples (points on the circle).
    u_max : float
        Radius of the control disk.
    on_circle : bool, default True
        If True, sample only on the circle. If False, use two radial levels.

    Returns
    -------
    np.ndarray
        Array of controls of shape (M, 2).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_controls, endpoint=False)
    controls = []
    if on_circle:
        r_values = [u_max]
    else:
        r_values = [0.5 * u_max, u_max]
    for r in r_values:
        controls.append(
            np.stack((r * np.cos(angles), r * np.sin(angles)), axis=-1)
        )
    return np.concatenate(controls, axis=0)


def generate_controls_box(num_per_dim: int, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """
    Generate controls for a box P = [u1_min, u1_max] x [u2_min, u2_max].

    Parameters
    ----------
    num_per_dim : int
        Number of grid points per dimension.
    bounds : tuple
        (u1_min, u1_max, u2_min, u2_max).

    Returns
    -------
    np.ndarray
        Array of controls of shape (M, 2).
    """
    u1_min, u1_max, u2_min, u2_max = bounds
    u1_grid = np.linspace(u1_min, u1_max, num_per_dim)
    u2_grid = np.linspace(u2_min, u2_max, num_per_dim)
    U1, U2 = np.meshgrid(u1_grid, u2_grid, indexing="xy")
    return np.stack((U1.ravel(), U2.ravel()), axis=-1)


def generate_controls_ellipse(num_controls: int, a: float, b: float) -> np.ndarray:
    """
    Generate controls on the boundary of an ellipse:

        P = { (u1,u2) : (u1/a)^2 + (u2/b)^2 <= 1 }.

    The controls lie on the curve (u1/a)^2 + (u2/b)^2 = 1 sampled uniformly
    in angle.

    Parameters
    ----------
    num_controls : int
        Number of angular samples.
    a : float
        Semi-axis along u1.
    b : float
        Semi-axis along u2.

    Returns
    -------
    np.ndarray
        Array of controls of shape (num_controls, 2).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_controls, endpoint=False)
    u1 = a * np.cos(angles)
    u2 = b * np.sin(angles)
    return np.stack((u1, u2), axis=-1)

# backend_numpy.py
"""
backend_numpy.py

NumPy-based backend for propagating a point cloud of states one time step
forward under all discrete controls. This is the baseline (non-accelerated)
implementation of the reachability update.
"""

import numpy as np

from system import ControlledSystem


def propagate_numpy(
    system: ControlledSystem,
    states: np.ndarray,
    controls: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Propagate the set of states one time step forward for all controls.

    Parameters
    ----------
    system : ControlledSystem
        The controlled system (provides f_numpy).
    states : np.ndarray
        Current point cloud of states, shape (N, 2).
    controls : np.ndarray
        Discrete set of controls, shape (M, 2).
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        New (unthinned) point cloud of shape (N * M, 2).
    """
    if states.size == 0:
        return states

    # Broadcast states and controls: (N, 1, 2) and (1, M, 2) -> (N, M, 2)
    X = states[:, np.newaxis, :]          # (N, 1, 2)
    U = controls[np.newaxis, :, :]        # (1, M, 2)
    dX = system.f_numpy(X, U)             # (N, M, 2)
    X_new = X + dt * dX                   # (N, M, 2)
    return X_new.reshape(-1, 2)

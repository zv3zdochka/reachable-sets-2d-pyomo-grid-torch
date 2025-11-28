# backend_torch.py
"""
backend_torch.py

PyTorch-based backend for propagating a point cloud of states one time step
forward under all discrete controls. This is the accelerated implementation,
which can use CPU or GPU depending on the selected device.
"""

from typing import Literal

import numpy as np

try:
    import torch
except ImportError as e:  # pragma: no cover - handled at runtime
    raise ImportError(
        "backend_torch requires PyTorch to be installed. "
        "Install it via `pip install torch`."
    ) from e

from system import ControlledSystemTorch


def propagate_torch(
    system: ControlledSystemTorch,
    states: np.ndarray,
    controls: np.ndarray,
    dt: float,
    device: Literal["cpu", "cuda"] = "cpu",
) -> np.ndarray:
    """
    Propagate the set of states one time step forward for all controls using Torch.

    Parameters
    ----------
    system : ControlledSystemTorch
        The controlled system (provides f_torch).
    states : np.ndarray
        Current point cloud of states, shape (N, 2), NumPy array.
    controls : np.ndarray
        Discrete set of controls, shape (M, 2), NumPy array.
    dt : float
        Time step.
    device : {"cpu", "cuda"}
        Torch device to use.

    Returns
    -------
    np.ndarray
        New (unthinned) point cloud of shape (N * M, 2), as NumPy array on CPU.
    """
    if states.size == 0:
        return states

    x = torch.from_numpy(states).to(device=device, dtype=torch.float32)  # (N, 2)
    u = torch.from_numpy(controls).to(device=device, dtype=torch.float32)  # (M, 2)

    X = x[:, None, :]      # (N, 1, 2)
    U = u[None, :, :]      # (1, M, 2)
    dX = system.f_torch(X, U)  # (N, M, 2)
    X_new = X + dt * dX    # (N, M, 2)

    X_new_flat = X_new.reshape(-1, 2).detach().cpu().numpy()
    return X_new_flat

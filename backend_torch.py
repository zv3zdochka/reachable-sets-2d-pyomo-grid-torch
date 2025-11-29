# backend_torch.py
"""
backend_torch.py

PyTorch-based utilities for the reachability method.

Here we provide:
- propagate_torch_numpy: backward-compatible wrapper that takes NumPy arrays
  and returns NumPy arrays (used only if needed).
- propagate_torch_tensor: core propagation step that works purely on torch.Tensor
  and is intended for the accelerated GPU backend.
- thin_grid_torch: grid-based thinning implemented in pure Torch, so that both
  the propagation and thinning can stay on the GPU.
"""

from typing import Literal

import numpy as np

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "backend_torch requires PyTorch to be installed. "
        "Install it via `pip install torch`."
    ) from e

from system import ControlledSystemTorch


# === Старый NumPy-совместимый вариант (оставляем про запас) ===

def propagate_torch_numpy(
        system: ControlledSystemTorch,
        states: np.ndarray,
        controls: np.ndarray,
        dt: float,
        device: Literal["cpu", "cuda"] = "cpu",
) -> np.ndarray:
    """
    Backward-compatible wrapper: NumPy -> Torch -> NumPy.

    This is kept for completeness, but the accelerated backend should use
    propagate_torch_tensor + thin_grid_torch directly (no CPU<->GPU conversions
    on each step).
    """
    if states.size == 0:
        return states

    x = torch.from_numpy(states).to(device=device, dtype=torch.float32)  # (N, 2)
    u = torch.from_numpy(controls).to(device=device, dtype=torch.float32)  # (M, 2)

    X = x[:, None, :]  # (N, 1, 2)
    U = u[None, :, :]  # (1, M, 2)
    dX = system.f_torch(X, U)  # (N, M, 2)
    X_new = X + dt * dX  # (N, M, 2)

    X_new_flat = X_new.reshape(-1, 2).detach().cpu().numpy()
    return X_new_flat


# === Новый тензорный вариант для быстрого backend'а ===

def propagate_torch_tensor(
        system: ControlledSystemTorch,
        states_t: "torch.Tensor",
        controls_t: "torch.Tensor",
        dt: float,
) -> "torch.Tensor":
    """
    Core propagation step on torch.Tensor.

    Parameters
    ----------
    system : ControlledSystemTorch
        System with f_torch().
    states_t : torch.Tensor
        Current point cloud, shape (N, 2), on the desired device.
    controls_t : torch.Tensor
        Discrete controls, shape (M, 2), on the same device.
    dt : float
        Time step.

    Returns
    -------
    torch.Tensor
        New (unthinned) point cloud of shape (N * M, 2) on the same device.
    """
    if states_t.numel() == 0:
        return states_t

    X = states_t[:, None, :]  # (N, 1, 2)
    U = controls_t[None, :, :]  # (1, M, 2)
    dX = system.f_torch(X, U)  # (N, M, 2)
    X_new = X + dt * dX  # (N, M, 2)
    return X_new.reshape(-1, 2)  # (N*M, 2)


def thin_grid_torch(
        points_t: "torch.Tensor",
        h: float,
) -> "torch.Tensor":
    """
    Grid-based thinning in pure Torch.

    We overlay a regular grid with step h over the current cloud of points and
    keep at most one point per grid cell. The representative is chosen as the
    first point for each cell (in the order of the input tensor).

    Parameters
    ----------
    points_t : torch.Tensor
        Input point cloud, shape (N, 2), on some device.
    h : float
        Grid step.

    Returns
    -------
    torch.Tensor
        Thinned point cloud on the same device.
    """
    if points_t.numel() == 0:
        return points_t

    # mins: (2,)
    mins = torch.min(points_t, dim=0)[0]
    # integer cell indices: (N, 2)
    cell_idx = torch.floor((points_t - mins) / h).to(torch.int64)

    # unique over rows; use return_inverse instead of return_index (лучше совместимость)
    # inverse[i] = index of the unique cell for point i
    unique_cells, inverse = torch.unique(cell_idx, dim=0, return_inverse=True)

    # Сортируем точки по cell-id
    sorted_inv, order = torch.sort(inverse)  # sorted_inv: (N,), order: (N,)
    # Помечаем первые точки каждого нового cell-id
    new_cell = torch.ones_like(sorted_inv, dtype=torch.bool)
    new_cell[1:] = sorted_inv[1:] != sorted_inv[:-1]
    # Индексы исходных точек, по одной на каждую ячейку
    first_indices = order[new_cell]

    return points_t[first_indices]

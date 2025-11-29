# grid_reachability.py
"""
grid_reachability.py

High-level routines for constructing the reachable set of the 2D controlled
system via a grid/point-cloud method with thinning. Supports both NumPy and
Torch backends for the expensive propagation + thinning steps.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from system import ControlledSystem, ControlledSystemTorch
from thinning import thin_grid, thin_poisson
from backend_numpy import propagate_numpy

try:
    import torch
    from backend_torch import (
        propagate_torch_tensor,
        thin_grid_torch,
    )

    HAS_TORCH_BACKEND = True
except ImportError:
    HAS_TORCH_BACKEND = False

ThinningMethod = Literal["grid", "poisson"]
BackendType = Literal["numpy", "torch"]


@dataclass
class ReachabilityConfig:
    """
    Configuration for grid-based reachability computation.
    """
    T: float
    num_time_steps: int
    thinning_method: ThinningMethod = "grid"
    thinning_param: float = 0.1  # h for grid, r for Poisson
    backend: BackendType = "numpy"
    torch_device: Literal["cpu", "cuda"] = "cpu"


def _reachable_set_numpy(
        system: ControlledSystem,
        x0: np.ndarray,
        controls: np.ndarray,
        cfg: ReachabilityConfig,
) -> np.ndarray:
    """
    Internal helper: pure NumPy implementation (baseline).
    """
    dt = cfg.T / cfg.num_time_steps
    points = x0.reshape(1, 2)

    for _ in range(cfg.num_time_steps):
        points_new = propagate_numpy(system, points, controls, dt)

        if cfg.thinning_method == "grid":
            points = thin_grid(points_new, h=cfg.thinning_param)
        elif cfg.thinning_method == "poisson":
            points = thin_poisson(points_new, r=cfg.thinning_param)
        else:
            raise ValueError(f"Unknown thinning method: {cfg.thinning_method}")

    return points


def _reachable_set_torch(
        system: ControlledSystem,
        x0: np.ndarray,
        controls: np.ndarray,
        cfg: ReachabilityConfig,
) -> np.ndarray:
    """
    Internal helper: accelerated implementation on Torch.

    The entire point cloud is kept as a torch.Tensor on the selected device.
    Both the propagation step and the grid-based thinning are done on Torch.
    Poisson thinning (if requested) falls back to NumPy (slower), but the
    typical accelerated use case is with 'grid' thinning.
    """
    if not HAS_TORCH_BACKEND:
        raise RuntimeError("Torch backend requested but backend_torch is not available.")

    if not isinstance(system, ControlledSystemTorch):
        system_torch = ControlledSystemTorch(u_max=system.u_max)
    else:
        system_torch = system

    device = cfg.torch_device
    dt = cfg.T / cfg.num_time_steps

    # Move initial state and controls to device
    points_t = torch.tensor(x0, dtype=torch.float32, device=device).reshape(1, 2)
    controls_t = torch.tensor(controls, dtype=torch.float32, device=device)

    for _ in range(cfg.num_time_steps):
        # Propagation on device
        points_new_t = propagate_torch_tensor(system_torch, points_t, controls_t, dt)

        # Thinning on device (grid) or fallback (Poisson)
        if cfg.thinning_method == "grid":
            points_t = thin_grid_torch(points_new_t, h=cfg.thinning_param)
        elif cfg.thinning_method == "poisson":
            # fallback: move to CPU, apply NumPy Poisson thinning, move back
            points_np = points_new_t.detach().cpu().numpy()
            points_thin_np = thin_poisson(points_np, r=cfg.thinning_param)
            points_t = torch.tensor(points_thin_np, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unknown thinning method: {cfg.thinning_method}")

    # Return result to CPU as NumPy array
    return points_t.detach().cpu().numpy()


def compute_reachable_set_grid(
        system: ControlledSystem,
        x0: np.ndarray,
        controls: np.ndarray,
        cfg: ReachabilityConfig,
) -> np.ndarray:
    """
    Compute the reachable set at time T via the grid/point-cloud method.

    Parameters
    ----------
    system : ControlledSystem
        The controlled system.
    x0 : np.ndarray
        Initial state, shape (2,).
    controls : np.ndarray
        Discrete set of controls, shape (M, 2).
    cfg : ReachabilityConfig
        Configuration parameters.

    Returns
    -------
    np.ndarray
        Approximate reachable set at time T as a thinned point cloud, shape (N, 2).
    """
    if cfg.backend == "numpy":
        return _reachable_set_numpy(system, x0, controls, cfg)
    elif cfg.backend == "torch":
        return _reachable_set_torch(system, x0, controls, cfg)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

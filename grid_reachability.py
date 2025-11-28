# grid_reachability.py
"""
grid_reachability.py

High-level routines for constructing the reachable set of the 2D controlled
system via a grid/point-cloud method with thinning. Supports both NumPy and
Torch backends for the expensive propagation step.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from system import ControlledSystem, ControlledSystemTorch
from thinning import thin_grid, thin_poisson
from backend_numpy import propagate_numpy

try:
    from backend_torch import propagate_torch
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
    thinning_param: float = 0.1   # h for grid, r for Poisson
    backend: BackendType = "numpy"
    torch_device: Literal["cpu", "cuda"] = "cpu"


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
    dt = cfg.T / cfg.num_time_steps
    points = x0.reshape(1, 2)

    for _ in range(cfg.num_time_steps):
        if cfg.backend == "numpy":
            points_new = propagate_numpy(system, points, controls, dt)
        elif cfg.backend == "torch":
            if not HAS_TORCH_BACKEND:
                raise RuntimeError("Torch backend requested but backend_torch is not available.")
            if not isinstance(system, ControlledSystemTorch):
                # Wrap base ControlledSystem into the Torch subclass for f_torch
                system_torch = ControlledSystemTorch(u_max=system.u_max)
            else:
                system_torch = system
            points_new = propagate_torch(system_torch, points, controls, dt, device=cfg.torch_device)
        else:
            raise ValueError(f"Unknown backend: {cfg.backend}")

        # Thinning
        if cfg.thinning_method == "grid":
            points = thin_grid(points_new, h=cfg.thinning_param)
        elif cfg.thinning_method == "poisson":
            points = thin_poisson(points_new, r=cfg.thinning_param)
        else:
            raise ValueError(f"Unknown thinning method: {cfg.thinning_method}")

    return points

# system.py
"""
system.py

Defines the controlled dynamical system used in the reachability experiments.
The state is 2D, the control is 2D. Dynamics:

    x1' = x2 + u1
    x2' = -x1 + u2

The control is constrained to lie in a disk of radius u_max.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ControlledSystem:
    """
    Simple 2D controlled system with additive 2D control.

    Dynamics:
        dx1/dt = x2 + u1
        dx2/dt = -x1 + u2
    """
    u_max: float = 1.0

    def f_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the ODE in NumPy.

        Parameters
        ----------
        x : np.ndarray
            State array of shape (..., 2).
        u : np.ndarray
            Control array of shape (..., 2), broadcast-compatible with x.

        Returns
        -------
        np.ndarray
            Time derivative dx/dt of shape (..., 2).
        """
        dx1 = x[..., 1] + u[..., 0]
        dx2 = -x[..., 0] + u[..., 1]
        return np.stack((dx1, dx2), axis=-1)


try:
    import torch

    class ControlledSystemTorch(ControlledSystem):
        """
        Torch variant of the system. Shares the same dynamics, but implemented
        for torch.Tensor inputs.
        """

        def f_torch(self, x: "torch.Tensor", u: "torch.Tensor") -> "torch.Tensor":
            """
            Right-hand side of the ODE in PyTorch.

            Parameters
            ----------
            x : torch.Tensor
                State tensor of shape (..., 2).
            u : torch.Tensor
                Control tensor of shape (..., 2).

            Returns
            -------
            torch.Tensor
                Time derivative dx/dt of shape (..., 2).
            """
            dx1 = x[..., 1] + u[..., 0]
            dx2 = -x[..., 0] + u[..., 1]
            return torch.stack((dx1, dx2), dim=-1)

except ImportError:
    # Torch is optional; the rest of the project can run without it.
    ControlledSystemTorch = ControlledSystem  # type: ignore

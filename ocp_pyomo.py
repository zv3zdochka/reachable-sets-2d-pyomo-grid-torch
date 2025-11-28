# ocp_pyomo.py
"""
ocp_pyomo.py

Optimal control formulation (discrete-time via explicit Euler) using Pyomo to
compute boundary points of the reachable set. For each direction l(phi) we
solve a maximization problem:

    max <l(phi), x_T>

subject to system dynamics and control constraints.

Also provides a simpler brute-force variant that uses constant controls
selected from a set of candidates on a circle/ellipse.
"""

from typing import List

import numpy as np

try:
    import pyomo.environ as pyo
except ImportError as e:  # pragma: no cover - handled at runtime
    raise ImportError(
        "ocp_pyomo requires Pyomo to be installed. Install via `pip install pyomo`."
    ) from e

from system import ControlledSystem


def solve_ocp_direction(
    phi: float,
    system: ControlledSystem,
    x0: np.ndarray,
    T: float,
    num_time_steps: int,
    solver_name: str = "ipopt",
) -> np.ndarray:
    """
    Solve the optimal control problem for a single direction l(phi).

    Parameters
    ----------
    phi : float
        Direction angle in radians.
    system : ControlledSystem
        The controlled system (used only for u_max and consistency).
    x0 : np.ndarray
        Initial state, shape (2,).
    T : float
        Final time.
    num_time_steps : int
        Number of Euler steps (N).
    solver_name : str
        Name of Pyomo solver to use (e.g., "ipopt").

    Returns
    -------
    np.ndarray
        Boundary point x(T) as a NumPy array of shape (2,).
    """
    dt = T / num_time_steps
    N = num_time_steps

    model = pyo.ConcreteModel()
    model.K = pyo.RangeSet(0, N)          # time steps for state
    model.Kc = pyo.RangeSet(0, N - 1)     # time steps for control

    # State variables
    model.x1 = pyo.Var(model.K, domain=pyo.Reals)
    model.x2 = pyo.Var(model.K, domain=pyo.Reals)

    # Control variables
    model.u1 = pyo.Var(model.Kc, domain=pyo.Reals)
    model.u2 = pyo.Var(model.Kc, domain=pyo.Reals)

    # Initial condition
    model.x1[0].fix(float(x0[0]))
    model.x2[0].fix(float(x0[1]))

    # Dynamics constraints (explicit Euler)
    def dyn1_rule(m, k):
        return m.x1[k + 1] == m.x1[k] + dt * (m.x2[k] + m.u1[k])

    def dyn2_rule(m, k):
        return m.x2[k + 1] == m.x2[k] + dt * (-m.x1[k] + m.u2[k])

    model.dyn1 = pyo.Constraint(model.Kc, rule=dyn1_rule)
    model.dyn2 = pyo.Constraint(model.Kc, rule=dyn2_rule)

    # Control constraints: disk ||u|| <= u_max
    u_max = float(system.u_max)

    def control_bound_rule(m, k):
        return m.u1[k] ** 2 + m.u2[k] ** 2 <= u_max ** 2

    model.u_bound = pyo.Constraint(model.Kc, rule=control_bound_rule)

    # Objective: maximize <l(phi), x_N> = l1 * x1_N + l2 * x2_N
    l1 = float(np.cos(phi))
    l2 = float(np.sin(phi))

    def obj_rule(m):
        return -(l1 * m.x1[N] + l2 * m.x2[N])  # minimize negative

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(model, tee=False)

    # Extract terminal state
    xT = np.array([pyo.value(model.x1[N]), pyo.value(model.x2[N])], dtype=float)
    return xT


def compute_oc_boundary(
    system: ControlledSystem,
    x0: np.ndarray,
    T: float,
    num_time_steps: int,
    num_directions: int,
    solver_name: str = "ipopt",
) -> np.ndarray:
    """
    Compute the reachable set boundary by solving OCPs in multiple directions.

    Parameters
    ----------
    system : ControlledSystem
        The controlled system.
    x0 : np.ndarray
        Initial state, shape (2,).
    T : float
        Final time.
    num_time_steps : int
        Number of Euler steps in the dynamic constraints.
    num_directions : int
        Number of directions l(phi) to consider (phi in [0, 2pi)).
    solver_name : str
        Name of the Pyomo solver.

    Returns
    -------
    np.ndarray
        Boundary point cloud of shape (num_directions, 2), ordered by phi.
    """
    phis = np.linspace(0.0, 2.0 * np.pi, num_directions, endpoint=False)
    boundary_points: List[np.ndarray] = []
    for phi in phis:
        xT = solve_ocp_direction(phi, system, x0, T, num_time_steps, solver_name=solver_name)
        boundary_points.append(xT)
    return np.stack(boundary_points, axis=0)


def compute_oc_boundary_bruteforce(
    system: ControlledSystem,
    x0: np.ndarray,
    T: float,
    num_time_steps: int,
    phis: np.ndarray,
    control_candidates: np.ndarray,
) -> np.ndarray:
    """
    Auxiliary fast method: for each direction phi, consider only constant
    controls u(t) = u over [0, T], where u is drawn from a finite set of
    control_candidates (e.g., points on the circle or ellipse). For each phi,
    pick the candidate with maximal <l(phi), x_T(u)>.

    Parameters
    ----------
    system : ControlledSystem
        The controlled system.
    x0 : np.ndarray
        Initial state, shape (2,).
    T : float
        Final time.
    num_time_steps : int
        Number of Euler steps.
    phis : np.ndarray
        Array of direction angles of shape (K,).
    control_candidates : np.ndarray
        Array of candidate controls of shape (M, 2).

    Returns
    -------
    np.ndarray
        Approximate boundary point cloud of shape (K, 2).
    """
    dt = T / num_time_steps
    K = phis.shape[0]
    boundary_points = np.zeros((K, 2), dtype=float)

    for i, phi in enumerate(phis):
        l = np.array([np.cos(phi), np.sin(phi)], dtype=float)
        best_val = -np.inf
        best_xT = None

        for u in control_candidates:
            x = x0.copy()
            for _ in range(num_time_steps):
                dx = system.f_numpy(x, u)
                x = x + dt * dx
            val = float(np.dot(l, x))
            if val > best_val:
                best_val = val
                best_xT = x

        boundary_points[i] = best_xT

    return boundary_points

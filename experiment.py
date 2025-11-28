# experiment.py
"""
experiment.py

Example driver code that uses all modules to:

1. Build the reachable set via the grid method (NumPy backend) with grid-based thinning.
2. Optionally, build the reachable set via the grid method with Poisson Disk thinning.
3. Build the reachable set via the grid method (Torch backend) and measure acceleration.
4. Build the boundary of the reachable set via Pyomo optimal control.
5. Plot both reachable sets on one figure.
6. Compute and visualize the Hausdorff distance between the two sets.

This script is intended to be run in a Jupyter/Colab environment. Adjust
parameters (T, num_time_steps, etc.) if needed.
"""

import time

import numpy as np

from system import ControlledSystem, ControlledSystemTorch
from controls import generate_controls_disk
from grid_reachability import ReachabilityConfig, compute_reachable_set_grid
from ocp_pyomo import compute_oc_boundary
from hausdorff import hausdorff_distance
from plotting import plot_reachable_sets


def run_experiment():
    # 1. Problem setup
    x0 = np.array([0.0, 0.0], dtype=float)
    T = 1.0
    num_time_steps = 40
    u_max = 1.0

    system = ControlledSystem(u_max=u_max)

    # Discrete set of controls on the boundary of the control disk
    num_controls = 16
    controls = generate_controls_disk(num_controls=num_controls, u_max=u_max, on_circle=True)

    # Thinning parameters
    thinning_h = 0.05      # grid step for thinning
    thinning_r = 0.05      # Poisson radius for thinning

    # 2. Reachable set via grid method (NumPy backend, grid-based thinning)
    cfg_numpy_grid = ReachabilityConfig(
        T=T,
        num_time_steps=num_time_steps,
        thinning_method="grid",
        thinning_param=thinning_h,
        backend="numpy",
    )

    t0 = time.perf_counter()
    R_grid_numpy = compute_reachable_set_grid(system, x0, controls, cfg_numpy_grid)
    t1 = time.perf_counter()
    time_numpy = t1 - t0
    print(f"[NumPy backend, grid thinning] Reachable set size: {R_grid_numpy.shape[0]}, time = {time_numpy:.3f} s")

    # 2a. Reachable set via grid method (NumPy backend, Poisson Disk thinning) — optional
    cfg_numpy_poisson = ReachabilityConfig(
        T=T,
        num_time_steps=num_time_steps,
        thinning_method="poisson",
        thinning_param=thinning_r,
        backend="numpy",
    )

    t0 = time.perf_counter()
    R_grid_poisson = compute_reachable_set_grid(system, x0, controls, cfg_numpy_poisson)
    t1 = time.perf_counter()
    time_numpy_poisson = t1 - t0
    print(f"[NumPy backend, Poisson thinning] Reachable set size: {R_grid_poisson.shape[0]}, time = {time_numpy_poisson:.3f} s")

    # 3. Reachable set via grid method (Torch backend, CPU or CUDA)
    R_grid = R_grid_numpy
    try:
        system_torch = ControlledSystemTorch(u_max=u_max)
        cfg_torch = ReachabilityConfig(
            T=T,
            num_time_steps=num_time_steps,
            thinning_method="grid",
            thinning_param=thinning_h,
            backend="torch",
            torch_device="cuda" if False else "cpu",  # set to "cuda" if GPU is available
        )

        t0 = time.perf_counter()
        R_grid_torch = compute_reachable_set_grid(system_torch, x0, controls, cfg_torch)
        t1 = time.perf_counter()
        time_torch = t1 - t0
        print(f"[Torch backend, grid thinning] Reachable set size: {R_grid_torch.shape[0]}, time = {time_torch:.3f} s")

        if time_torch > 0:
            speedup = time_numpy / time_torch
            print(f"Measured speedup (NumPy grid / Torch grid) = {speedup:.2f}x")
        else:
            print("Torch time is too small to compute speedup reliably.")

        # Use the NumPy grid-thinned result as canonical for comparison.
        R_grid = R_grid_numpy

    except Exception as e:
        print(f"Torch backend not available or failed ({e}); using NumPy result only.")
        R_grid = R_grid_numpy

    # 4. Boundary via optimal control (Pyomo)
    num_directions = 32
    print("Solving optimal control problems for boundary (this may take some time)...")
    R_oc = compute_oc_boundary(
        system=system,
        x0=x0,
        T=T,
        num_time_steps=num_time_steps,
        num_directions=num_directions,
        solver_name="ipopt",   # make sure IPOPT is installed, or change to an available solver
    )
    print(f"OC boundary computed with {R_oc.shape[0]} points.")

    # 5. Hausdorff distance (between grid-based reachable set and OC boundary)
    hd = hausdorff_distance(R_grid, R_oc)
    print(f"Hausdorff distance between grid (grid thinning) and OC sets: d_H = {hd.distance:.4f}")

    # 6. Plot
    plot_reachable_sets(
        R_grid=R_grid,
        R_oc=R_oc,
        hd=hd,
        title="Reachable set (grid method) vs OC boundary",
        save_path=None,  # or specify a filename
    )


if __name__ == "__main__":
    run_experiment()

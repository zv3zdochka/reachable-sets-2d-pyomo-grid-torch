# experiment.py
"""
experiment.py

This module contains two main experiments:

1) Linear system (harmonic oscillator with additive control, disk of controls):
   - grid-based reachable set (NumPy backend),
   - Poisson disk thinning,
   - accelerated reachable set (Torch backend, grid thinning) and timing,
   - boundary via optimal control in Pyomo,
   - boundary via a brute-force sweep of constant controls,
   - Hausdorff distance between the grid reachable set and the OC boundary.

   For this linear system with convex control set the reachable set is
   theoretically convex for any T.

2) Li–Markus example (nonlinear in control, elliptic control set):
   - grid-based reachable set for several values of T,
   - boundary via brute-force constant controls on the elliptical control set,
   - convex hull of the reachable set (scipy.spatial.ConvexHull),
   - Hausdorff distance between the reachable set and its convex hull,
   - qualitative conclusion ("numerically non-convex" vs "approximately convex").

run_experiment() runs both experiments in sequence.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from system import ControlledSystem, ControlledSystemTorch
from controls import generate_controls_disk
from grid_reachability import ReachabilityConfig, compute_reachable_set_grid
from ocp_pyomo import compute_oc_boundary, compute_oc_boundary_bruteforce
from hausdorff import hausdorff_distance
from plotting import plot_reachable_sets


# ---------------------------------------------------------------------
# 1. Linear system experiment (current model)
# ---------------------------------------------------------------------


def run_linear_example():
    """
    Linear system:
        x1' = x2 + u1
        x2' = -x1 + u2
        ||u||_2 <= u_max

    For this system with a convex control set, the reachable set is
    theoretically convex for any T. Here we:
      - build the reachable set via the grid method (NumPy backend),
      - apply Poisson thinning (optional),
      - run an accelerated version on Torch (GPU) and measure speedup,
      - build the boundary via Pyomo optimal control,
      - build a second boundary via brute-force constant controls,
      - compute the Hausdorff distances and visualise the results.
    """
    print("=" * 80)
    print("Linear system: harmonic oscillator with additive control (disk control set)")
    print("Reachable set should be convex for any T.")
    print("=" * 80)

    # Problem setup
    x0 = np.array([0.0, 0.0], dtype=float)
    T = 1.0
    num_time_steps = 40
    u_max = 1.0

    system = ControlledSystem(u_max=u_max)

    # Discrete set of controls on the boundary of the control disk
    num_controls = 16
    controls = generate_controls_disk(num_controls=num_controls, u_max=u_max, on_circle=True)

    # Thinning parameters
    thinning_h = 0.05  # grid step for thinning
    thinning_r = 0.05  # Poisson radius for thinning

    # --- 1) Reachable set via grid method (NumPy backend, grid-based thinning)
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
    print(f"[Linear / NumPy backend, grid thinning] "
          f"Reachable set size: {R_grid_numpy.shape[0]}, time = {time_numpy:.3f} s")

    # --- 2) Reachable set via grid method (NumPy backend, Poisson Disk thinning) — optional
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
    print(f"[Linear / NumPy backend, Poisson thinning] "
          f"Reachable set size: {R_grid_poisson.shape[0]}, time = {time_numpy_poisson:.3f} s")

    # --- 3) Reachable set via grid method (Torch backend, CPU or CUDA)
    R_grid_for_OC = R_grid_numpy  # we will compare OC boundary with the NumPy grid
    try:
        system_torch = ControlledSystemTorch(u_max=u_max)
        cfg_torch = ReachabilityConfig(
            T=T,
            num_time_steps=num_time_steps,
            thinning_method="grid",
            thinning_param=thinning_h,
            backend="torch",
            torch_device="cuda",  # set to "cpu" if GPU is not available
        )

        t0 = time.perf_counter()
        R_grid_torch = compute_reachable_set_grid(system_torch, x0, controls, cfg_torch)
        t1 = time.perf_counter()
        time_torch = t1 - t0
        print(f"[Linear / Torch backend, grid thinning] "
              f"Reachable set size: {R_grid_torch.shape[0]}, time = {time_torch:.3f} s")

        if time_torch > 0:
            speedup = time_numpy / time_torch
            print(f"[Linear] Measured speedup (NumPy grid / Torch grid) = {speedup:.2f}x")
        else:
            print("[Linear] Torch time is too small to compute speedup reliably.")

    except Exception as e:
        print(f"[Linear] Torch backend not available or failed ({e}); "
              f"using NumPy result only.")

    # --- 4) Boundary via optimal control (Pyomo)
    num_directions = 32
    print("Solving optimal control problems for boundary (Pyomo) in the linear system...")
    R_oc_pyomo = compute_oc_boundary(
        system=system,
        x0=x0,
        T=T,
        num_time_steps=num_time_steps,
        num_directions=num_directions,
        solver_name="ipopt",  # make sure IPOPT is installed
    )
    print(f"[Linear] OC boundary (Pyomo) computed with {R_oc_pyomo.shape[0]} points.")

    # --- 5) Boundary via brute-force sweep over constant controls (disk)
    print("Computing OC boundary by brute-force over constant controls on the disk (linear system)...")

    phis = np.linspace(0.0, 2.0 * np.pi, num_directions, endpoint=False)
    control_candidates = generate_controls_disk(
        num_controls=32,
        u_max=u_max,
        on_circle=True,
    )

    R_oc_bruteforce = compute_oc_boundary_bruteforce(
        system=system,
        x0=x0,
        T=T,
        num_time_steps=num_time_steps,
        phis=phis,
        control_candidates=control_candidates,
    )
    print(f"[Linear] OC boundary (bruteforce) computed with {R_oc_bruteforce.shape[0]} points.")

    # --- 6) Hausdorff distance: grid reachable set vs OC (Pyomo)
    hd_grid_oc = hausdorff_distance(R_grid_for_OC, R_oc_pyomo)
    print(f"[Linear] Hausdorff distance between grid (grid thinning) and OC (Pyomo) sets: "
          f"d_H = {hd_grid_oc.distance:.4f}")

    # --- 7) Hausdorff distance: OC (Pyomo) vs OC (bruteforce)
    hd_oc_vs_bf = hausdorff_distance(R_oc_pyomo, R_oc_bruteforce)
    print(f"[Linear] Hausdorff distance between OC (Pyomo) and OC (bruteforce) sets: "
          f"d_H = {hd_oc_vs_bf.distance:.4f}")

    # --- 8) Plot: grid reachable set vs OC boundary (Pyomo)
    plot_reachable_sets(
        R_grid=R_grid_for_OC,
        R_oc=R_oc_pyomo,
        hd=hd_grid_oc,
        title="Linear system (disk control): grid reachable set vs OC boundary (Pyomo)",
        save_path=None,
    )

    # --- 9) Plot: OC (Pyomo) vs OC (bruteforce) — separate plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")

    R_oc_py_closed = np.vstack([R_oc_pyomo, R_oc_pyomo[0]])
    R_oc_bf_closed = np.vstack([R_oc_bruteforce, R_oc_bruteforce[0]])

    ax.plot(R_oc_py_closed[:, 0], R_oc_py_closed[:, 1],
            linewidth=2.0, label="OC boundary (Pyomo, linear system)")
    ax.plot(R_oc_bf_closed[:, 0], R_oc_bf_closed[:, 1],
            linestyle="--", marker="o", markersize=4,
            label="OC boundary (bruteforce, linear system)")

    pa = hd_oc_vs_bf.point_a
    pb = hd_oc_vs_bf.point_b
    ax.scatter([pa[0]], [pa[1]], color="red", s=60, label="Hausdorff point A")
    ax.scatter([pb[0]], [pb[1]], color="green", s=60, label="Hausdorff point B")
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
            linestyle=":", color="black",
            label=f"Hausdorff segment (d={hd_oc_vs_bf.distance:.3f})")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Linear system (disk control): OC boundary (Pyomo) vs bruteforce")
    ax.grid(True)
    ax.legend(loc="best")

    plt.show()


# ---------------------------------------------------------------------
# 2. Li–Markus example
# ---------------------------------------------------------------------


class LiMarkusSystem:
    """
    Li–Markus example:

        x1' = x2 * u1 - x1 * u2
        x2' = -x1 * u1 - x2 * u2

    with elliptic control set

        u1^2 + 25 u2^2 <= 1

    and initial condition x(0) = (1, 0).

    This system is nonlinear in the control and is a classical example
    where the reachable set is non-convex for small times T.
    """

    def f_numpy(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Vectorised dynamics for use with the grid-based method.

        X : array (..., 2)
        U : array (..., 2)
        returns array (..., 2) with (x1', x2').
        """
        x1 = X[..., 0]
        x2 = X[..., 1]
        u1 = U[..., 0]
        u2 = U[..., 1]

        dx1 = x2 * u1 - x1 * u2
        dx2 = -x1 * u1 - x2 * u2

        return np.stack((dx1, dx2), axis=-1)


def generate_li_markus_controls(num_controls: int) -> np.ndarray:
    """
    Generate controls on the boundary of the ellipse

        u1^2 + 25 u2^2 = 1.

    Parameterisation:
        u1 = cos(theta),
        u2 = (1/5) * sin(theta),
    so that u1^2 + 25 u2^2 = cos^2 + sin^2 = 1.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_controls, endpoint=False)
    u1 = np.cos(angles)
    u2 = (1.0 / 5.0) * np.sin(angles)
    return np.stack((u1, u2), axis=1)


def run_li_markus_example():
    """
    Run the Li–Markus example for several values of T.

    For each T we:
      - compute the reachable set via the grid method (NumPy backend),
      - compute a boundary via brute-force constant controls (on the ellipse),
      - if there are at least 3 points, compute the convex hull of the
        reachable set and the Hausdorff distance to it,
      - classify the reachable set as "numerically non-convex" or
        "approximately convex",
      - produce a plot with clear annotations.
    """
    from scipy.spatial import ConvexHull

    print("=" * 80)
    print("Li–Markus example:")
    print("x1' = x2*u1 - x1*u2, x2' = -x1*u1 - x2*u2,  u1^2 + 25 u2^2 <= 1,  x(0) = (1, 0)")
    print("Investigating non-convexity of the reachable set for different T.")
    print("=" * 80)

    system_li = LiMarkusSystem()
    x0_li = np.array([1.0, 0.0], dtype=float)

    # Controls on the ellipse boundary
    num_controls_li = 32
    controls_li = generate_li_markus_controls(num_controls_li)

    # Values of T to investigate
    T_list = [0.2, 0.4, 0.6, 0.8, 1.2, 2.0]
    num_time_steps_li = 80
    thinning_h_li = 0.01  # slightly less aggressive thinning
    num_directions_li = 64

    for T in T_list:
        print(f"\n[Li–Markus] T = {T:.2f}")

        cfg = ReachabilityConfig(
            T=T,
            num_time_steps=num_time_steps_li,
            thinning_method="grid",
            thinning_param=thinning_h_li,
            backend="numpy",
        )

        # Grid-based reachable set
        R_grid_li = compute_reachable_set_grid(system_li, x0_li, controls_li, cfg)
        n_pts = R_grid_li.shape[0]
        print(f"[Li–Markus] Reachable set size (grid) for T = {T:.2f}: {n_pts} points")

        # Boundary via brute-force constant controls on the ellipse
        phis = np.linspace(0.0, 2.0 * np.pi, num_directions_li, endpoint=False)
        R_oc_li = compute_oc_boundary_bruteforce(
            system=system_li,
            x0=x0_li,
            T=T,
            num_time_steps=num_time_steps_li,
            phis=phis,
            control_candidates=controls_li,
        )
        print(f"[Li–Markus] OC boundary (constant controls on ellipse) has "
              f"{R_oc_li.shape[0]} points.")

        # ====== degenerate case: too few points for convex hull ======
        if n_pts < 3:
            print(f"[Li–Markus] Too few points ({n_pts}) to build a convex hull. "
                  "The reachable set is essentially a single point (very small T).")
            status = "degenerate (almost a single point)"

            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            ax.set_aspect("equal", adjustable="box")

            ax.scatter(
                R_grid_li[:, 0],
                R_grid_li[:, 1],
                s=50,
                color="tab:blue",
                label="Grid reachable set (Li–Markus)",
            )

            R_oc_closed = np.vstack([R_oc_li, R_oc_li[0]])
            ax.plot(
                R_oc_closed[:, 0],
                R_oc_closed[:, 1],
                lw=2.0,
                color="tab:orange",
                label="Boundary (constant controls on ellipse)",
            )

            ax.set_title(
                f"Li–Markus reachable set, T = {T:.2f}\n"
                f"Status: {status}"
            )
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.grid(True)
            ax.tick_params(labelsize=11)

            # легенда вне графика, чтобы ничего не перекрывала
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                fontsize=10,
            )

            fig.tight_layout()
            plt.show()
            continue

        # ====== нормальный случай: можно строить выпуклую оболочку ======
        hull = ConvexHull(R_grid_li)
        hull_points = R_grid_li[hull.vertices]

        # Hausdorff distance between reachable set and its convex hull
        hd_hull = hausdorff_distance(R_grid_li, hull_points)
        print("[Li–Markus] Hausdorff distance between reachable set and its "
              f"convex hull: d_H = {hd_hull.distance:.3f}")

        tol = 0.03
        status = ("numerically non-convex"
                  if hd_hull.distance > tol
                  else "approximately convex")
        print(f"[Li–Markus] Qualitative convexity status for T = {T:.2f}: {status}")

        # Plotting
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.set_aspect("equal", adjustable="box")

        ax.scatter(
            R_grid_li[:, 0],
            R_grid_li[:, 1],
            s=8,
            alpha=0.4,
            label="Grid reachable set (Li–Markus)",
        )

        # Boundary (constant controls)
        R_oc_closed = np.vstack([R_oc_li, R_oc_li[0]])
        ax.plot(
            R_oc_closed[:, 0],
            R_oc_closed[:, 1],
            lw=2.0,
            color="tab:orange",
            label="Boundary (constant controls on ellipse)",
        )

        # Convex hull
        hull_closed = np.vstack([hull_points, hull_points[0]])
        ax.plot(
            hull_closed[:, 0],
            hull_closed[:, 1],
            lw=1.8,
            linestyle="--",
            color="black",
            label="Convex hull of reachable set",
        )

        # Hausdorff segment between reachable set and its convex hull
        pa = hd_hull.point_a
        pb = hd_hull.point_b
        ax.scatter([pa[0]], [pa[1]], color="red", s=60, label="Hausdorff point A")
        ax.scatter([pb[0]], [pb[1]], color="green", s=60, label="Hausdorff point B")
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            linestyle=":",
            color="black",
            label=f"Hausdorff segment (d={hd_hull.distance:.3f})",
        )

        ax.set_title(
            f"Li–Markus reachable set, T = {T:.2f}\n"
            f"Status: {status}"
        )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True)
        ax.tick_params(labelsize=11)

        # легенда за пределами области рисования
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=10,
        )

        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# 3. Optional benchmark (unchanged logic, can be used if needed)
# ---------------------------------------------------------------------


def benchmark_reachability_speed():
    """
    Benchmark for the grid-based reachability method using NumPy vs Torch backends
    on the linear system.

    It sweeps over different numbers of controls (num_controls) and time steps
    (num_time_steps), measures runtime for each backend, and prints the speedup.

    Run this in Colab with GPU enabled to find a configuration where Torch
    (on cuda) is faster than NumPy.
    """
    import time
    import torch as _torch

    print("Benchmark reachability (NumPy vs Torch on cuda)")
    print("CUDA available:", _torch.cuda.is_available())
    print("-" * 80)

    x0 = np.array([0.0, 0.0], dtype=float)
    T = 1.0
    u_max = 1.0

    system_np = ControlledSystem(u_max=u_max)
    system_torch = ControlledSystemTorch(u_max=u_max)

    num_controls_list = [32, 64, 128]
    num_time_steps_list = [40, 80, 160]
    thinning_h = 0.02

    for num_controls in num_controls_list:
        controls = generate_controls_disk(
            num_controls=num_controls,
            u_max=u_max,
            on_circle=True,
        )

        for num_time_steps in num_time_steps_list:
            cfg_numpy = ReachabilityConfig(
                T=T,
                num_time_steps=num_time_steps,
                thinning_method="grid",
                thinning_param=thinning_h,
                backend="numpy",
            )

            cfg_torch = ReachabilityConfig(
                T=T,
                num_time_steps=num_time_steps,
                thinning_method="grid",
                thinning_param=thinning_h,
                backend="torch",
                torch_device="cuda",
            )

            # Warm-up Torch
            _ = compute_reachable_set_grid(system_torch, x0, controls, cfg_torch)

            # NumPy
            t0 = time.perf_counter()
            R_np = compute_reachable_set_grid(system_np, x0, controls, cfg_numpy)
            t1 = time.perf_counter()
            t_numpy = t1 - t0

            # Torch
            t0 = time.perf_counter()
            R_torch = compute_reachable_set_grid(system_torch, x0, controls, cfg_torch)
            t1 = time.perf_counter()
            t_torch = t1 - t0

            size_np = R_np.shape[0]
            size_torch = R_torch.shape[0]

            speedup = t_numpy / t_torch if t_torch > 0 else float("inf")
            print(
                f"num_controls={num_controls:3d}, "
                f"num_time_steps={num_time_steps:3d} | "
                f"NumPy={t_numpy:6.3f}s, Torch={t_torch:6.3f}s, "
                f"speedup={speedup:4.2f}x, "
                f"sizes: np={size_np}, torch={size_torch}"
            )

    print("-" * 80)


# ---------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------


def run_experiment():
    """
    Run both experiments:

    1) Linear system with disk control (convex reachable set).
    2) Li–Markus example with elliptic control (non-convex reachable set for
       small T, approaching convexity for larger T).
    """
    run_linear_example()
    run_li_markus_example()


if __name__ == "__main__":
    run_experiment()
    # If needed, you can also run the benchmark separately:
    # benchmark_reachability_speed()

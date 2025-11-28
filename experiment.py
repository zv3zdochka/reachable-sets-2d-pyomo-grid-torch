# experiment.py
"""
experiment.py

Example driver code that uses all modules to:

1. Build the reachable set via the grid method (NumPy backend) with grid-based thinning.
2. Optionally, build the reachable set via the grid method with Poisson Disk thinning.
3. Build the reachable set via the grid method (Torch backend) and measure acceleration.
4. Build the boundary of the reachable set via Pyomo optimal control.
5. Build an alternative boundary via a brute-force sweep of constant controls on the circle.
6. Plot:
   - reachable set (grid) vs OC boundary (Pyomo),
   - OC boundary (Pyomo) vs OC boundary (bruteforce).
7. Compute and visualize the Hausdorff distance:
   - between grid reachable set and OC boundary (Pyomo),
   - between OC boundary (Pyomo) and OC boundary (bruteforce).

This script is intended to be run in a Jupyter/Colab environment. Adjust
parameters (T, num_time_steps, etc.) if needed.
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
    thinning_h = 0.05  # grid step for thinning
    thinning_r = 0.05  # Poisson radius for thinning

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
    print(
        f"[NumPy backend, Poisson thinning] Reachable set size: {R_grid_poisson.shape[0]}, time = {time_numpy_poisson:.3f} s")

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
            torch_device="cuda",  # поставь "cpu", если GPU нет/не нужен
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

        # Используем NumPy-результат как основной для сравнения с ОУ
        R_grid = R_grid_numpy

    except Exception as e:
        print(f"Torch backend not available or failed ({e}); using NumPy result only.")
        R_grid = R_grid_numpy

    # 4. Boundary via optimal control (Pyomo)
    num_directions = 32
    print("Solving optimal control problems for boundary (Pyomo)...")
    R_oc = compute_oc_boundary(
        system=system,
        x0=x0,
        T=T,
        num_time_steps=num_time_steps,
        num_directions=num_directions,
        solver_name="ipopt",  # make sure IPOPT is installed
    )
    print(f"OC boundary (Pyomo) computed with {R_oc.shape[0]} points.")

    # 4a. Boundary via brute-force sweep over constant controls on the circle
    print("Computing OC boundary by brute-force over constant controls on the circle...")

    # Directions the same as for Pyomo boundary
    phis = np.linspace(0.0, 2.0 * np.pi, num_directions, endpoint=False)
    # Candidate controls on the control circle (можно взять сетку погуще, чем для сеточного метода)
    num_controls_bruteforce = 32
    control_candidates = generate_controls_disk(
        num_controls=num_controls_bruteforce,
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
    print(f"OC boundary (bruteforce) computed with {R_oc_bruteforce.shape[0]} points.")

    # 5. Hausdorff distance: grid reachable set vs OC (Pyomo)
    hd_grid_oc = hausdorff_distance(R_grid, R_oc)
    print(f"Hausdorff distance between grid (grid thinning) and OC (Pyomo) sets: d_H = {hd_grid_oc.distance:.4f}")

    # 5a. Hausdorff distance: OC (Pyomo) vs OC (bruteforce)
    hd_oc_vs_bf = hausdorff_distance(R_oc, R_oc_bruteforce)
    print(f"Hausdorff distance between OC (Pyomo) and OC (bruteforce) sets: d_H = {hd_oc_vs_bf.distance:.4f}")

    # 6. Plot: grid reachable set vs OC (Pyomo)
    plot_reachable_sets(
        R_grid=R_grid,
        R_oc=R_oc,
        hd=hd_grid_oc,
        title="Reachable set (grid method) vs OC boundary (Pyomo)",
        save_path=None,
    )

    # 6a. Plot: OC (Pyomo) vs OC (bruteforce) — отдельный график
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")

    # Замыкаем ломаные для красоты
    R_oc_closed = np.vstack([R_oc, R_oc[0]])
    R_oc_bf_closed = np.vstack([R_oc_bruteforce, R_oc_bruteforce[0]])

    ax.plot(R_oc_closed[:, 0], R_oc_closed[:, 1],
            linewidth=2.0, label="OC boundary (Pyomo)")
    ax.plot(R_oc_bf_closed[:, 0], R_oc_bf_closed[:, 1],
            linestyle="--", marker="o", markersize=4,
            label="OC boundary (bruteforce)")

    # Пара точек Хаусдорфа Pyomo vs bruteforce
    pa = hd_oc_vs_bf.point_a
    pb = hd_oc_vs_bf.point_b
    ax.scatter([pa[0]], [pa[1]], color="red", s=60, label="Hausdorff point A")
    ax.scatter([pb[0]], [pb[1]], color="green", s=60, label="Hausdorff point B")
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
            linestyle=":", color="black",
            label=f"Hausdorff segment (d={hd_oc_vs_bf.distance:.3f})")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("OC boundary: Pyomo vs bruteforce constant controls")
    ax.grid(True)
    ax.legend(loc="best")

    plt.show()


def benchmark_reachability_speed():
    """
    Benchmark for the grid-based reachability method using NumPy vs Torch backends.

    It sweeps over different numbers of controls (num_controls) and time steps
    (num_time_steps), measures runtime for each backend, and prints the speedup.

    Run this in Colab with GPU enabled to find a configuration where Torch
    (on cuda) is at least 2x faster than NumPy.
    """
    import time

    x0 = np.array([0.0, 0.0], dtype=float)
    T = 1.0
    u_max = 1.0

    system_np = ControlledSystem(u_max=u_max)
    system_torch = ControlledSystemTorch(u_max=u_max)

    # Наборы, по которым будем перебирать.
    # Если будет мало нагрузки, можно увеличить.
    num_controls_list = [16, 64, 128, 256]
    num_time_steps_list = [40, 80, 160]

    thinning_h = 0.03  # шаг для grid-thinning (фиксируем один)

    print("Benchmark reachability (NumPy vs Torch on cuda)")
    print("CUDA available:", __import__("torch").cuda.is_available())
    print("-" * 80)

    for num_controls in num_controls_list:
        controls = generate_controls_disk(
            num_controls=num_controls,
            u_max=u_max,
            on_circle=True,
        )

        for num_time_steps in num_time_steps_list:
            # Конфиг для NumPy
            cfg_numpy = ReachabilityConfig(
                T=T,
                num_time_steps=num_time_steps,
                thinning_method="grid",
                thinning_param=thinning_h,
                backend="numpy",
            )

            # Конфиг для Torch (cuda)
            cfg_torch = ReachabilityConfig(
                T=T,
                num_time_steps=num_time_steps,
                thinning_method="grid",
                thinning_param=thinning_h,
                backend="torch",
                torch_device="cuda",
            )

            # Прогрев Torch (маленький запуск, чтобы не считать его время)
            _ = compute_reachable_set_grid(system_torch, x0, controls, cfg_torch)

            # Замер NumPy
            t0 = time.perf_counter()
            R_np = compute_reachable_set_grid(system_np, x0, controls, cfg_numpy)
            t1 = time.perf_counter()
            t_numpy = t1 - t0

            # Замер Torch
            t0 = time.perf_counter()
            R_torch = compute_reachable_set_grid(system_torch, x0, controls, cfg_torch)
            t1 = time.perf_counter()
            t_torch = t1 - t0

            # На всякий случай проверим, что размеры сопоставимы
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
    print("Pick any combination with speedup >= 2.0x and reuse those parameters")
    print("in run_experiment() for the 'accelerated method' part of the report.")


def benchmark_reachability_speed():
    """
    Benchmark for the grid-based reachability method using NumPy vs Torch backends.

    It sweeps over different numbers of controls (num_controls) and time steps
    (num_time_steps), measures runtime for each backend, and prints the speedup.

    Run this in Colab with GPU enabled to find a configuration where Torch
    (on cuda) is at least 2x faster than NumPy.
    """
    import time

    x0 = np.array([0.0, 0.0], dtype=float)
    T = 1.0
    u_max = 1.0

    system_np = ControlledSystem(u_max=u_max)
    system_torch = ControlledSystemTorch(u_max=u_max)

    # Чуть более тяжёлые параметры
    num_controls_list = [32, 64, 128]
    num_time_steps_list = [40, 80, 160]

    thinning_h = 0.02  # поменьше шаг -> больше точек

    print("Benchmark reachability (NumPy vs Torch on cuda)")
    import torch as _torch
    print("CUDA available:", _torch.cuda.is_available())
    print("-" * 80)

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

            # Прогрев Torch
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


if __name__ == "__main__":
    run_experiment()

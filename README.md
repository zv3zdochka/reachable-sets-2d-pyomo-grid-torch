# Reachable Sets for 2D Controlled Systems

This repository contains a small numerical study of reachable sets for a two–dimensional controlled system. The project implements:

- a grid-based (point–cloud) method for reachable sets in Python,
- Poisson disk and grid-based thinning of point clouds,
- an accelerated implementation of the grid method using PyTorch (CPU / GPU),
- construction of the reachable set boundary via optimal control problems in Pyomo,
- computation and visualisation of the Hausdorff distance between two reachable-set approximations.

The code is designed to run both locally and in Google Colab.

---

## Mathematical model

We consider a controlled system
$$
\dot x(t) = f(x(t), u(t)), \quad x(t) \in \mathbb{R}^2,\ u(t) \in P,\ t \in [0, T],
$$
with a fixed initial condition \(x(0) = x_0\).

In this project the dynamics are chosen as a simple linear system with additive control
$$
\begin{aligned}
\dot x_1 &= x_2 + u_1, \\
\dot x_2 &= -x_1 + u_2,
\end{aligned}
$$
where \(u = (u_1, u_2)\) is a 2-dimensional control.

The control set is a disk
$$
P = \{ u \in \mathbb{R}^2 : \lVert u \rVert_2 \le u_{\max} \}.
$$

The reachable set at time \(T\) is
$$
\mathcal{R}(T)
=
\{ x(T) : x(0) = x_0,\ \dot x = f(x,u),\ u(\cdot) \in \mathcal{U} \}.
$$

In the code we build two numerical approximations:

- \(\mathcal{R}_{\mathrm{grid}}(T)\) — via the grid (point–cloud) method,
- \(\mathcal{R}_{\mathrm{OC}}(T)\) — via optimal control problems in given directions.

---

## Implemented methods

### Grid-based reachable set

Time is discretised as
\( 0 = t_0 < t_1 < \dots < t_N = T \)
with step \(\Delta t = T / N\).
The control set \(P\) is approximated by a finite subset
\(\{u^k\}_{k=1}^M \subset P\) (points on a circle or ellipse).

On each time step \(i\):

1. At step \(i\) we have a cloud of points \(W_i\).
2. For every \(x \in W_i\) and every control \(u^k\) we compute
   $$
   x^{\text{new}} = x + \Delta t\, f(x, u^k)
   $$
   using the explicit Euler scheme.
3. All new points are collected into a cloud \(\widetilde W_{i+1}\).
4. A thinning procedure is applied to \(\widetilde W_{i+1}\), producing \(W_{i+1}\).

After \(N\) steps, \(W_N\) is taken as an approximation of \(\mathcal{R}_{\mathrm{grid}}(T)\).

Two thinning strategies are implemented:

- **Grid thinning** — a rectangular grid with step \(h\); at most one representative per grid cell.
- **Poisson disk thinning** — a simple greedy algorithm that enforces a minimum distance \(r\) between any two kept points.

The baseline implementation uses NumPy on CPU. An accelerated implementation uses PyTorch and can run on GPU; in the accelerated version both the propagation step and grid thinning are done in pure Torch.

### Boundary via optimal control (Pyomo)

For a direction
\(\ell(\varphi) = (\cos \varphi, \sin \varphi)\)
we solve the optimal control problem
$$
\max_{u(\cdot)} \ \langle \ell(\varphi), x(T) \rangle
$$
subject to the system dynamics and the control constraint \(\lVert u(t) \rVert_2 \le u_{\max}\).

The problem is discretised with the same time grid. The decision variables are
\(x_k \in \mathbb{R}^2\) and \(u_k \in \mathbb{R}^2\) for \(k = 0, \dots, N-1\).  
The discrete dynamics are
$$
x_{k+1} = x_k + \Delta t\, f(x_k, u_k), \quad k = 0, \dots, N-1.
$$

For each direction angle \(\varphi\) we solve this problem in Pyomo and obtain one boundary point \(x_\varphi(T)\). A set of such points for uniformly spaced angles in \([0, 2\pi)\) forms the boundary approximation \(\mathcal{R}_{\mathrm{OC}}(T)\).

In addition, a simpler brute-force variant is implemented: for each direction \(\varphi\) we consider only constant controls \(u(t) \equiv u\) with \(u\) taken from a discrete set on the control circle and choose the one that maximises \(\langle \ell(\varphi), x(T) \rangle\).

### Hausdorff distance

For two finite point clouds \(A, B \subset \mathbb{R}^2\) the (symmetric) Hausdorff distance is
$$
d_H(A,B)
=
\max\left\{
\max_{a \in A} \min_{b \in B} \lVert a - b \rVert_2,\,
\max_{b \in B} \min_{a \in A} \lVert b - a \rVert_2
\right\}.
$$

The code computes:

- the directed distances \(d(A,B)\) and \(d(B,A)\),
- the Hausdorff distance \(d_H(A,B)\),
- one pair of points \((a^\*, b^\*)\) where this distance is attained (approximately).

Both reachable-set approximations are plotted on one figure, and the Hausdorff distance is visualised as a line segment between \(a^\*\) and \(b^\*\).

---

## Project structure

Main modules:

- `system.py`  
  Definition of the controlled system:
  - `ControlledSystem` with NumPy dynamics `f_numpy`,
  - `ControlledSystemTorch` with Torch dynamics `f_torch`.

- `controls.py`  
  Generation of discrete control sets:
  - `generate_controls_disk` — points on a circle (disk boundary),
  - `generate_controls_box` — grid in a rectangle,
  - `generate_controls_ellipse` — grid on an ellipse.

- `thinning.py`  
  Point-cloud thinning:
  - `thin_grid(points, h)` — grid-based thinning in NumPy,
  - `thin_poisson(points, r)` — Poisson disk-like thinning in NumPy.

- `backend_numpy.py`  
  Baseline propagation step:
  - `propagate_numpy(system, states, controls, dt)`.

- `backend_torch.py`  
  Torch-based propagation and thinning:
  - `propagate_torch_tensor(system, states_t, controls_t, dt)` — propagation on Torch,
  - `thin_grid_torch(points_t, h)` — grid thinning on Torch,
  - `propagate_torch_numpy(...)` — compatibility wrapper (NumPy → Torch → NumPy).

- `grid_reachability.py`  
  High-level grid-based reachable-set computation:
  - `ReachabilityConfig` — configuration (final time, number of steps, backend, thinning),
  - `compute_reachable_set_grid(system, x0, controls, cfg)` — returns a point cloud at time \(T\),
  - internally selects either NumPy or Torch implementation.

- `ocp_pyomo.py`  
  Optimal control boundary computation:
  - `solve_ocp_direction(phi, system, x0, T, num_time_steps, solver_name)` — one direction,
  - `compute_oc_boundary(system, x0, T, num_time_steps, num_directions, solver_name)` — Pyomo boundary,
  - `compute_oc_boundary_bruteforce(system, x0, T, num_time_steps, phis, control_candidates)` — brute-force variant with constant controls.

- `hausdorff.py`  
  Hausdorff distance utilities:
  - `HausdorffResult` — distance and the corresponding pair of points,
  - `hausdorff_distance(A, B)` — symmetric Hausdorff distance between two clouds.

- `plotting.py`  
  Plotting helper:
  - `plot_reachable_sets(R_grid, R_oc, hd, title, save_path)` — one figure with both sets and the Hausdorff segment.

- `experiment.py`  
  Main script with experiments:
  - `run_experiment()`:
    - builds \(\mathcal{R}_{\mathrm{grid}}(T)\) with NumPy and Torch backends,
    - computes the Pyomo-based boundary \(\mathcal{R}_{\mathrm{OC}}(T)\),
    - computes a brute-force constant-control boundary,
    - evaluates the Hausdorff distances,
    - produces plots.
  - `benchmark_reachability_speed()`:
    - sweeps over `num_controls` and `num_time_steps`,
    - measures runtime for NumPy vs Torch (GPU),
    - prints speedup and cloud sizes.

---

## Installation

The project targets Python 3.10+.

Required Python packages:

- `numpy`
- `matplotlib`
- `pyomo`
- `torch` (PyTorch)
- `scipy` (optional, used for a KD-tree in the Hausdorff computation)

Install with:

```bash
pip install numpy matplotlib pyomo torch scipy
```

For the Pyomo optimal control problems you also need a nonlinear optimisation solver such as IPOPT. Installation depends on your environment. In Google Colab you can, for example, download prebuilt IPOPT binaries and add them to `PATH` (see the notebook for an example).

---

## Usage

### Running locally

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt  # if you decide to add one
```

Make sure a Pyomo-compatible solver (e.g. `ipopt`) is available on your `PATH`.

Then run:

```bash
python experiment.py
```

The script will:

* build the reachable set using the grid method (NumPy and Torch backends),
* solve optimal control problems in multiple directions,
* compute Hausdorff distances between the different approximations,
* show two plots:

  * grid reachable set vs Pyomo boundary,
  * Pyomo boundary vs brute-force constant-control boundary.

### Running in Google Colab

A typical Colab workflow:

1. Upload all `*.py` files to the working directory or unzip the project archive.

2. Install dependencies:

   ```python
   !pip install numpy matplotlib pyomo torch scipy
   # plus installation of IPOPT or another solver
   ```

3. Enable GPU in the Colab runtime (Runtime → Change runtime type → GPU).

4. Import and run:

   ```python
   from experiment import run_experiment
   run_experiment()
   ```

To run the speed benchmark:

```python
from experiment import benchmark_reachability_speed
benchmark_reachability_speed()
```

This prints NumPy vs Torch timings and speedup for several parameter combinations.

---

## Notes on acceleration

The Torch backend keeps the point cloud as a `torch.Tensor` on the selected device and performs both the propagation step and grid-based thinning on that device. This minimises CPU↔GPU data transfers and allows the method to benefit from GPU parallelism for sufficiently large configurations (number of controls, number of time steps, thinning step).

The function `benchmark_reachability_speed()` can be used to explore parameter ranges where the Torch backend provides a significant speedup over the NumPy baseline.

---

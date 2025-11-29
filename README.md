# Reachable Sets of 2D Controlled Systems (Grid vs Optimal Control, PyTorch Acceleration)

This repository contains numerical experiments with reachable sets of two–dimensional controlled systems.  
The main goals are:

1. Build the reachable set of a planar controlled system by a **grid (point–cloud) method** in Python.
2. **Accelerate** the grid method using a PyTorch backend (CPU / GPU) and measure the speedup.
3. Construct the **boundary** of the reachable set via **optimal control problems in Pyomo**.
4. Plot the two reachable–set approximations on the same figure and compute the **Hausdorff distance** between them.
5. Study a **non-convex reachable set** using the classical **Li–Markus example** and quantify its deviation from convexity via the distance to its convex hull.

The code is structured so that it can be run both locally and in Google Colab (with GPU).

---

## 1. Mathematical models

### 1.1 Linear system with disk control (convex reachable set)

This is the "baseline" system used for most of the comparisons and the Pyomo boundary:

$$\dot x(t) = f(x(t), u(t)), \quad x(t) \in \mathbb{R}^2,\ u(t) \in P.$$

Dynamics:

$$\begin{aligned}
\dot x_1 &= x_2 + u_1,\\
\dot x_2 &= -x_1 + u_2,
\end{aligned}$$

with control set

$$P = \{u \in \mathbb{R}^2 : \lVert u \rVert_2 \le u_{\max}\}.$$

Initial condition:

$$x(0) = x_0.$$

The reachable set at time $T$ is

$$\mathcal{R}(T)
  = \{x(T) : x(0) = x_0,\ \dot x = f(x,u),\ u(\cdot) \in \mathcal{U}\}.$$

Because the system is linear and the control set is convex, $\mathcal{R}(T)$ is convex for any $T > 0$.

### 1.2 Li–Markus example (nonlinear in control, non-convex reachable set)

The second model is the classical Li–Markus example, used here to demonstrate non-convex reachable sets:

$$\begin{aligned}
\dot x_1 &= x_2 u_1 - x_1 u_2,\\
\dot x_2 &= -x_1 u_1 - x_2 u_2,
\end{aligned}$$

with elliptic control set

$$u_1^2 + 25 u_2^2 \le 1,$$

and initial condition

$$x(0) = (1, 0).$$

For this system the reachable set in a fixed time interval is generally **non-convex**.  
In the project, non-convexity is visualised and quantified via the Hausdorff distance between the reachable set and its convex hull.

---

## 2. Implemented methods

### 2.1 Grid (point-cloud) reachable set

Time is discretised into $N$ steps:

$$0 = t_0 < t_1 < \dots < t_N = T,\quad \Delta t = T/N.$$

The control set $P$ is approximated by a finite subset
$\{u^k\}_{k=1}^M \subset P$ (points on a circle or ellipse).

On each time step $i$:

1. The current point cloud is $W_i$.
2. For every $x \in W_i$ and every discrete control $u^k$, one step of the explicit Euler scheme is applied:
   $$x^{\text{new}} = x + \Delta t\, f(x, u^k).$$
3. All new points are collected into a temporary cloud $\widetilde{W}\_{i+1}$.
4. A **thinning** procedure is applied to $\widetilde{W}\_{i+1}$, producing a reduced cloud $W\_{i+1}$.

After $N$ steps, $W\_N$ is taken as a numerical approximation of $\mathcal{R}(T)$.

Two thinning strategies are implemented:

- **Grid thinning** — a rectangular grid with step $h$; at most one representative per grid cell.
- **Poisson disk thinning** — a greedy algorithm enforcing a minimum pairwise distance $r$ between retained points.

The baseline implementation uses **NumPy** on CPU.  
An accelerated backend uses **PyTorch** and can run both the propagation and grid thinning on GPU.

### 2.2 Pyomo method via directions on a circle / ellipse

For the linear system, the boundary of the reachable set is constructed in Pyomo by solving a family of optimal control problems along different directions.

For each direction angle $\varphi$ define

$$\ell(\varphi) = (\cos\varphi,\ \sin\varphi).$$

The optimisation problem is

$$\max_{u(\cdot)}\ \langle \ell(\varphi), x(T) \rangle$$

subject to

- dynamics $\dot x = f(x,u)$,
- control constraint $\lVert u(t)\rVert_2 \le u_{\max}$,
- initial condition $x(0) = x_0$.

The problem is discretised on the same time grid as the grid method:

- decision variables: $x_k \in \mathbb{R}^2$, $u_k \in \mathbb{R}^2$ for $k = 0,\dots,N-1$;
- dynamics:
  $$x_{k+1} = x_k + \Delta t\, f(x_k, u_k).$$

For each $\varphi$ one optimal final point $x_\varphi(T)$ is obtained; the set $\{x_\varphi(T)\}$ over a uniform grid of angles approximates the boundary of the reachable set.  
This is exactly **method 1** in the supervisor's description: "through a circle/ellipse and a cycle of optimal control problems with linear functionals".

For the Li–Markus example this method would approximate the **convex hull** of the reachable set (wells/non-convex parts are not seen by support functionals), therefore non-convexity is analysed using the grid method instead.

A simpler Pyomo-free variant is also implemented: "brute force constant controls". For each direction $\varphi$, a finite set of constant controls $u(t)\equiv u$ on a circle/ellipse is tried; the best end point in the direction $\ell(\varphi)$ is chosen.

### 2.3 Hausdorff distance and convexity analysis

For two finite point clouds $A, B \subset \mathbb{R}^2$ the symmetric Hausdorff distance is

$d_H(A,B)
= \max\left\\{
\max_{a\in A}\min_{b\in B}\lVert a-b\rVert_2,\,
\max_{b\in B}\min_{a\in A}\lVert b-a\rVert_2
\right\\}.$

The code computes:

- the Hausdorff distance between the grid reachable set and the Pyomo boundary;
- for the Li–Markus example: the Hausdorff distance between the grid reachable set and its **convex hull** (via `scipy.spatial.ConvexHull`).

The distance to the convex hull is used as a **numerical measure of non-convexity**; a qualitative status ("numerically non-convex" vs "approximately convex") is printed for each time horizon $T$.

---

## 3. Project structure and file responsibilities

All files are in the project root.

### 3.1 Core dynamics and controls

- **`system.py`**

  - `ControlledSystem`  
    Linear system with disk control, provides `f_numpy(x, u)` used in the grid method.
  - `ControlledSystemTorch`  
    Same dynamics implemented in PyTorch, provides `f_torch(X, U)` for batched propagation on CPU/GPU.

- **`controls.py`**

  - `generate_controls_disk(num_controls, u_max, on_circle)`  
    Discrete controls uniformly distributed on a circle (boundary of the control disk) or inside the disk.
  - `generate_controls_box(...)`, `generate_controls_ellipse(...)`  
    Helpers for rectangular and elliptic control sets (not all variants are used in the final experiment but can be reused).

### 3.2 Thinning and backends

- **`thinning.py`**

  - `thin_grid(points, h)`  
    NumPy implementation of grid-based thinning: at most one point per grid cell of size $h$.
  - `thin_poisson(points, r)`  
    Simple Poisson disk-like thinning: greedy selection of points at distance at least $r$ from each other.

- **`backend_numpy.py`**

  - `propagate_numpy(system, states, controls, dt)`  
    Vectorised explicit Euler step for the grid method. Given a cloud of states and a finite control set, returns the concatenated cloud of all successors.

- **`backend_torch.py`**

  - `propagate_torch_tensor(system, states_t, controls_t, dt)`  
    Torch version of the propagation step on device tensors.
  - `thin_grid_torch(points_t, h)`  
    Grid thinning fully implemented in Torch (no CPU round-trips), suitable for GPU acceleration.
  - `propagate_torch_numpy(...)`  
    Backwards-compatible wrapper that accepts NumPy arrays and internally uses Torch.

### 3.3 High-level grid method

- **`grid_reachability.py`**

  - `ReachabilityConfig`  
    Configuration dataclass:
    - final time `T`,
    - number of time steps `num_time_steps`,
    - backend (`"numpy"` or `"torch"`),
    - thinning method (`"grid"` or `"poisson"`),
    - thinning parameter (`h` or `r`),
    - optional `torch_device` (`"cpu"` or `"cuda"`).
  - `compute_reachable_set_grid(system, x0, controls, cfg)`  
    Main function of the grid method:
    - iterates over time steps,
    - calls the chosen backend,
    - applies thinning on each step,
    - returns the final point cloud at time $T$.

### 3.4 Pyomo optimal control

- **`ocp_pyomo.py`**

  - `solve_ocp_direction(phi, system, x0, T, num_time_steps, solver_name)`  
    Builds and solves one optimal control problem in direction $\ell(\varphi)$ for the linear system.
  - `compute_oc_boundary(system, x0, T, num_time_steps, num_directions, solver_name)`  
    Runs a loop over directions and collects the optimal end points into a boundary point set.
  - `compute_oc_boundary_bruteforce(system, x0, T, num_time_steps, phis, control_candidates)`  
    Builds an approximate boundary by sweeping constant controls on a circle/ellipse. Used both in the linear system and in the Li–Markus example.

For the Li–Markus system a small class `LiMarkusSystem` with `f_numpy` is defined directly in `experiment.py`, because it is only used in that example.

### 3.5 Hausdorff distance and plotting

- **`hausdorff.py`**

  - `HausdorffResult`  
    Dataclass storing:
    - `distance` — Hausdorff distance,
    - `point_a`, `point_b` — the pair of points where the distance is (approximately) attained.
  - `hausdorff_distance(A, B)`  
    Computes symmetric Hausdorff distance between finite point clouds `A` and `B` (NumPy arrays of shape `(N, 2)` and `(M, 2)`).

- **`plotting.py`**

  - `plot_reachable_sets(R_grid, R_oc, hd, title, save_path)`  
    Plots:
    - grid reachable set `R_grid`,
    - boundary `R_oc`,
    - and the Hausdorff segment between `hd.point_a` and `hd.point_b`.  
    Used in the linear system experiment.

### 3.6 Main experiments and benchmark

- **`experiment.py`**

  Contains the high-level orchestration and all figures:

  - `run_linear_example()`  
    Linear system with disk control:
    - grid reachable set (NumPy, grid thinning),
    - Poisson thinning (for comparison of densities),
    - grid reachable set (Torch, grid thinning) with timing and speedup measurement,
    - Pyomo boundary via optimal control in directions $\ell(\varphi)$,
    - boundary via brute-force constant controls,
    - Hausdorff distance:
      - between grid set and Pyomo boundary,
      - between Pyomo and brute-force boundaries,
    - visualisation on two figures.

  - `run_li_markus_example()`  
    Li–Markus example:
    - for several time horizons $T$ builds the grid reachable set (NumPy backend),
    - builds an approximate boundary via brute-force constant controls on the elliptic control set,
    - computes the convex hull of the reachable set and the Hausdorff distance to it,
    - prints a qualitative convexity status ("degenerate", "numerically non-convex", "approximately convex"),
    - draws figures with:
      - grid reachable set,
      - boundary,
      - convex hull,
      - Hausdorff segment.

  - `benchmark_reachability_speed()`  
    Compares runtime of the grid method for NumPy vs Torch (CUDA) on the linear system for several combinations of:
    - `num_controls`,
    - `num_time_steps`.  
    Prints timings, cloud sizes and the measured speedup.

  - `run_experiment()`  
    Convenience function that runs both `run_linear_example()` and `run_li_markus_example()` in sequence.

---

## 4. Installation

### 4.1 Requirements

Python 3.10+ and the following packages:

- `numpy`
- `matplotlib`
- `pyomo`
- `torch`
- `scipy`

Install via:

```bash
pip install numpy matplotlib pyomo torch scipy
```

For the Pyomo optimal control problems a nonlinear solver is required, e.g. **IPOPT**.
In Google Colab, prebuilt IPOPT binaries can be downloaded and added to `PATH`; the example notebook typically uses commands like:

```bash
!pip install -q pyomo
# download and unpack ipopt, then ensure the binary is in PATH
```

Adjust this part to your environment.

### 4.2 Running locally

Clone the repository:

```bash
git clone https://github.com/zv3zdochka/reachable-sets-2d-pyomo-grid-torch.git
```

Install dependencies (optionally add a `requirements.txt`):

```bash
pip install -r requirements.txt
# or install the packages listed above manually
```

Make sure an NLP solver (e.g. `ipopt`) is available to Pyomo.

Run:

```bash
python experiment.py
```

The script will:

1. Run the **linear system experiment**, printing timing and Hausdorff distances and showing two plots (grid vs Pyomo, Pyomo vs brute force).
2. Run the **Li–Markus experiment**, printing the size of the reachable set, status of convexity and displaying a series of plots for different $T$.

### 4.3 Running in Google Colab

Typical Colab workflow:

1. Upload all `*.py` files to the working directory or unzip the project archive.

2. Enable GPU (Runtime → Change runtime type → GPU).

3. Install dependencies:

   ```python
   !pip install numpy matplotlib pyomo torch scipy
   # plus installation of IPOPT or another solver
   ```

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

---

## 5. Purpose and expected outcomes

The repository is intended as a compact but complete implementation of the assignment:

* show **two different constructions** of reachable sets (grid method vs optimal control via Pyomo),
* demonstrate how a **GPU backend** (PyTorch) can significantly accelerate a grid-based method for large configurations,
* illustrate the **difference between convex and non-convex reachable sets**:

  * convex case: linear system with disk control, where Pyomo and grid methods agree;
  * non-convex case: Li–Markus example, where the grid method and convex hull differ noticeably,
* quantify the discrepancy between various approximations via the **Hausdorff distance** and visualise it directly on the plots.

This structure can be reused for other planar controlled systems by replacing the dynamics and the control-set generators.

---

## 6. Example results

### 6.1 Li–Markus non-convex reachable set at T = 0.60

![Li-Markus reachable set, T=0.60](docs/li_markus_t060.png)

At the early time horizon $T = 0.60$, the Li–Markus system exhibits a **clearly non-convex reachable set**. The figure shows:

- **Grid reachable set** (blue points) — computed via the grid method with NumPy backend
- **Boundary** (orange curve) — constructed using constant controls on the elliptic control set
- **Convex hull** (black dashed line) — the convex envelope of the reachable set
- **Hausdorff segment** (dotted line) — connects the two points realizing the Hausdorff distance between the reachable set and its convex hull

The Hausdorff distance of **0.175** quantifies the deviation from convexity. The reachable set has a characteristic narrow "finger" shape, demonstrating that the system's nonlinearity in control produces geometric structures that cannot be captured by convex combinations alone.

### 6.2 Li–Markus reachable set at T = 2.00

![Li-Markus reachable set, T=2.00](docs/li_markus_t200.png)

At the longer time horizon $T = 2.00$, the reachable set grows significantly and maintains its **non-convex character**, now with a more complex crescent or "moon" shape. Key observations:

- The reachable set (blue region) has a large interior region that is **not reachable**, creating a pronounced non-convex cavity
- The **convex hull** (black dashed boundary) significantly overestimates the actual reachable set
- The Hausdorff distance increases to **0.759**, indicating stronger non-convexity as time evolves
- The boundary (orange) closely follows the edge of the grid set, validating both methods

This example illustrates why optimal control methods based on support functions (which would only recover the convex hull) are insufficient for non-convex reachability analysis, and highlights the necessity of grid-based or level-set approaches for such systems.


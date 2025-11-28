# plotting.py
"""
plotting.py

Visualization utilities: plotting the reachable set from the grid method and
the boundary from the optimal control method on a single figure, as well as
visualizing the Hausdorff distance between them.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from hausdorff import HausdorffResult


def plot_reachable_sets(
    R_grid: np.ndarray,
    R_oc: np.ndarray,
    hd: HausdorffResult,
    title: str = "Reachable sets and Hausdorff distance",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the grid-based reachable set and the OC-based boundary together, and
    visualize the Hausdorff distance as a segment between two points.

    Parameters
    ----------
    R_grid : np.ndarray
        Point cloud from the grid method, shape (N, 2).
    R_oc : np.ndarray
        Boundary points from optimal control, shape (M, 2).
    hd : HausdorffResult
        Result of Hausdorff distance computation.
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path. Otherwise, just show it.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")

    if R_grid.size > 0:
        ax.scatter(R_grid[:, 0], R_grid[:, 1], s=10, alpha=0.4, label="Grid reachable set")
    if R_oc.size > 0:
        # Close the boundary for nicer visualization
        R_oc_closed = np.vstack([R_oc, R_oc[0]])
        ax.plot(R_oc_closed[:, 0], R_oc_closed[:, 1], linewidth=2.0, label="OC boundary")

    # Hausdorff pair
    pa = hd.point_a
    pb = hd.point_b
    ax.scatter([pa[0]], [pa[1]], color="red", s=60, label="Hausdorff point A")
    ax.scatter([pb[0]], [pb[1]], color="green", s=60, label="Hausdorff point B")
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], linestyle="--", color="black",
            label=f"Hausdorff segment (d={hd.distance:.3f})")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()

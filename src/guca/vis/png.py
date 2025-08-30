from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable, Tuple

from guca.core.graph import GUMGraph


def _circular_layout(node_ids: Iterable[int]) -> dict[int, Tuple[float, float]]:
    ids = list(node_ids)
    n = max(len(ids), 1)
    pos: dict[int, Tuple[float, float]] = {}
    for i, nid in enumerate(ids):
        theta = (2.0 * math.pi * i) / n
        pos[nid] = (math.cos(theta), math.sin(theta))
    return pos


def save_png(graph: GUMGraph, out_path: Path, *, dpi: int = 150) -> float:
    """
    Render a GUMGraph to a PNG file at out_path. Returns plotting time (seconds).
    Layout: simple circular. Dependencies: matplotlib only.
    """
    import matplotlib.pyplot as plt  # local import to keep import-time light

    t0 = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect nodes/edges up front (graph is a light container).
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    # Empty graph -> write a blank placeholder for pipeline consistency.
    if not nodes:
        fig = plt.figure(figsize=(4, 4), dpi=dpi)
        plt.axis("off")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return time.perf_counter() - t0

    ids = [n.id for n in nodes]
    pos = _circular_layout(ids)

    # Draw
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_aspect("equal")

    # Edges first (thin lines).
    for a, b in edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], linewidth=0.6, alpha=0.35)

    # Nodes on top (small dots).
    xs = [pos[nid][0] for nid in ids]
    ys = [pos[nid][1] for nid in ids]
    ax.scatter(xs, ys, s=20, alpha=0.9)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return time.perf_counter() - t0

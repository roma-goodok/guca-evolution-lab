# src/guca/utils/lattices.py
from __future__ import annotations

from math import ceil, floor, sqrt
from typing import Literal, Tuple
import networkx as nx
from networkx.generators.lattice import triangular_lattice_graph, hexagonal_lattice_graph


def make_tri_patch(kind: Literal["strip", "block", "compact"] = "block", faces: int = 4) -> nx.Graph:
    """
    Build a small triangular-lattice patch with positions.

    Notes:
      * We do NOT try to hit 'faces' exactly (that requires bespoke trimming),
        but we ensure growth and correct 3-face detection under geometry.
      * For small 'faces', we avoid the old min=2 clamp so tiny patches vary.
    """
    faces = max(1, int(faces))
    if kind == "strip":
        # skinny strip that grows with 'faces'
        # n increases, m stays 2 to avoid degeneracy
        m = 2
        n = max(2, faces + 1)
    else:
        # aim for ~faces triangles; triangular_lattice_graph yields ~ 2*m*n small triangles
        cells = max(1, faces)                  # treat 'faces' as target cells directly
        m = max(1, floor(sqrt(cells / 2.0)))   # allow m=1
        if m == 0:
            m = 1
        n = max(1, ceil(cells / max(1, m)))

    G = triangular_lattice_graph(m, n)  # has 'pos' attribute
    # Ensure 'pos' exists (triangular_lattice_graph sets it; keep as-is)
    return G


def make_quad_patch(rows: int = 2, cols: int = 2) -> nx.Graph:
    """
    Square grid patch with rows*cols quadrilateral faces; includes 'pos'.
    Nodes are kept as (i,j) tuples so positions are exact.
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    G = nx.grid_2d_graph(rows + 1, cols + 1)
    pos = {(i, j): (float(i), float(j)) for i, j in G.nodes()}
    nx.set_node_attributes(G, pos, "pos")
    return G


def make_hex_patch(kind: Literal["strip", "block", "compact"] = "block", faces: int = 4) -> nx.Graph:
    """
    Hexagonal-lattice (honeycomb) patch with positions.

    We choose (m,n) so that the number of hex cells is exactly 'faces' when possible:
      - "strip": (1, faces)
      - "block"/"compact": factor pair (m,n) with m*n == faces and |m-n| minimal.
        For prime 'faces', this becomes a strip automatically.
    """
    faces = max(1, int(faces))
    if kind == "strip":
        m, n = 1, faces
    else:
        m, n = _best_factor_pair(faces)

    G = hexagonal_lattice_graph(m, n)  # has 'pos' attribute
    # Keep whatever node labels the generator uses; preserve 'pos'
    return G


def _best_factor_pair(k: int) -> Tuple[int, int]:
    """Return (m, n) with m*n == k and |m-n| minimal; if k is prime, return (1, k)."""
    m = int(floor(sqrt(k)))
    while m > 1 and k % m != 0:
        m -= 1
    if k % m == 0:
        return m, k // m
    return 1, k

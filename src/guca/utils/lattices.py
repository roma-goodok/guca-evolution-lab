# src/guca/utils/lattices.py
from __future__ import annotations

from math import ceil, sqrt
from typing import Literal
import networkx as nx
from networkx.generators.lattice import (
    triangular_lattice_graph,
    hexagonal_lattice_graph,
)


def make_tri_patch(kind: Literal["strip", "block", "compact"] = "block", faces: int = 4) -> nx.Graph:
    """
    Build a small triangular-lattice patch.

    We parameterize by (m, n) nodes in the triangular lattice generator. The exact
    number of triangular faces depends on m,n; we aim for ~faces and prefer monotone growth.

    kind:
      - "strip":  1 x N (slim)   → emphasizes shell penalty
      - "block":  ~sqrt target   → compact patch (default)
      - "compact": alias of "block" for now
    """
    faces = max(1, int(faces))
    if kind == "strip":
        m, n = 2, max(2, faces + 1)  # skinny but not degenerate
    else:
        approx_cells = max(1, faces // 2)  # triangular lattice has ~2*m*n small triangles
        m = max(2, int(sqrt(approx_cells)))
        n = max(2, int(ceil(approx_cells / m)))
    # triangular_lattice_graph(m, n) makes a connected patch with interior deg≈6
    G = triangular_lattice_graph(m, n)
    return G


def make_quad_patch(rows: int = 2, cols: int = 2) -> nx.Graph:
    """Standard orthogonal grid patch; rows*cols quadrilateral faces."""
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    G = nx.grid_2d_graph(rows + 1, cols + 1)  # nodes are (i,j)
    # add horizontal/vertical edges already exist; this is fine for quads
    return nx.convert_node_labels_to_integers(G)  # make ids compact ints for consistency


def make_hex_patch(kind: Literal["strip", "block", "compact"] = "block", faces: int = 4) -> nx.Graph:
    """
    Build a small hexagonal-lattice (honeycomb) patch.

    We use NetworkX's hexagonal_lattice_graph which yields interior deg≈3 and
    true 6-cycle faces in standard embeddings.

    kind:
      - "strip":  1 x N (chain of hex cells)
      - "block":  ~sqrt target
      - "compact": alias of "block"
    """
    faces = max(1, int(faces))
    if kind == "strip":
        m, n = 1, max(1, faces)          # a single-row strip of hex cells
    else:
        m = max(1, int(sqrt(faces)))
        n = max(1, int(ceil(faces / m)))
    G = hexagonal_lattice_graph(m, n)
    # Compact integer labels; keep pos if present for drawing
    pos = nx.get_node_attributes(G, "pos")
    G = nx.convert_node_labels_to_integers(G)
    if pos:
        # remap positions to new integer labels (order preserved)
        new_pos = {}
        for idx, node in enumerate(G.nodes()):
            # nodes() yields in insertion order after conversion; map by index
            new_pos[node] = list(pos.values())[idx]
        nx.set_node_attributes(G, new_pos, "pos")
    return G

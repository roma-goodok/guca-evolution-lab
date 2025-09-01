# src/guca/utils/lattices.py
from __future__ import annotations

from math import ceil, floor, sqrt
from typing import Literal
import networkx as nx


def make_tri_patch(kind: Literal["strip", "block", "compact"] = "block", faces: int = 4) -> nx.Graph:
    """
    Build a small triangular-lattice *patch* with approximately `faces` triangular faces.

    Strategy:
      - faces == 1: single triangle (3-cycle)
      - otherwise: start from an (r x c) rect grid and add alternating diagonals,
        which yields ~ 2*r*c triangular faces. We choose r,c to hit ~faces/2.
      - kind controls rough aspect ratio; for now:
          * "strip": r=1,  c=ceil(faces/2)
          * "block": r≈c≈sqrt(faces/2)
          * "compact": same as block (placeholder, can be tuned later)
    """
    faces = max(1, int(faces))

    if faces == 1:
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        return G

    if kind == "strip":
        r = 1
        c = max(1, ceil(faces / 2))
    else:
        cells = max(1, ceil(faces / 2))
        r = max(1, floor(sqrt(cells)))
        c = max(1, ceil(cells / r))

    # Build rectangular grid nodes (r+1) x (c+1) and add orthogonal edges
    G = nx.Graph()
    for i in range(r + 1):
        for j in range(c + 1):
            G.add_node((i, j))

    for i in range(r + 1):
        for j in range(c + 1):
            if j + 1 <= c:
                G.add_edge((i, j), (i, j + 1))
            if i + 1 <= r:
                G.add_edge((i, j), (i + 1, j))

    # Add one diagonal per cell, alternating to keep planarity and form triangles
    for i in range(r):
        for j in range(c):
            if (i + j) % 2 == 0:
                G.add_edge((i, j), (i + 1, j + 1))
            else:
                G.add_edge((i + 1, j), (i, j + 1))

    return G


def make_quad_patch(rows: int = 2, cols: int = 2) -> nx.Graph:
    """
    Build a small square-grid patch with `rows*cols` quadrilateral faces.
    The graph is the (rows+1) x (cols+1) orthogonal grid.
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    G = nx.Graph()

    for i in range(rows + 1):
        for j in range(cols + 1):
            G.add_node((i, j))

    for i in range(rows + 1):
        for j in range(cols + 1):
            if j + 1 <= cols:
                G.add_edge((i, j), (i, j + 1))
            if i + 1 <= rows:
                G.add_edge((i, j), (i + 1, j))
    return G


def make_hex_patch(kind: Literal["strip", "block", "compact"] = "block", faces: int = 4) -> nx.Graph:
    """
    Build a small hexagonal-lattice (honeycomb) *skeleton* with approximately `faces` hexagonal faces.

    Implementation (simple & robust for <= ~10 faces):
      - "strip": chain N hexagons sharing an edge; produces exactly N hex faces.
      - "block"/"compact": for now, fallback to "strip" (keeps code minimal for Week 3).
    """
    faces = max(1, int(faces))
    if kind != "strip":
        kind = "strip"

    # Helper to add a single hex cycle using given 6 labels (in order)
    def add_hex(G, labels):
        for k in range(6):
            a = labels[k]
            b = labels[(k + 1) % 6]
            G.add_edge(a, b)

    G = nx.Graph()

    if faces == 1:
        lbl = [(0, k) for k in range(6)]
        G.add_nodes_from(lbl)
        add_hex(G, lbl)
        return G

    # Build a strip: each new hex shares one edge with the previous one.
    # For hex i>0, share the edge between vertices 1-2 with previous hex,
    # and create 4 new vertices for the remaining corners.
    # Label scheme: (i, k) for hex index i and corner index k (0..5), but
    # reuse (i-1,1) and (i-1,2) as the shared edge.
    # Orientation is consistent so faces remain planar.
    # Start with first hex
    prev_labels = [(0, k) for k in range(6)]
    G.add_nodes_from(prev_labels)
    add_hex(G, prev_labels)

    for i in range(1, faces):
        shared = [prev_labels[1], prev_labels[2]]
        new_vs = [(i, 0), (i, 3), (i, 4), (i, 5)]
        # Define order of the new hex's 6 corners so that it shares edge (1-2) with previous
        # We map: 0->prev[2], 1->prev[1], 2->(i,0), 3->(i,3), 4->(i,4), 5->(i,5)
        curr = [prev_labels[2], prev_labels[1], new_vs[0], new_vs[1], new_vs[2], new_vs[3]]
        G.add_nodes_from(new_vs)
        add_hex(G, curr)
        prev_labels = curr

    return G

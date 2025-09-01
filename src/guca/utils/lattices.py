# src/guca/utils/lattices.py
from __future__ import annotations

from typing import Literal, Optional
import networkx as nx


def make_tri_patch(kind: Literal["strip", "block", "compact"] = "block", faces: int = 4) -> nx.Graph:
    """
    Build a small triangular-lattice patch with approximately `faces` triangular faces.
    kind:
      - "strip": thin strip (n x 1) style
      - "block": roughly square-ish patch (default)
      - "compact": compact hex-like outline filled by triangles
    """
    # Implementation incoming in the next step.
    return nx.Graph()


def make_quad_patch(rows: int = 2, cols: int = 2) -> nx.Graph:
    """
    Build a small square grid patch with rows*cols quadrilateral faces.
    """
    # Implementation incoming in the next step.
    return nx.Graph()


def make_hex_patch(kind: Literal["strip", "block", "compact"] = "block", faces: int = 4) -> nx.Graph:
    """
    Build a small hexagonal-lattice (honeycomb) patch with approximately `faces` hex faces.
    kind:
      - "strip": thin strip of hexes
      - "block": roughly rectangular block
      - "compact": near-hexagonal compact patch
    """
    # Implementation incoming in the next step.
    return nx.Graph()

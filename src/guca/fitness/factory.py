# src/guca/fitness/factory.py
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from guca.fitness.by_sample import BySample, BySampleWeights
from guca.utils.lattices import make_tri_patch, make_quad_patch, make_hex_patch

@dataclass
class BySampleFromTarget(BySample):
    """
    Hydra-friendly BySample that builds its reference graph from a tiny lattice target.
    """
    def __init__(
        self,
        target_family: str = "tri",
        faces: int = 6,
        tri_kind: str = "block",
        hex_kind: str = "block",
        rows: int = 2,
        cols: int = 2,
        weights: Optional[BySampleWeights] = None,
        **kwargs,
    ):
        if target_family == "tri":
            T = make_tri_patch(tri_kind, faces)
        elif target_family == "quad":
            T = make_quad_patch(rows, cols)
        elif target_family == "hex":
            T = make_hex_patch(hex_kind, faces)
        else:
            raise ValueError(f"Unknown target_family: {target_family}")
        super().__init__(reference_graph=T, weights=weights, **kwargs)

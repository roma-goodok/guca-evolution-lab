# src/guca/fitness/meshes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import networkx as nx

from .planar_basic import PlanarBasic, ViabilityResult, EmbeddingInfo


@dataclass
class MeshWeights:
    """Weights for mesh scoring."""
    w_face: float = 1.0
    w_deg: float = 0.6
    w_shell: float = 0.4
    w_nontarget: float = 0.5
    genome_len_bonus: bool = False


class _MeshBase(PlanarBasic):
    """Base mesh heuristic; concrete classes define target face length and target interior degree."""
    target_face_len: int = 0
    target_interior_deg: int = 0

    def __init__(self, *, weights: Optional[MeshWeights] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weights = weights or MeshWeights()

    def score(self, G: nx.Graph, meta: Optional[Dict] = None) -> float:
        """Compute the mesh fitness score for graph G."""
        vr: ViabilityResult = self.viability_filter(G, meta)
        if not vr.viable:
            return vr.base_score

        GG = self.prepare_graph(G)
        emb: EmbeddingInfo = self.compute_embedding_info(GG)
        # --- placeholder: will be replaced with the real scoring in next step ---
        # We return the base score for now so the CLI/tests we add later can import these classes.
        return vr.base_score


class TriangleMesh(_MeshBase):
    """Triangle mesh heuristic: prefer faces of length 3, interior deg ≈ 6."""
    target_face_len = 3
    target_interior_deg = 6


class QuadMesh(_MeshBase):
    """Quad mesh heuristic: prefer faces of length 4, interior deg ≈ 4."""
    target_face_len = 4
    target_interior_deg = 4


class HexMesh(_MeshBase):
    """Hex mesh heuristic: prefer faces of length 6, interior deg ≈ 3."""
    target_face_len = 6
    target_interior_deg = 3

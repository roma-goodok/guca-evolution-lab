# src/guca/fitness/meshes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

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
    """
    Base mesh heuristic; concrete classes define:
      - target_face_len  (3 for triangles, 4 for quads, 6 for hexes)
      - target_interior_deg (6, 4, 3 respectively)
    """
    target_face_len: int = 0
    target_interior_deg: int = 0

    def __init__(self, *, weights: Optional[MeshWeights] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.weights = weights or MeshWeights()

    # ---- public API ---------------------------------------------------------
    def score(self, G: nx.Graph, meta: Optional[Dict] = None) -> float:
        """
        Compute the mesh fitness score for graph G.

        meta can include:
          - steps, max_steps, diverged (for viability)
          - genome_len or genome_length or genome (list) for optional bonus
        """
        vr: ViabilityResult = self.viability_filter(G, meta)
        if not vr.viable:
            return vr.base_score

        GG = self.prepare_graph(G)
        emb: EmbeddingInfo = self.compute_embedding_info(GG)

        # Faces excluding the chosen shell face (exclude by identity, not by set equality).
        faces_wo_shell: List[List] = [f for f in emb.faces if f is not emb.shell]

        # --- Face ratio term: fraction of interior faces matching target length ---
        f_int_total = len(faces_wo_shell)
        f_int_target = sum(1 for f in faces_wo_shell if len(f) == self.target_face_len)
        face_ratio = (f_int_target / f_int_total) if f_int_total > 0 else 0.0

        # --- Interior degree matching (reward is 1 - normalized deviation) ---
        if emb.interior_nodes:
            deviations = [
                abs(GG.degree(v) - self.target_interior_deg) / max(1, self.target_interior_deg)
                for v in emb.interior_nodes
            ]
            deg_pen = min(1.0, sum(deviations) / len(deviations))
            deg_reward = 1.0 - deg_pen
        else:
            # No interior nodes (tiny patches) – treat as neutral/good
            deg_reward = 1.0

        # --- Boundary compactness penalty: ratio of shell vertices to all vertices ---
        shell_pen = (len(emb.shell) / max(1, GG.number_of_nodes()))

        # --- Non-target faces penalty among interior faces ---
        non_target_pen = ((f_int_total - f_int_target) / f_int_total) if f_int_total > 0 else 0.0

        # --- Optional genome-length bonus ---
        gl_bonus = 0.0
        if self.weights.genome_len_bonus:
            gl = _infer_genome_len(meta)
            if gl and gl > 0:
                gl_bonus = 1.0 / gl

        score = (
            vr.base_score
            + self.weights.w_face * face_ratio
            + self.weights.w_deg * deg_reward
            - self.weights.w_shell * shell_pen
            - self.weights.w_nontarget * non_target_pen
            + gl_bonus
        )
        return float(score)


def _infer_genome_len(meta: Optional[Dict]) -> Optional[int]:
    if not meta:
        return None
    if "genome_len" in meta and isinstance(meta["genome_len"], int):
        return meta["genome_len"]
    if "genome_length" in meta and isinstance(meta["genome_length"], int):
        return meta["genome_length"]
    if "genome" in meta and isinstance(meta["genome"], (list, tuple)):
        return len(meta["genome"])
    return None


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

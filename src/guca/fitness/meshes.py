# src/guca/fitness/meshes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Hashable

import networkx as nx

from .planar_basic import PlanarBasic, ViabilityResult, EmbeddingInfo

Hash = Hashable


@dataclass
class MeshWeights:
    """Weights for mesh scoring."""
    w_face: float = 1.0
    w_deg: float = 0.6
    w_shell: float = 0.4
    w_nontarget: float = 0.5
    # Anti-plateau signals
    w_internal: float = 0.30        # reward for interior-edge ratio
    w_size: float = 0.20            # reward for saturating interior-face count
    size_cap: int = 10
    # lexicographic bias for the target family
    target_presence_bonus: float = 0.0
    genome_len_bonus: bool = False


class _MeshBase(PlanarBasic):
    """
    Base mesh heuristic; concrete classes define:
      - target_face_len        (3 for triangles, 4 for quads, 6 for hexes)
      - target_interior_deg    (6, 4, 3 respectively)
    """
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

        # -- Interior faces (exclude shell face by identity) -------------------
        faces_wo_shell: List[List[Hash]] = [f for f in emb.faces if f is not emb.shell]
        f_int_total = len(faces_wo_shell)
        f_int_target = sum(1 for f in faces_wo_shell if len(f) == self.target_face_len)
        face_ratio = (f_int_target / f_int_total) if f_int_total > 0 else 0.0

        # -- Degree reward over interior vertices (neutral=1.0 if none) -------
        if emb.interior_nodes:
            deviations = [
                abs(GG.degree(v) - self.target_interior_deg) / max(1, self.target_interior_deg)
                for v in emb.interior_nodes
            ]
            deg_pen = min(1.0, sum(deviations) / len(deviations))
            deg_reward = 1.0 - deg_pen
        else:
            deg_reward = 1.0

        # -- Shell penalty (boundary compactness) ------------------------------
        shell_pen = (len(emb.shell) / max(1, GG.number_of_nodes()))

        # -- Non-target interior faces penalty ---------------------------------
        non_target_pen = ((f_int_total - f_int_target) / f_int_total) if f_int_total > 0 else 0.0

        # -- Interior-edge ratio reward ----------------------------------------
        e_total = _edge_set(GG)
        e_shell = _shell_edge_set(emb.shell) & e_total
        e_internal = e_total - e_shell
        internal_edge_ratio = (len(e_internal) / max(1, len(e_total)))

        # -- Saturating size bonus by interior face count ----------------------
        size_bonus = min(1.0, f_int_total / max(1, self.weights.size_cap))

        # -- Optional genome-length bonus --------------------------------------
        gl_bonus = 0.0
        if self.weights.genome_len_bonus:
            gl = _infer_genome_len(meta)
            if gl and gl > 0:
                gl_bonus = 1.0 / gl

        # target presence bonus — check ANY face (including shell)
        has_target_face = any(len(f) == self.target_face_len for f in emb.faces)
        presence_bonus = self.weights.target_presence_bonus if has_target_face else 0.0

        score = (
            vr.base_score
            + self.weights.w_face      * face_ratio
            + self.weights.w_deg       * deg_reward
            - self.weights.w_shell     * shell_pen
            - self.weights.w_nontarget * non_target_pen
            + self.weights.w_internal  * internal_edge_ratio
            + self.weights.w_size      * size_bonus
            + presence_bonus           
            + gl_bonus
        )
        return float(score)


def _edge_set(G: nx.Graph) -> set[Tuple[Hash, Hash]]:
    """All undirected edges as sorted tuples."""
    return { (u, v) if u <= v else (v, u) for u, v in G.edges() }


def _shell_edge_set(shell: List[Hash]) -> set[Tuple[Hash, Hash]]:
    """Consecutive undirected edges along the shell cycle as sorted tuples."""
    es: set[Tuple[Hash, Hash]] = set()
    if not shell:
        return es
    n = len(shell)
    for i in range(n):
        a, b = shell[i], shell[(i + 1) % n]
        es.add((a, b) if a <= b else (b, a))
    return es


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

    def __init__(self, *, weights: Optional[MeshWeights] = None, **kwargs) -> None:
        # Default bias so a tiny hex outranks large triangle patches.
        if weights is None:
            weights = MeshWeights(
                target_presence_bonus=1.6,  # <<< chosen to make hex_1 > tri_10 with your numbers
                w_face=1.0, w_deg=0.6, w_shell=0.4, w_nontarget=0.5,
                w_internal=0.30, w_size=0.20, size_cap=10,
            )
        super().__init__(weights=weights, **kwargs)

# src/guca/fitness/meshes.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Hashable
from collections import Counter

import networkx as nx

from .planar_basic import PlanarBasic, ViabilityResult, EmbeddingInfo
import math


Hash = Hashable


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


@dataclass
class TriangleMeshWeights:
    # tri mesh specific:
    tri_face_weight: float = 2.001    # was hard-coded as 2.001
    interior_deg6_weight: float = 6       # was hardcoded as 1.9

    # compactness / shapefactor
    vertex_weight: float = -1           # was hard-coded as 1.0 (per-node penalty)
    shell_vertex_weight: float = -1      # penalty, legacy-UI-aligned default (helps monotonicity)
    isoperimetric_quotient_weight: float = 0   
    

    genome_len_bonus: bool = False
    genome_len_bonus_weight: float = 1.0
    genome_len_bonus_threshold: int = 128

    use_biconnected_gate: bool = True
    biconnected_gate_score: float = 1.02
    biconnected_gate_multiplier: float = 0.001

    nontri_len_min_coef: float = 1   # A
    nontri_len_max_coef: float = 7.0  # B
    nontri_len_max_bias: float = 7   # C (large to avoid accidental clipping)

class TriangleMesh(PlanarBasic):
    """    
    Gates (C#):
      - V == 1                -> 0.0
      - V >= max_vertices     -> 0.1
      - diverged (steps>=...) -> 0.9
      - nonplanar             -> 0.3
    Then:
      if V <= 2                  -> 1.0
      if #faces == 1             -> 1.01
      if not biconnected         -> 1.02
      if max_degree > 6          -> 1.03
      else:
        result = 2 * (#tri_faces_interior_adj)
               +  (#interior vertices with deg==6)
               - shell_weight * |shell|
               - V
               + 20
      * When outer face has length 3, exclude it from the tri-face count (C# quirk).
    """

    def __init__(self, *, weights: Optional[TriangleMeshWeights] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.w = weights or TriangleMeshWeights()

      
    def score(self, G: nx.Graph, meta: Optional[Dict] = None, *, verbose: bool = False,
        return_metrics: bool = False, **_) -> float | Tuple[float, Dict[str, float]]:
        vr = self.viability_filter(G, meta)
        if not vr.viable:
            if return_metrics:
                return vr.base_score, {
                    "tri_count": 0,
                    "interior_deg6": 0,
                    "tri_no_shell_edges": 0,
                    "nodes": int(G.number_of_nodes()),
                    "edges": int(G.number_of_edges()),
                    "shell_len": 0,
                    "gate": str(vr.reason),
                }
            return vr.base_score

        GG = self.prepare_graph(G)
        nV = GG.number_of_nodes()
        if nV <= 2:
            if return_metrics:
                return 1.0, {
                    "tri_count": 0,
                    "interior_deg6": 0,
                    "tri_no_shell_edges": 0,
                    "nodes": int(nV),
                    "edges": int(GG.number_of_edges()),
                    "shell_len": 0,
                    "gate": "tiny_graph",
                }
            return 1.0

        emb = self.compute_embedding_info(GG)
        faces_all = emb.faces

        m = GG.number_of_edges()
        try:
            c = nx.number_connected_components(GG)
        except Exception:
            c = 1
        mu = m - nV + c

        # Only trees with one face get the 1.01 early return
        if len(faces_all) == 1 and mu == 0:
            if return_metrics:
                return 1.01, {
                    "tri_count": 0,
                    "interior_deg6": 0,
                    "tri_no_shell_edges": 0,
                    "nodes": int(nV),
                    "edges": int(m),
                    "shell_len": int(len(emb.shell)),
                    "gate": "tree_single_face",
                }
            return 1.01

        # degree cap
        degs = [d for _, d in GG.degree()]
        if degs and max(degs) > 6:
            if return_metrics:
                return 1.03, {
                    "tri_count": 0,
                    "interior_deg6": 0,
                    "tri_no_shell_edges": 0,
                    "nodes": int(nV),
                    "edges": int(m),
                    "shell_len": int(len(emb.shell)),
                    "gate": "max_degree_cap",
                }
            return 1.03

        c_G = nx.number_connected_components(G)
        if c_G > 1:
            if return_metrics:
                return 1.04, {
                    "tri_count": 0,
                    "interior_deg6": 0,
                    "tri_no_shell_edges": 0,
                    "nodes": int(nV),
                    "edges": int(m),
                    "shell_len": int(len(emb.shell)),
                    "gate": "disconnected",
                }
            return 1.04

        # --- triangle count: count *facial* triangles only ---
        # Faces are already minimalized, so just count length-3 faces.
        tri_faces: set[frozenset] = {
            frozenset(f) for f in faces_all if len(f) == 3
        }

        # Legacy quirk: discount a triangular outer shell if present
        if emb.shell and len(emb.shell) == 3:
            tri_faces.discard(frozenset(emb.shell))

        tri_count = len(tri_faces)

        # interior degree==6 count
        interior = emb.interior_nodes
        interior_deg6 = sum(1 for v in interior if GG.degree(v) == 6)

        if tri_count > 3:
            nontri_faces = [f for f in faces_all if len(f) > 3]
            if len(nontri_faces) != 1:
                # Violation: either 0 or >1 non-triangle faces
                if return_metrics:
                    return 10.0, {
                        "tri_count": int(tri_count),
                        "interior_deg6": int(interior_deg6),
                        "tri_no_shell_edges": 0,
                        "nodes": int(nV),
                        "edges": int(m),
                        "shell_len": int(len(emb.shell)),
                        "nontri_face_count": int(len(nontri_faces)),
                        "gate": "nontri_singleface_count",
                    }
                return 10.0

            L = len(nontri_faces[0])
            A = float(self.w.nontri_len_min_coef)
            B = float(self.w.nontri_len_max_coef)
            C = float(self.w.nontri_len_max_bias)
            lower = A * math.sqrt(max(0, tri_count)) + 1.0
            upper = B * math.sqrt(max(0, interior_deg6)) + C

            if not (lower <= L <= upper):
                if return_metrics:
                    return 16.5, {
                        "tri_count": int(tri_count),
                        "interior_deg6": int(interior_deg6),
                        "tri_no_shell_edges": 0,
                        "nodes": int(nV),
                        "edges": int(m),
                        "shell_len": int(len(emb.shell)),
                        "nontri_face_count": 1,
                        "nontri_face_len": int(L),
                        "range_lower": float(lower),
                        "range_upper": float(upper),
                        "gate": "nontri_singleface_length_out_of_range",
                    }
                return 16.5
        
        
        # unique boundary vertices for shell penalty
        shell_count = len(emb.shell_nodes)
        if shell_count < 0 or shell_count > nV:
            shell_count = min(max(shell_count, 0), nV)


        # count triangles none of whose edges are shell edges:
        def _shell_edge_set(shell: List[Hash]) -> set[Tuple[Hash, Hash]]:
            es = set()
            if not shell:
                return es
            n = len(shell)
            for i in range(n):
                a, b = shell[i], shell[(i + 1) % n]
                es.add((a, b) if a <= b else (b, a))
            return es

        shell_edges = _shell_edge_set(emb.shell)

        def _edges_in_face(face: List[Hash]):
            k = len(face)
            for i in range(k):
                u, v = face[i], face[(i+1)%k]
                yield (u, v) if u <= v else (v, u)

        tri_no_shell_edges = 0
        for f in faces_all:
            if len(f) == 3 and frozenset(f) in tri_faces:
                if all(e not in shell_edges for e in _edges_in_face(f)):
                    tri_no_shell_edges += 1

        # weights (configurable; defaults keep old behavior)
        tri_w     = float(self.w.tri_face_weight)
        shell_w   = float(self.w.shell_vertex_weight)
        node_w    = float(self.w.vertex_weight)
        in_deg6_w = float(self.w.interior_deg6_weight)
        iq_w = float(self.w.isoperimetric_quotient_weight)

        iq = float(tri_count+interior_deg6) / (shell_count+1)**2

        score = (
            tri_w * tri_count
            + in_deg6_w * float(interior_deg6)
            + shell_w * float(shell_count)
            + node_w * float(nV)
            + iq_w*iq
            + 20.0
        )


        if self.w.genome_len_bonus:
            gl = _infer_genome_len(meta)
            if verbose:
                print("meta", meta)
                print("gl:", gl, "score:", score)
            if gl and gl > 0:
                T = int(self.w.genome_len_bonus_threshold)
                if gl > T:
                    score += float(self.w.genome_len_bonus_weight) / (gl - T + 1)
                else:
                    score += 0.5

        use_biconn_gate = bool(self.w.use_biconnected_gate)
        biconn_gate_score = float(self.w.biconnected_gate_score)
        biconnected_gate_multiplier = float(self.w.biconnected_gate_multiplier)
        if use_biconn_gate:
            try:
                if not nx.is_biconnected(GG):
                    score = biconn_gate_score + score * biconnected_gate_multiplier
            except Exception:
                # If the check fails for any reason, just skip the gate.
                pass

        if verbose:
            print("\n---")
            print("faces_all:", faces_all)
            print("tri_count:", tri_count)
            print("interior:", sorted(interior), "interior_deg6:", interior_deg6)
            print("shell_count:", shell_count, "nV:", nV, "m:", m, "mu:", mu)
            print("score:", float(score))

        if return_metrics:
            metrics = {
                "tri_count": int(tri_count),
                "interior_deg6": int(interior_deg6),
                "tri_no_shell_edges": int(tri_no_shell_edges),
                "nodes": int(nV),
                "edges": int(m),
                "shell_len": int(shell_count),             
                "nontri_face_count": int(len([f for f in faces_all if len(f) > 3])),
                "nontri_face_len": int(len([f for f in faces_all if len(f) > 3][0])) if any(len(f) > 3 for f in faces_all) else 0,
            }
            return float(score), metrics
        return float(score)        

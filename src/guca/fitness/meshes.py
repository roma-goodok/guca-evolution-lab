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

# --- add in this file: a shared metrics helper for meshes ---

def _common_face_and_degree_metrics(GG: nx.Graph, emb: EmbeddingInfo) -> Dict[str, int]:
    """Common, family-agnostic metrics derived from the embedding."""
    faces_all = emb.faces
    shell = emb.shell or []

    def _is_shell_face(face: List[Hash]) -> bool:
        # Robust equality: same length and same node-set as shell
        return bool(shell) and (len(face) == len(shell)) and (set(face) == set(shell))

    # EXCLUDE the shell from interior faces; if there is only one face,
    # treat it as an interior face for counting (simple cycle case).
    faces_interior = [f for f in faces_all if not _is_shell_face(f)]
    if not faces_interior and len(faces_all) == 1:
        faces_interior = [faces_all[0]]

    tri_count  = sum(1 for f in faces_interior if len(f) == 3)
    quad_count = sum(1 for f in faces_interior if len(f) == 4)
    hex_count  = sum(1 for f in faces_interior if len(f) == 6)

    interior_nodes = emb.interior_nodes
    interior_deg3 = sum(1 for v in interior_nodes if GG.degree(v) == 3)
    interior_deg4 = sum(1 for v in interior_nodes if GG.degree(v) == 4)
    interior_deg6 = sum(1 for v in interior_nodes if GG.degree(v) == 6)

    return {
        "nodes": GG.number_of_nodes(),
        "edges": GG.number_of_edges(),
        "shell_len": len(shell),
        "faces_total": len(faces_all),
        "faces_interior": len(faces_interior),
        "tri_count": tri_count,
        "quad_count": quad_count,
        "hex_count": hex_count,
        "interior_deg3": interior_deg3,
        "interior_deg4": interior_deg4,
        "interior_deg6": interior_deg6,
    }





@dataclass
class QuadMeshWeights:
    # primary signals
    quad_face_weight: float = 4.001
    interior_deg4_weight: float = 6.0

    # gentle compactness/size shaping (same signs as TriangleMesh)
    vertex_weight: float = -1.0
    shell_vertex_weight: float = 0.0
    isoperimetric_quotient_weight: float = 0.0

    # soft penalties (NOT hard gates)
    # penalize some non-target faces but do not forbid
    w_forbidden_faces: Dict[int, float] = field(default_factory=lambda: {3: 4, 5: 4})

    # faces longer than this allowed max get a linear penalty
    # allowed_max_len = coef * sqrt(quad_count) + bias
    max_face_len_coef: float = 6.0
    max_face_len_bias: float = 4.0
    long_face_penalty_weight: float = 0.6

    # optional genome-length bonus (consistency with Triangle)
    genome_len_bonus: bool = False
    genome_len_bonus_weight: float = 1.0
    genome_len_bonus_threshold: int = 128

    wh_mean_weight: float = 100

    # gates (TriangleMesh parity)
    use_biconnected_gate: bool = True
    biconnected_gate_score: float = 1.02
    biconnected_gate_multiplier: float = 0.001

    # degree cap (Triangle Mesh uses 6; keep same for parity)
    max_degree_cap: int = 6


# --- NEW: add QuadMesh scorer class ---

class QuadMesh(PlanarBasic):
    """
    Quad-optimized scorer: rewards interior quads and interior degree≈4.
    Non-target faces (triangles, pentagons) are penalized softly (not forbidden).
    Faces longer than an allowed max (depending on quad_count) are softly penalized.
    """
    def __init__(self, *, weights: Optional[QuadMeshWeights] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.w = weights or QuadMeshWeights()

    def score(self, G: nx.Graph, meta: Optional[Dict] = None, *,
              return_metrics: bool = False, verbose: bool = False, **_) -> float | Tuple[float, Dict[str, float]]:
        vr = self.viability_filter(G, meta)
        if not vr.viable:
            if return_metrics:
                return vr.base_score, {
                    "nodes": int(G.number_of_nodes()),
                    "edges": int(G.number_of_edges()),
                    "shell_len": 0,
                    "faces_total": 0,
                    "faces_interior": 0,
                    "tri_count": 0, "quad_count": 0, "hex_count": 0,
                    "interior_deg3": 0, "interior_deg4": 0, "interior_deg6": 0,
                }
            return vr.base_score

        GG = self.prepare_graph(G)
        emb = self.compute_embedding_info(GG)
        faces_all = emb.faces
        shell_nodes = emb.shell_nodes

        
        # common metrics
        cm = _common_face_and_degree_metrics(GG, emb)
        nV = cm["nodes"]
        quad_count = cm["quad_count"]
        interior_deg4 = cm["interior_deg4"]

        m  = cm["edges"]
        # components of the working graph for mu; keep safe fallback
        try:
            c = nx.number_connected_components(GG)
        except Exception:
            c = 1
        mu = m - nV + c  # 0 for forests/trees

        # tiny graphs → 1.0 (same plateau)
        if nV <= 2:
            if return_metrics:
                return 1.0, cm
            return 1.0

        # tree/forest plateau → 1.01
        if mu == 0:
            if return_metrics:
                return 1.01 + interior_deg4 * 0.001, cm
            return 1.01 + interior_deg4 * 0.001

        # degree cap (keep parity with TriangleMesh threshold)
        degs = [d for _, d in GG.degree()]
        if degs and max(degs) > int(self.w.max_degree_cap):
            if return_metrics:
                return 1.03, cm
            return 1.03

        # disconnected penalty (strictly below cyclic band)
        c_G = nx.number_connected_components(G)
        if c_G > 1:
            if return_metrics:
                return 1.04, cm
            return 1.04

        

        # --- STRICT shell lower-bound gate (requested) ----------------------
        # require: shell_len >= 4 * sqrt(#quad_faces); otherwise force 8.0
        if quad_count > 0:            
            required_shell = 4.0 * math.sqrt(float(quad_count))
            if cm["shell_len"] < required_shell:
                if return_metrics:
                    mx = dict(cm)
                    mx["wh_mean"] = 0.0
                    mx["gate"] = "shell_lower_bound"
                    mx["required_shell_min"] = float(required_shell)
                    return 3.5, mx
                return 3.5


        # forbidden faces soft penalty (interior only)
        faces_interior = [f for f in faces_all if f is not emb.shell]
        len_hist = Counter(len(f) for f in faces_interior)
        forbidden_pen = 0.0
        for k, w in (self.w.w_forbidden_faces or {}).items():
            forbidden_pen += float(w) * float(len_hist.get(int(k), 0))

        # long-face soft penalty (use quad_count to scale the allowed max)
        allowed_max = float(self.w.max_face_len_coef) * math.sqrt(max(0, quad_count)) + float(self.w.max_face_len_bias)
        over = sum(max(0.0, float(len(f)) - allowed_max) for f in faces_all if len(f) > 0)
        longface_pen = float(self.w.long_face_penalty_weight) * over

        
        # --- mean WH over quad faces -----------------------------------
        wh_mean = self._mean_wh_quads(GG, emb)

        # simple IQ-like compactness proxy (optional)
        iq_w = float(self.w.isoperimetric_quotient_weight)
        iq = float(quad_count + interior_deg4) / (cm["shell_len"] + 1) ** 2 if iq_w != 0 else 0.0

        score = (
            vr.base_score
            + float(self.w.quad_face_weight) * float(quad_count)
            + float(self.w.interior_deg4_weight) * float(interior_deg4)
            + float(self.w.shell_vertex_weight) * float(len(shell_nodes))
            + float(self.w.vertex_weight) * float(nV)
            + iq_w * iq
            - forbidden_pen
            - longface_pen
            + float(self.w.wh_mean_weight) * float(wh_mean)  
            + 20.0
        )

        # optional biconnected gate (compresses non-biconnected scores toward baseline)
        if bool(self.w.use_biconnected_gate):
            try:
                if not nx.is_biconnected(GG):
                    score = float(self.w.biconnected_gate_score) + score * float(self.w.biconnected_gate_multiplier)
            except Exception:
                pass

        


        # optional genome-length bonus (same semantics as Triangle)
        if self.w.genome_len_bonus:
            gl = _infer_genome_len(meta)
            if gl and gl > 0:
                T = int(self.w.genome_len_bonus_threshold)
                if gl > T:
                    score += float(self.w.genome_len_bonus_weight) / (gl - T + 1)
                else:
                    score += 0.5
        
        if return_metrics:
            # expose common metrics + penalties for analysis
            mx = dict(cm)
            mx.update({
                "forbidden_penalty": float(forbidden_pen),
                "longface_penalty": float(longface_pen),
                "allowed_max_face_len": float(allowed_max),
                "wh_mean": float(wh_mean),
            })
            return float(score), mx
        return float(score)

    def _mean_wh_quads(self, G: nx.Graph, emb: EmbeddingInfo) -> float:
        """
        Mean oriented dual run-length product
        Oriented dual-graph geodesic extents (two principal, opposite-edge directions)

        For each interior quad face f = [v0,v1,v2,v3], define edges e_i=(v_i,v_{i+1})
        and walk on the dual along two axes:
          - axis A: enter via edge index 0 (and back via 2), always exit opposite
          - axis B: enter via 1 (and back via 3)
        Count steps in both directions without revisiting faces; stop on shell edges
        (edge used by exactly one face) or when the neighbor is not a quad.
        WH(f) = width(f) * height(f); return mean over interior quads.
        """
        faces = emb.faces
        if not faces:
            return 0.0

        # Build per-face edge lists and a map edge->faces
        def _edges_of(face):
            k = len(face)
            return [ (face[i], face[(i+1)%k]) if face[i] <= face[(i+1)%k] else (face[(i+1)%k], face[i])
                     for i in range(k) ]

        all_edges_to_faces: dict[tuple, list[int]] = {}
        face_edges: list[list[tuple]] = []
        for idx, f in enumerate(faces):
            es = _edges_of(f)
            face_edges.append(es)
            for e in es:
                all_edges_to_faces.setdefault(e, []).append(idx)

        # Shell edges: appear only once among minimal faces
        shell_edges = {e for e, owners in all_edges_to_faces.items() if len(owners) == 1}

        # Select interior quad faces (exclude the outer shell face by identity)
        quads: list[int] = [i for i,f in enumerate(faces) if len(f) == 4 and f is not emb.shell]
        if not quads:
            return 0.0

        # Fast local: neighbor face via this face's edge index i, or None
        def _neighbor(face_idx: int, edge_idx: int) -> tuple[int|None, int|None]:
            e = face_edges[face_idx][edge_idx]
            if e in shell_edges:
                return None, None
            owners = all_edges_to_faces.get(e, [])
            if len(owners) != 2:
                return None, None
            nb = owners[0] if owners[1] == face_idx else owners[1]
            # find index of shared edge inside neighbor
            try:
                nb_i = face_edges[nb].index(e)
            except ValueError:
                nb_i = None
            return nb, nb_i

        def _axis_extent(fid: int, enter_idx_a: int, enter_idx_b: int) -> tuple[int,int]:
            """Return (width, height) for face fid. Count the starting face, too."""
            def _walk_bidir(start_enter_idx: int) -> int:
                visited = {fid}
                steps = 0
                # forward
                cur, ent = fid, start_enter_idx
                while True:
                    exit_idx = (ent + 2) % 4
                    nb, nb_ent = _neighbor(cur, exit_idx)
                    if nb is None or nb_ent is None: break
                    if len(faces[nb]) != 4: break
                    if nb in visited: break
                    visited.add(nb); steps += 1
                    cur, ent = nb, nb_ent
                # backward
                cur, ent = fid, (start_enter_idx + 2) % 4
                while True:
                    exit_idx = (ent + 2) % 4
                    nb, nb_ent = _neighbor(cur, exit_idx)
                    if nb is None or nb_ent is None: break
                    if len(faces[nb]) != 4: break
                    if nb in visited: break
                    visited.add(nb); steps += 1
                    cur, ent = nb, nb_ent
                # IMPORTANT: include the starting face itself
                return 1 + steps

            width  = _walk_bidir(0)  # axis 0↔2
            height = _walk_bidir(1)  # axis 1↔3
            return width, height

        total = 0.0
        for fid in quads:
            w,h = _axis_extent(fid, 0, 1)
            total += float(w * h)
        return total / max(1, len(quads))


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

        # --- EDIT: extend TriangleMesh return_metrics with common metrics (tail of function) ---

        if return_metrics:
            cm = _common_face_and_degree_metrics(GG, emb)  # NEW: add general metrics
            metrics = {
                # legacy
                "tri_count": int(tri_count),
                "interior_deg6": int(interior_deg6),
                "tri_no_shell_edges": int(tri_no_shell_edges),
                "nodes": int(nV),
                "edges": int(m),
                "shell_len": int(shell_count),
                "nontri_face_count": int(len([f for f in faces_all if len(f) > 3])),
                "nontri_face_len": int(len([f for f in faces_all if len(f) > 3][0])) if any(len(f) > 3 for f in faces_all) else 0,
            }
            # merge common metrics (tri/quad/hex counts, degrees, totals)
            metrics.update(cm)
            return float(score), metrics

        return float(score)        

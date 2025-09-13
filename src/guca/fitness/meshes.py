# src/guca/fitness/meshes.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Hashable
from collections import Counter

import networkx as nx

from .planar_basic import PlanarBasic, ViabilityResult, EmbeddingInfo

Hash = Hashable


@dataclass
class MeshWeights:
    """Weights for mesh scoring."""
    w_face: float = 1.0
    w_deg: float = 1.0
    w_shell: float = 0.4
    w_nontarget: float = 0.01
    # Anti-plateau signals
    w_internal: float = 0.30        # reward for interior-edge ratio
    w_size: float = 0.20            # reward for saturating interior-face count
    size_cap: int = 10
    # Lexicographic bias for the target family (presence of at least one target face anywhere)
    target_presence_bonus: float = 0.0
    # forbid/penalize particular interior face sizes (length -> penalty weight)
    #      Example for HexMesh: {3: 0.8} to penalize triangles; {4: 0.2} to also discourage quads.
    w_forbidden_faces: Dict[int, float] = field(default_factory=dict)
    genome_len_bonus: bool = False
    no_edge_score: float = 0.0       # if G has 0 edges -> exact 0.0
    disconnected_mul: float = 0.05   # strong down-weight if G is disconnected
    forest_ceiling: float = 1.19     # hard cap for acyclic (forest) graphs    
    cycle_floor: float = 1.21     # keep for backward-compat (used as a baseline)
    cycle_eps: float = 1e-3       # NEW: preserves ordering among cyclic graphs    
    forest_cap_min: float = 1.05      # cap for tiny trees (e.g., 2 nodes, 1 edge)
    forest_cap_ref_n: int = 12        # nodes at which forest cap ~ forest_ceiling
    cycle_gap: float = 0.001          # keep forests strictly below cycles
    forest_size_bonus: float = 0.08   # small upward drift for bigger trees
    forest_size_ref_n: int = 12       # nodes at which size bonus ~ forest_size_bonus


def _connectivity_cycle_gates(G: nx.Graph, raw_score: float, w: MeshWeights) -> float:
    """
    Enforce:
      - 0 edges -> 0.0
      - disconnected -> heavy multiplicative penalty
      - cycles outrank forests
    and allow trees (forests) a small monotone growth with size,
    yet still < any cyclic graph.
    """
    m = G.number_of_edges()
    if m == 0:
        return float(w.no_edge_score)

    n = G.number_of_nodes()
    c = nx.number_connected_components(G)
    mu = m - n + c  # cyclomatic number on ORIGINAL graph

    s = float(raw_score)
    if c > 1:  # penalize any disconnection strongly
        s *= float(w.disconnected_mul)

    # Cyclic graphs: keep ordering and strictly above any forest
    if mu > 0:
        baseline = max(float(w.forest_ceiling), float(w.cycle_floor))
        if s < baseline:
            # lift above baseline and preserve differences among cyclic graphs
            return baseline + float(w.cycle_eps) * max(s, 0.0)
        return s

    # Forests (acyclic): add a small size bonus and cap by a dynamic ceiling
    # size factor in [0,1] based on node count (n=2 -> 0, grows to 1 by ~forest_cap_ref_n)
    refn = max(1, int(getattr(w, "forest_cap_ref_n", 12)))
    size_factor = min(1.0, max(0.0, (n - 2) / refn))

    # forest growth bonus (applied only once component penalty applied)
    refb = max(1, int(getattr(w, "forest_size_ref_n", 12)))
    bonus_factor = min(1.0, max(0.0, (n - 2) / refb))
    s += float(getattr(w, "forest_size_bonus", 0.0)) * bonus_factor

    # dynamic cap rises with size but stays strictly below cycles
    cap_min = float(getattr(w, "forest_cap_min", 1.05))
    cap_max = float(getattr(w, "forest_ceiling", 1.19))
    cap = cap_min + (cap_max - cap_min) * size_factor
    cap = min(cap, float(getattr(w, "cycle_floor", 1.21)) - float(getattr(w, "cycle_gap", 0.001)))

    return min(s, cap)







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
        # Coerce dict / OmegaConf DictConfig into MeshWeights with defaults
        try:
            from omegaconf import DictConfig, OmegaConf  # soft import
        except Exception:  # pragma: no cover
            DictConfig, OmegaConf = None, None  # type: ignore

        w = weights
        if w is None:
            self.weights = MeshWeights()
        elif isinstance(w, MeshWeights):
            self.weights = w
        elif DictConfig is not None and isinstance(w, DictConfig):  # type: ignore
            try:
                d = OmegaConf.to_container(w, resolve=True)  # type: ignore
                self.weights = MeshWeights(**(d if isinstance(d, dict) else {}))
            except Exception:
                self.weights = MeshWeights()
        elif isinstance(w, dict):
            self.weights = MeshWeights(**w)
        else:
            # Unknown type → fall back safely
            self.weights = MeshWeights()
    
    def score(self, G: nx.Graph, meta: Optional[Dict] = None, *, return_metrics: bool = False, **_) -> float | Tuple[float, Dict[str, float]]:
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

        # Target presence bonus — check ANY face (including shell)
        has_target_face = any(len(f) == self.target_face_len for f in emb.faces)
        presence_bonus = self.weights.target_presence_bonus if has_target_face else 0.0

        # Forbidden face penalties over interior faces only
        forbidden_pen = 0.0
        if f_int_total > 0 and self.weights.w_forbidden_faces:
            hist = Counter(len(f) for f in faces_wo_shell)
            for k, w in self.weights.w_forbidden_faces.items():
                if w > 0:
                    forbidden_pen += w * (hist.get(k, 0) / f_int_total)

        score = (
            vr.base_score
            + self.weights.w_face      * face_ratio
            + self.weights.w_deg       * deg_reward
            - self.weights.w_shell     * shell_pen
            - self.weights.w_nontarget * non_target_pen
            - forbidden_pen                                   # <— NEW
            + self.weights.w_internal  * internal_edge_ratio
            + self.weights.w_size      * size_bonus
            + presence_bonus
            + gl_bonus
        )
        
        metrics = {
            "nodes": GG.number_of_nodes(),
            "edges": GG.number_of_edges(),
            "shell_len": len(emb.shell),
            "faces_total": len(emb.faces),
            "faces_interior": f_int_total,
            "target_face_count": f_int_target,
            "deg_reward": float(deg_reward),
            "internal_edge_ratio": float(internal_edge_ratio),
            "size_bonus": float(size_bonus),
        }
        final = _connectivity_cycle_gates(G, score, self.weights)
        return (float(final), metrics) if return_metrics else float(final)

        


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
        # Keep the upward bias (presence bonus), but no forbidden faces by default.
        if weights is None:
            weights = MeshWeights(
                target_presence_bonus=1.6,   # ensures hex_1 > large tri patches
                # You can add: w_forbidden_faces={3: 0.8} in evolution configs for stronger bias
                w_face=1.0, w_deg=0.6, w_shell=0.4, w_nontarget=0.5,
                w_internal=0.30, w_size=0.20, size_cap=10,
            )
        super().__init__(weights=weights, **kwargs)

# --- C#-parity triangle fitness ---------------------------------------------

@dataclass
class TriangleLegacyWeights:
    # tri mesh specific:
    tri_face_weight: float = 2.601         # was hard-coded as 2.001
    interior_deg6_weight: float = 2.0        # was hardcoded as 1.9

    # compactness / shapefactor
    vertex_weight: float = -0.5           # was hard-coded as 1.0 (per-node penalty)
    shell_vertex_weight: float = -1.5      # penalty, legacy-UI-aligned default (helps monotonicity)
    isoperimetric_quotient_weight: float = 0   
    

    genome_len_bonus: bool = False
    genome_len_bonus_weight: float = 1.0

    use_biconnected_gate: bool = True
    biconnected_gate_score: float = 1.02
    biconnected_gate_multiplier: float = 0.001

class TriangleMeshLegacyCS(PlanarBasic):
    """
    Legacy C#-aligned triangle fitness.

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

    def __init__(self, *, weights: Optional[TriangleLegacyWeights] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.w = weights or TriangleLegacyWeights()

      
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
                score += float(self.w.genome_len_bonus_weight) / gl

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
            }
            return float(score), metrics
        return float(score)

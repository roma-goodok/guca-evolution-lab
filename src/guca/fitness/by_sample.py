# src/guca/fitness/by_sample.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Hashable, Tuple
from collections import Counter

import networkx as nx

from .planar_basic import PlanarBasic, ViabilityResult, EmbeddingInfo

Hash = Hashable


@dataclass
class BySampleWeights:
    """Weights for BySample scoring."""
    w_faces: float = 1.0          # similarity of face-length distributions
    w_degrees: float = 1.0        # similarity of (interior) vertex-degree distributions
    w_shell: float = 0.4          # boundary compactness penalty
    w_internal: float = 0.30      # reward for interior-edge ratio (avoids plateaus)
    w_size: float = 0.20          # reward for saturating interior-face count
    size_cap: int = 10            # faces at which size bonus saturates
    smoothing: float = 0.0        # Laplace smoothing for distributions (optional)
    genome_len_bonus: bool = False


class BySample(PlanarBasic):
    """
    Compare a graph's interior face-length and degree distributions against a target.

    Targets can be:
      * explicit distributions (dict[int->float], normalized)
      * a reference graph: we'll compute its distributions using the same machinery

    Scoring:
      score = base
            + w_faces   * face_similarity
            + w_degrees * degree_similarity
            - w_shell   * shell_penalty
            + w_internal* internal_edge_ratio
            + w_size    * saturating_size_bonus
            + optional genome_len_bonus
    """
    def __init__(
        self,
        *,
        target_face_dist: Optional[Dict[int, float]] = None,
        target_degree_dist: Optional[Dict[int, float]] = None,
        reference_graph: Optional[nx.Graph] = None,
        weights: Optional[BySampleWeights] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.weights = weights or BySampleWeights()

        if reference_graph is not None:
            # Build targets from reference
            ref_emb = self.compute_embedding_info(reference_graph)
            ref_faces = [f for f in ref_emb.faces if f is not ref_emb.shell]
            self.target_face_dist = _normalize_counter(
                Counter(len(f) for f in ref_faces), self.weights.smoothing
            )
            deg_hist = _degree_hist(reference_graph, ref_emb.interior_nodes)
            # If no interior nodes, fallback to all nodes so degrees are not empty
            if not ref_emb.interior_nodes:
                deg_hist = Counter(dict(reference_graph.degree()))
            self.target_degree_dist = _normalize_counter(deg_hist, self.weights.smoothing)
        else:
            # Use explicit targets; validate/normalize
            self.target_face_dist = _normalize_mapping(target_face_dist, self.weights.smoothing) if target_face_dist else {}
            self.target_degree_dist = _normalize_mapping(target_degree_dist, self.weights.smoothing) if target_degree_dist else {}

    # Factory helper
    @classmethod
    def from_graph(cls, reference_graph: nx.Graph, *, weights: Optional[BySampleWeights] = None, **kwargs) -> "BySample":
        return cls(reference_graph=reference_graph, weights=weights, **kwargs)

    
    
    def score(self, G: nx.Graph, meta: Optional[Dict] = None, *, return_metrics: bool = False, **_) -> float | Tuple[float, Dict[str, float]]:
        vr: ViabilityResult = self.viability_filter(G, meta)        
        if not vr.viable:
            return (vr.base_score, {}) if return_metrics else vr.base_score

        GG = self.prepare_graph(G)
        emb: EmbeddingInfo = self.compute_embedding_info(GG)

        # Interior faces (exclude shell)
        faces_wo_shell: List[List[Hash]] = [f for f in emb.faces if f is not emb.shell]
        cand_face_dist = _normalize_counter(Counter(len(f) for f in faces_wo_shell), self.weights.smoothing)

        # Interior degrees (fallback to all nodes if there are no interior nodes)
        if emb.interior_nodes:
            deg_hist = _degree_hist(GG, emb.interior_nodes)
        else:
            deg_hist = Counter(dict(GG.degree()))
        cand_degree_dist = _normalize_counter(deg_hist, self.weights.smoothing)

        # Similarities (1 - 0.5*L1 distance) in [0,1]
        face_sim = _distribution_similarity(self.target_face_dist, cand_face_dist)
        degree_sim = _distribution_similarity(self.target_degree_dist, cand_degree_dist)

        # Shell compactness
        shell_pen = len(emb.shell) / max(1, GG.number_of_nodes())

        # Interior-edge ratio (avoids plateaus when all faces are "correct")
        e_total = _edge_set(GG)
        e_shell = _shell_edge_set(emb.shell) & e_total
        e_internal = e_total - e_shell
        internal_edge_ratio = len(e_internal) / max(1, len(e_total))

        # Saturating size bonus by number of interior faces
        size_bonus = min(1.0, len(faces_wo_shell) / max(1, self.weights.size_cap))

        # Optional genome-length bonus
        gl_bonus = 0.0
        if self.weights.genome_len_bonus:
            gl = _infer_genome_len(meta)
            if gl and gl > 0:
                gl_bonus = 1.0 / gl

        score = (
            vr.base_score
            + self.weights.w_faces   * face_sim
            + self.weights.w_degrees * degree_sim
            - self.weights.w_shell   * shell_pen
            + self.weights.w_internal* internal_edge_ratio
            + self.weights.w_size    * size_bonus
            + gl_bonus
        )
        if return_metrics:
            metrics = {
                "face_similarity": float(face_sim),
                "degree_similarity": float(degree_sim),
                "shell_penalty": float(shell_pen),
                "internal_edge_ratio": float(internal_edge_ratio),
                "size_bonus": float(size_bonus),
                "nodes": GG.number_of_nodes(),
                "edges": GG.number_of_edges(),
                "shell_len": len(emb.shell),
            }
            return float(score), metrics
        return float(score)


# ---------------------------
# Helpers
# ---------------------------
def _normalize_counter(c: Counter, smoothing: float = 0.0) -> Dict[int, float]:
    keys = list(c.keys())
    if smoothing > 0:
        # Laplace smoothing: add alpha to each seen category
        total = sum(c.values()) + smoothing * len(keys)
        return {k: (c[k] + smoothing) / total for k in keys}
    total = sum(c.values())
    return {k: (c[k] / total) for k in keys} if total > 0 else {}


def _normalize_mapping(m: Optional[Dict[int, float]], smoothing: float = 0.0) -> Dict[int, float]:
    if not m:
        return {}
    # Ensure non-negative and normalize to 1
    filtered = {int(k): max(0.0, float(v)) for k, v in m.items()}
    s = sum(filtered.values())
    if s == 0 and smoothing > 0:
        # if all zeros, assign uniform over given keys
        u = 1.0 / len(filtered)
        return {k: u for k in filtered}
    return {k: (v / s) for k, v in filtered.items()} if s > 0 else filtered


def _distribution_similarity(p: Dict[int, float], q: Dict[int, float]) -> float:
    """Return 1 - 0.5*L1 distance over the union of keys (in [0,1])."""
    keys = set(p.keys()) | set(q.keys())
    l1 = sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
    sim = 1.0 - 0.5 * min(2.0, l1)
    # If no target specified, similarity should be neutral (1.0) rather than penalizing.
    if not p and not q:
        sim = 1.0
    elif not p:
        sim = 1.0  # No target means "don't care"
    return sim


def _degree_hist(G: nx.Graph, nodes) -> Counter:
    return Counter(G.degree(n) for n in nodes)


def _edge_set(G: nx.Graph) -> set[Tuple[Hash, Hash]]:
    return {(u, v) if u <= v else (v, u) for u, v in G.edges()}


def _shell_edge_set(shell: List[Hash]) -> set[Tuple[Hash, Hash]]:
    es = set()
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

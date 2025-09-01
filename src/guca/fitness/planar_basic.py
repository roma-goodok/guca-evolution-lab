# src/guca/fitness/planar_basic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Hashable
from collections import Counter

import networkx as nx
from networkx.algorithms.planarity import check_planarity


Hash = Hashable


@dataclass
class EmbeddingInfo:
    """Computed geometric/topological features for a planar (or near-planar) graph."""
    faces: List[List[Hash]]                 # all faces as node-cycles (outer face included)
    shell: List[Hash]                       # nodes along the outer face (largest cycle)
    shell_nodes: Set[Hash]                  # set(shell)
    face_lengths: Counter                   # histogram of |face|
    interior_nodes: Set[Hash]               # V \ shell_nodes
    interior_degree_hist: Counter           # histogram of deg(v) over interior_nodes


@dataclass
class ViabilityResult:
    """Outcome of the fast pre-fitness filter."""
    viable: bool
    base_score: float                       # usually 1.0 when viable; small penalty otherwise
    reason: str                             # machine-readable reason for non-viability (or "ok")


class PlanarBasic:
    """
    Common scaffolding for GUCA graph fitness functions.

    Responsibilities:
      1) Select largest connected component (to avoid tiny fragments dominating).
      2) Apply fast viability filters (size, divergence, planarity).
      3) Compute a planar embedding (if possible) and extract faces + outer shell.
      4) Provide helpers (face-length hist, interior degree hist) for downstream scoring.

    Notes
    -----
    * When a planar embedding isn't available (rare if graph is planar), we fall back to a
      cycle-basis proxy for faces. This proxy is good enough for tiny lattice fixtures and
      v0 scoring, but we document it as an approximation.
    * Divergence is passed in via `meta={'steps': int, 'max_steps': int, 'diverged': bool}`.
      If unavailable, only size and planarity checks are applied.
    """

    def __init__(
        self,
        *,
        max_vertices: int = 2000,
        oversize_penalty: float = 0.10,
        one_node_penalty: float = 0.00,
        nonplanar_penalty: float = 0.30,
        diverged_penalty: float = 0.90,
        take_lcc: bool = True,
        require_planarity: bool = True,
    ) -> None:
        self.max_vertices = int(max_vertices)
        self.oversize_penalty = float(oversize_penalty)
        self.one_node_penalty = float(one_node_penalty)
        self.nonplanar_penalty = float(nonplanar_penalty)
        self.diverged_penalty = float(diverged_penalty)
        self.take_lcc = bool(take_lcc)
        self.require_planarity = bool(require_planarity)

    # --------------------
    # Public API
    # --------------------
    def viability_filter(
        self,
        G: nx.Graph,
        meta: Optional[Dict] = None,
    ) -> ViabilityResult:
        """Apply cheap, order-of-magnitude filters before expensive scoring."""
        if G.number_of_nodes() <= 1:
            return ViabilityResult(False, self.one_node_penalty, "one_node")

        if G.number_of_nodes() > self.max_vertices:
            return ViabilityResult(False, self.oversize_penalty, "oversize")

        if meta:
            # Prefer explicit diverged marker; otherwise infer from steps >= max_steps
            diverged = bool(meta.get("diverged", False))
            steps = meta.get("steps", None)
            max_steps = meta.get("max_steps", None)
            if diverged or (isinstance(steps, int) and isinstance(max_steps, int) and steps >= max_steps):
                return ViabilityResult(False, self.diverged_penalty, "diverged")

        # Planarity check; we don't compute embedding here yet to keep this step cheap.
        planar, _ = check_planarity(G, counterexample=False)
        if self.require_planarity and not planar:
            return ViabilityResult(False, self.nonplanar_penalty, "nonplanar")

        return ViabilityResult(True, 1.0, "ok")

    def prepare_graph(self, G: nx.Graph) -> nx.Graph:
        """Optionally reduce to the largest connected component and copy."""
        if not self.take_lcc:
            return G.copy()
        if G.number_of_nodes() == 0:
            return G.copy()
        if nx.is_connected(G):
            return G.copy()
        # Choose the largest connected component
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        return G.subgraph(comps[0]).copy()

    def compute_embedding_info(self, G: nx.Graph) -> EmbeddingInfo:
        """
        Compute faces, shell (outer face), and interior degree histogram.
        """
        planar, emb = check_planarity(G, counterexample=False)

        if planar:
            faces = self._faces_from_embedding(G, emb)
        else:
            faces = self._faces_from_cycles_proxy(G)

        # --- Pick the shell (outer) face by maximal polygon area if possible ---
        shell = None
        try:
            # Try to use a planar drawing to measure face areas
            pos = nx.planar_layout(G)
            if faces:
                areas = [abs(self._polygon_area(f, pos)) for f in faces]
                shell = faces[max(range(len(faces)), key=lambda i: areas[i])]
        except Exception:
            shell = None

        # Fallback: if anything failed, keep the previous “longest cycle” heuristic
        if shell is None:
            shell = max(faces, key=len) if faces else list(G.nodes())

        shell_nodes = set(shell)
        face_lengths = Counter(len(f) for f in faces)

        interior_nodes = set(G.nodes()) - shell_nodes
        interior_degree_hist = Counter(G.degree(n) for n in interior_nodes) if interior_nodes else Counter()

        return EmbeddingInfo(
            faces=faces,
            shell=shell,
            shell_nodes=shell_nodes,
            face_lengths=face_lengths,
            interior_nodes=interior_nodes,
            interior_degree_hist=interior_degree_hist,
        )

    @staticmethod
    def _polygon_area(face: Sequence[Hash], pos: Dict[Hash, Sequence[float]]) -> float:
        """Signed polygon area for a face using node positions (shoelace formula)."""
        if not face:
            return 0.0
        area = 0.0
        for i in range(len(face)):
            x1, y1 = pos[face[i]]
            x2, y2 = pos[face[(i + 1) % len(face)]]
            area += x1 * y2 - x2 * y1
        return 0.5 * area


    # --------------------
    # Helpers
    # --------------------
    @staticmethod
    def _faces_from_embedding(G: nx.Graph, emb) -> List[List[Hash]]:
        """
        Enumerate faces from a NetworkX PlanarEmbedding.

        Implementation detail:
          * Prefer emb.faces() when available (NetworkX >= 2.8+),
            otherwise traverse each *directed* edge's left face once
            using emb.traverse_face(u, v), marking visited directed edges.
          * We intentionally DO NOT deduplicate faces that share the same node-set
            with reverse orientation; keeping both is necessary to distinguish the
            outer vs inner face on simple cycles.
        """
        faces: List[List[Hash]] = []

        # Some NetworkX versions expose faces() directly.
        if hasattr(emb, "faces") and callable(getattr(emb, "faces")):
            # emb.faces() may yield tuples or lists; normalize to list[List[Hash]]
            for f in emb.faces():
                faces.append(list(f))
            return faces

        # Robust fallback: traverse each directed edge exactly once.
        visited: Set[Tuple[Hash, Hash]] = set()
        for u in emb:
            for v in emb[u]:
                if (u, v) in visited:
                    continue
                try:
                    cycle = list(emb.traverse_face(u, v))
                except Exception:
                    continue
                if len(cycle) >= 2:
                    # mark all directed edges along this face (orientation matters)
                    for i in range(len(cycle)):
                        a = cycle[i]
                        b = cycle[(i + 1) % len(cycle)]
                        visited.add((a, b))
                    faces.append(cycle)

        return faces


    @staticmethod
    def _faces_from_cycles_proxy(G: nx.Graph) -> List[List[Hash]]:
        """
        Proxy for faces when no embedding is available: use a simple cycle basis.

        This is an approximation (not all fundamental cycles correspond to faces),
        but is sufficient for tiny lattice fixtures in v0.
        """
        try:
            basis = nx.cycle_basis(G)
        except Exception:
            basis = []
        # Normalize to lists, ensure simple cycles
        faces = [list(cyc) for cyc in basis if len(cyc) >= 3]
        return faces

    @staticmethod
    def degree_histogram(G: nx.Graph, nodes: Optional[Iterable[Hash]] = None) -> Counter:
        """Histogram of degrees over the provided node subset (or all nodes if None)."""
        if nodes is None:
            nodes = G.nodes()
        return Counter(G.degree(n) for n in nodes)

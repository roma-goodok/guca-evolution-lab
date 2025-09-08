# src/guca/fitness/planar_basic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Hashable
from collections import Counter
import math

import networkx as nx
from networkx.algorithms.planarity import check_planarity

Hash = Hashable


@dataclass
class EmbeddingInfo:
    faces: List[List[Hash]]                 # unique faces as node-cycles (shell included)
    shell: List[Hash]                       # nodes along the outer face
    shell_nodes: Set[Hash]
    face_lengths: Counter
    interior_nodes: Set[Hash]
    interior_degree_hist: Counter


@dataclass
class ViabilityResult:
    viable: bool
    base_score: float
    reason: str


class PlanarBasic:
    """
    Common scaffolding for GUCA graph fitness functions.

    Strategy:
      1) Geometry-first: if nodes carry `pos`, compute faces by clockwise traversal in 2D.
         This aligns faces with the intended lattice (triangle/hex).
      2) Else fall back to NetworkX PlanarEmbedding traversal.
      3) Canonicalize cycles (ignore rotation & direction) to dedup.
      4) Choose shell (outer) by maximal polygon area, then classify interior.
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
    def viability_filter(self, G: nx.Graph, meta: Optional[Dict] = None) -> ViabilityResult:
        if G.number_of_nodes() <= 1:
            return ViabilityResult(False, self.one_node_penalty, "one_node")

        if G.number_of_nodes() > self.max_vertices:
            return ViabilityResult(False, self.oversize_penalty, "oversize")

        if meta:
            diverged = bool(meta.get("diverged", False))
            steps = meta.get("steps", None)
            max_steps = meta.get("max_steps", None)
            if diverged or (isinstance(steps, int) and isinstance(max_steps, int) and steps >= max_steps):
                return ViabilityResult(False, self.diverged_penalty, "diverged")

        planar, _ = check_planarity(G, counterexample=False)
        if self.require_planarity and not planar:
            return ViabilityResult(False, self.nonplanar_penalty, "nonplanar")

        return ViabilityResult(True, 1.0, "ok")

    def prepare_graph(self, G: nx.Graph) -> nx.Graph:
        if not self.take_lcc or G.number_of_nodes() == 0 or nx.is_connected(G):
            return G.copy()
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        return G.subgraph(comps[0]).copy()

    def compute_embedding_info(self, G: nx.Graph) -> EmbeddingInfo:
        # 1) Gather positions if available (prefer real positions to keep lattice faces correct)
        pos_attr = nx.get_node_attributes(G, "pos")
        pos: Optional[Dict[Hash, Tuple[float, float]]]
        if pos_attr:
            # normalize to tuples of floats
            pos = {n: (float(p[0]), float(p[1])) for n, p in pos_attr.items()}
        else:
            pos = None

        # 2) Extract faces (geometry-first)
        if pos:
            faces_raw = self._faces_from_pos(G, pos)
        else:
            planar, emb = check_planarity(G, counterexample=False)
            faces_raw = self._faces_from_embedding(G, emb) if planar else self._faces_from_cycles_proxy(G)

        # 3) Canonicalize & deduplicate cycles (ignore rotation & direction)
        canon_faces: Dict[Tuple[Hash, ...], List[Hash]] = {}
        for cyc in faces_raw:
            if len(cyc) < 3:
                continue
            key = self._canon_cycle_key(cyc)
            canon_faces[key] = list(cyc)  # last writer wins; they are equivalent

        faces = list(canon_faces.values())

        # 4) Choose shell by maximal polygon area
        if not faces:
            shell = list(G.nodes())
        else:
            if not pos:
                # compute a quick layout to measure areas when pos is missing
                pos = nx.planar_layout(G)
            areas = [abs(self._polygon_area(f, pos)) for f in faces]
            shell = faces[max(range(len(faces)), key=lambda i: areas[i])]

        # 5) Compute counts/histograms
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

    # --------------------
    # Helpers — face extraction
    # --------------------
    @staticmethod
    def _simplify_face_cycle(cyc: List[Hash]) -> List[Hash]:
        """
        Make a DCEL-traversed face a simple polygon without over-merging:
        - drop closing duplicate,
        - collapse immediate duplicates,
        - remove only *local* backtracks a,b,a -> a.
        """
        s = list(cyc)
        # drop closing duplicate
        if len(s) > 1 and s[0] == s[-1]:
            s = s[:-1]

        # collapse immediate duplicates
        t: List[Hash] = []
        for v in s:
            if not t or t[-1] != v:
                t.append(v)
        s = t

        # remove local backtracks a,b,a
        i = 0
        while i + 2 < len(s):
            if s[i] == s[i + 2]:
                # delete middle 'b' and the second 'a'; keep the first 'a'
                del s[i + 1:i + 3]
                if i > 0:
                    i -= 1
            else:
                i += 1

        return s

    @staticmethod
    def _faces_from_embedding(G: nx.Graph, emb) -> List[List[Hash]]:
        faces: List[List[Hash]] = []
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
                    for i in range(len(cycle)):
                        a = cycle[i]; b = cycle[(i + 1) % len(cycle)]
                        visited.add((a, b))
                    # keep only proper polygons
                    cycle = PlanarBasic._simplify_face_cycle(cycle)
                    if len(cycle) >= 3:
                        faces.append(cycle)
        return faces

    @staticmethod
    def _faces_from_cycles_proxy(G: nx.Graph) -> List[List[Hash]]:
        """Fallback: use a simple cycle basis as a proxy (approximation)."""
        try:
            basis = nx.cycle_basis(G)
        except Exception:
            basis = []
        return [list(cyc) for cyc in basis if len(cyc) >= 3]

    @staticmethod
    def _faces_from_pos(G: nx.Graph, pos: Dict[Hash, Tuple[float, float]]) -> List[List[Hash]]:
        """
        Geometry-driven face walking (clockwise DCEL) using node coordinates `pos`.

        For each directed half-edge (u->v), we take the face on the right by picking,
        at vertex v, the neighbor w that is immediately clockwise from u around v.
        """
        # Precompute clockwise neighbor orderings
        cw_neighbors: Dict[Hash, List[Hash]] = {}
        for v in G.nodes():
            x0, y0 = pos[v]
            neigh = list(G.neighbors(v))
            neigh.sort(
                key=lambda u: math.atan2(pos[u][1] - y0, pos[u][0] - x0),
                reverse=True,  # descending angle = clockwise
            )
            cw_neighbors[v] = neigh

        faces: List[List[Hash]] = []
        visited_half_edges: Set[Tuple[Hash, Hash]] = set()

        for u in G.nodes():
            for v in cw_neighbors[u]:
                if (u, v) in visited_half_edges:
                    continue

                # traverse one face to the right of half-edge (u->v)
                face: List[Hash] = []
                a, b = u, v
                while True:
                    visited_half_edges.add((a, b))
                    face.append(a)
                    nb = cw_neighbors[b]
                    # find index of 'a' in cw order around 'b'
                    try:
                        idx = nb.index(a)
                    except ValueError:
                        break  # disconnected / inconsistent
                    # clockwise next (right turn)
                    c = nb[(idx - 1) % len(nb)]
                    a, b = b, c
                    if (a, b) == (u, v):
                        break

                face = PlanarBasic._simplify_face_cycle(face)
                if len(face) >= 3:
                    faces.append(face)                

        return faces

    # --------------------
    # Helpers — geometry & canonicalization
    # --------------------
    @staticmethod
    def _polygon_area(face: Sequence[Hash], pos: Dict[Hash, Sequence[float]]) -> float:
        area = 0.0
        for i in range(len(face)):
            x1, y1 = pos[face[i]]
            x2, y2 = pos[face[(i + 1) % len(face)]]
            area += x1 * y2 - x2 * y1
        return 0.5 * area

    @staticmethod
    def _canon_cycle_key(cyc: Sequence[Hash]) -> Tuple[Hash, ...]:
        """
        Canonical key for a cycle ignoring rotation and direction.
        """
        s = list(cyc)
        if len(s) > 1 and s[0] == s[-1]:
            s = s[:-1]

        # best rotation forward / backward by lexicographic node id (string)
        def best_rotation(seq: List[Hash]) -> Tuple[Hash, ...]:
            n = len(seq)
            # find all indices with minimal label (by string) and pick the lexicographically smallest rotation
            labels = [str(x) for x in seq]
            min_label = min(labels)
            idxs = [i for i, lab in enumerate(labels) if lab == min_label]
            candidates = [tuple(seq[i:] + seq[:i]) for i in idxs]
            return min(candidates)

        fwd = best_rotation(s)
        bwd = best_rotation(list(reversed(s)))
        return min(fwd, bwd)

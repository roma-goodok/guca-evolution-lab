# src/guca/fitness/planar_basic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Hashable
from collections import Counter
import itertools

import networkx as nx
from networkx.algorithms.planarity import check_planarity

Hash = Hashable


@dataclass
class EmbeddingInfo:
    faces: List[List[Hash]]                 # unique faces as node-cycles (shell included)
    shell: List[Hash]                       # nodes along the outer face (maximal outer shell)
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

    Refactored to use a purely topological pipeline:
      1) Obtain a planar embedding (NetworkX).
      2) Enumerate embedding faces (half-edge walk).
      3) For each inner face, split by existing chords to get chordless minimal faces.
      4) Deduplicate cycles (ignore rotation & reversal).
      5) Shell (outer) = longest cycle among edges used by exactly one minimal inner face.
         Fallback: embedding's outer face if shell reconstruction yields none (e.g., trees).
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
        """
        Core: compute minimal inner faces and a maximal outer shell (purely topological),
        with full robustness for edgeless / acyclic / disconnected graphs.
        """
        # Early outs for degenerate inputs
        if G.number_of_nodes() == 0:
            return EmbeddingInfo(
                faces=[],
                shell=[],
                shell_nodes=set(),
                face_lengths=Counter(),
                interior_nodes=set(),
                interior_degree_hist=Counter(),
            )

        if G.number_of_edges() == 0:
            # No edges ⇒ no cycles ⇒ no faces; shell is undefined/empty
            interior_nodes = set(G.nodes())
            interior_degree_hist = Counter(G.degree(n) for n in interior_nodes)
            return EmbeddingInfo(
                faces=[],
                shell=[],
                shell_nodes=set(),
                face_lengths=Counter(),
                interior_nodes=interior_nodes,
                interior_degree_hist=interior_degree_hist,
            )

        # 1) Planarity + embedding faces
        planar, emb = check_planarity(G, counterexample=False)
        if not planar or emb is None:
            # Nonplanar or no embedding (paranoia): return empty facial data
            interior_nodes = set(G.nodes())
            interior_degree_hist = Counter(G.degree(n) for n in interior_nodes)
            return EmbeddingInfo(
                faces=[],
                shell=[],
                shell_nodes=set(),
                face_lengths=Counter(),
                interior_nodes=interior_nodes,
                interior_degree_hist=interior_degree_hist,
            )

        emb_faces, outer_idx = self._planar_faces(G, emb)  # robust: may return outer_idx=None
        # If there are no embedded cycles (e.g., forests), emb_faces == [] and outer_idx is None.

        # 2) Decompose inner faces into minimal chordless faces (if any faces exist)
        raw_minimal: List[List[Hash]] = []
        if emb_faces:
            for i, f in enumerate(emb_faces):
                if outer_idx is not None and i == outer_idx:
                    continue
                raw_minimal.extend(self._decompose_face_by_allowed_chords(f, G))

        # 3) Deduplicate minimal faces (rotation + reversal)
        minimal_inner = self._unique_cycles(raw_minimal)

        # 4) Build shell from edges on exactly one minimal inner face
        shell: List[Hash] = []
        if minimal_inner:
            shell_edges_cnt = Counter()
            for f in minimal_inner:
                for e in self._edges_in_cycle(f):
                    shell_edges_cnt[e] += 1
            shell_edges = [e for e, c in shell_edges_cnt.items() if c == 1]

            # Reconstruct maximal outer shell as the longest cycle in shell-edge subgraph
            if shell_edges:
                H = nx.Graph()
                H.add_edges_from(shell_edges)
                shell_cycles = nx.cycle_basis(H)
                if shell_cycles:
                    shell = max(shell_cycles, key=len)

        # Fallback for cases where we have cycles but reconstruction yielded none
        if not shell and emb_faces and outer_idx is not None and 0 <= outer_idx < len(emb_faces):
            shell = emb_faces[outer_idx]

        # 5) Final face list = minimal inner + (outer shell if present)
        faces = minimal_inner + ([shell] if shell else [])

        # 6) Canonicalize & deduplicate faces (safety)
        canon_faces: Dict[Tuple[Hash, ...], List[Hash]] = {}
        for cyc in faces:
            if len(cyc) < 3:
                continue
            key = self._canon_cycle_key(cyc)
            canon_faces[key] = list(cyc)
        faces_out = list(canon_faces.values())

        # 7) Shell fields and interior stats
        shell_nodes = set(shell) if shell else set()
        face_lengths = Counter(len(f) for f in faces_out)
        interior_nodes = set(G.nodes()) - shell_nodes
        interior_degree_hist = Counter(G.degree(n) for n in interior_nodes) if interior_nodes else Counter()

        return EmbeddingInfo(
            faces=faces_out,
            shell=list(shell),
            shell_nodes=shell_nodes,
            face_lengths=face_lengths,
            interior_nodes=interior_nodes,
            interior_degree_hist=interior_degree_hist,
        )

    @staticmethod
    def _planar_faces(G: nx.Graph, emb) -> Tuple[List[List[Hash]], Optional[int]]:
        """
        Enumerate faces from a NetworkX PlanarEmbedding (half-edge traversal).
        Returns (faces, outer_face_index). If no facial cycles exist, returns ([], None).
        """
        faces: List[List[Hash]] = []
        visited: Set[Tuple[Hash, Hash]] = set()
        for u in emb:
            for v in emb[u]:
                if (u, v) in visited:
                    continue
                cyc = list(emb.traverse_face(u, v, mark_half_edges=visited))
                # drop closing duplicate and require a proper polygon (>=3)
                if len(cyc) > 1 and cyc[0] == cyc[-1]:
                    cyc = cyc[:-1]
                if len(cyc) >= 3:
                    faces.append(cyc)

        if not faces:
            return [], None

        # Heuristic for outer: face with the most distinct vertices
        outer_idx = max(range(len(faces)), key=lambda i: len(set(faces[i])))
        return faces, outer_idx


    # --------------------
    # Minimal helper set (all used)
    # --------------------
    @staticmethod
    def _sorted_edge(u: Hash, v: Hash) -> Tuple[Hash, Hash]:
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def _edges_in_cycle(cycle: Sequence[Hash]):
        n = len(cycle)
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            yield PlanarBasic._sorted_edge(u, v)

      
    @staticmethod
    def _decompose_face_by_allowed_chords(face: List[Hash], G: nx.Graph) -> List[List[Hash]]:
        """
        Split a face boundary into smaller chordless cycles using only existing
        graph edges between non-consecutive boundary vertices (purely topological).
        """
        n = len(face)
        if n <= 3:
            return [face[:]]

        edges_set = {PlanarBasic._sorted_edge(*e) for e in G.edges}
        idxs = list(range(n))

        def solve(indices: List[int]) -> List[List[Hash]]:
            m = len(indices)
            if m <= 3:
                return [[face[i] for i in indices]]

            # eligible chords are nonconsecutive pairs on this fragment that exist as edges
            allowed = []
            for ai in range(m):
                for aj in range(ai + 2, m):
                    if ai == 0 and aj == m - 1:
                        continue  # wraparound neighbors are consecutive
                    u = face[indices[ai]]
                    v = face[indices[aj]]
                    if PlanarBasic._sorted_edge(u, v) in edges_set:
                        allowed.append((ai, aj))

            if not allowed:
                return [[face[i] for i in indices]]

            # choose a chord that balances the two sub-polygons
            def score(ai: int, aj: int):
                left = aj - ai + 1
                right = m - (aj - ai) + 1
                return (max(left, right), min(left, right))

            ai, aj = min(allowed, key=lambda p: score(*p))
            left = indices[ai:aj + 1]
            right = indices[aj:] + indices[:ai + 1]
            return solve(left) + solve(right)

        return solve(idxs)

    @staticmethod
    def _unique_cycles(cycles: List[List[Hash]]) -> List[List[Hash]]:
        """
        Deduplicate cycles up to rotation and reversal using _canon_cycle_key.
        """
        seen: Set[Tuple[Hash, ...]] = set()
        out: List[List[Hash]] = []
        for c in cycles:
            if len(c) < 3:
                continue
            key = PlanarBasic._canon_cycle_key(c)
            if key not in seen:
                seen.add(key)
                out.append(list(key))
        return out

    # --------------------
    # Canonicalization helper (kept from your code)
    # --------------------
    @staticmethod
    def _canon_cycle_key(cyc: Sequence[Hash]) -> Tuple[Hash, ...]:
        """
        Canonical key for a cycle ignoring rotation and direction (uses str() labels).
        """
        s = list(cyc)
        if len(s) > 1 and s[0] == s[-1]:
            s = s[:-1]

        def best_rotation(seq: List[Hash]) -> Tuple[Hash, ...]:
            n = len(seq)
            labels = [str(x) for x in seq]
            min_label = min(labels)
            idxs = [i for i, lab in enumerate(labels) if lab == min_label]
            candidates = [tuple(seq[i:] + seq[:i]) for i in idxs]
            return min(candidates)

        fwd = best_rotation(s)
        bwd = best_rotation(list(reversed(s)))
        return min(fwd, bwd)

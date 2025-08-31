from __future__ import annotations
from collections import deque, Counter
from typing import Dict, Iterable, Optional, Set, Tuple, List
from .node import Node


class GUMGraph:
    """Tiny undirected graph container for GUM with integer node ids."""

    def __init__(self) -> None:
        self._nodes: Dict[int, Node] = {}
        self._next_id: int = 0

    # --- nodes / edges -----------------------------------------------------
    def add_vertex(self, state: str, parents_count: int = 0, mark_new: bool = True) -> int:
        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = Node(id=nid, state=state, parents_count=parents_count, marked_new=mark_new)
        return nid

    def remove_vertex(self, nid: int) -> None:
        if nid not in self._nodes:
            return
        for nb in list(self._nodes[nid].neighbors):
            self._nodes[nb].neighbors.discard(nid)
        del self._nodes[nid]

    def add_edge(self, a: int, b: int) -> None:
        if a == b or a not in self._nodes or b not in self._nodes:
            return
        self._nodes[a].neighbors.add(b)
        self._nodes[b].neighbors.add(a)

    def remove_edge(self, a: int, b: int) -> None:
        if a in self._nodes:
            self._nodes[a].neighbors.discard(b)
        if b in self._nodes:
            self._nodes[b].neighbors.discard(a)

    # --- accessors ---------------------------------------------------------
    def node(self, nid: int) -> Node:
        return self._nodes[nid]

    def nodes(self) -> Iterable[Node]:
        return self._nodes.values()

    def node_ids(self) -> Iterable[int]:
        return self._nodes.keys()

    def edges(self) -> Iterable[Tuple[int, int]]:
        seen: Set[Tuple[int, int]] = set()
        for n in self._nodes.values():
            for nb in n.neighbors:
                a, b = (n.id, nb) if n.id < nb else (nb, n.id)
                if (a, b) not in seen:
                    seen.add((a, b))
                    yield (a, b)

    # --- iteration snapshots ----------------------------------------------
    def snapshot_nodes(self) -> None:
        for n in self._nodes.values():
            n.marked_new = False
            n.saved_state = n.state
            n.saved_parents_count = n.parents_count
            # Node.degree is assumed to be a property; if not, adapt here
            n.saved_degree = n.degree

    def delete_marked(self) -> None:
        for nid in list(self._nodes.keys()):
            if self._nodes[nid].marked_deleted:
                self.remove_vertex(nid)

    # --- helpers (legacy) --------------------------------------------------
    def nearest_with_saved_state(self, start: int, target_state: str, max_distance: int = 2) -> Optional[int]:
        """Legacy: returns the first found nearest node (by BFS) with saved_state == target_state."""
        if start not in self._nodes:
            return None
        q = deque([(start, 0)])
        visited = {start}
        while q:
            nid, dist = q.popleft()
            for nb in self._nodes[nid].neighbors:
                if nb in visited:
                    continue
                visited.add(nb)
                if dist + 1 <= max_distance:
                    q.append((nb, dist + 1))
                if nb != start:
                    nbn = self._nodes[nb]
                    if (not nbn.marked_new) and (nbn.saved_state == target_state) and (not self._nodes[start].marked_deleted):
                        return nb
        return None

    # --- NEW: deterministic nearest with options ---------------------------
    def try_connect_with_nearest(
        self,
        u_id: int,
        *,
        required_state: Optional[str] = None,
        max_depth: int = 2,
        tie_breaker: str = "stable",     # "stable" | "random" | "by_id" | "by_creation"
        connect_all: bool = False,
        rng=None,
    ) -> None:
        """
        Deterministic nearest connection with BFS and optional state filter.
        - BFS from u_id up to max_depth
        - Let d* be the minimal depth where any eligible candidates exist
        - If connect_all: connect (u_id, v) for every v at depth d*
          Else: choose exactly one v by tie_breaker and connect
        Eligibility for a candidate v:
          * v != u_id
          * v not already adjacent to u_id
          * (if required_state is not None) compare v.saved_state (fallback v.state) to required_state
          * start node is not marked_deleted
        The BFS neighbor enqueue order is sorted by neighbor id to ensure determinism.
        """

        # preconditions
        if u_id not in self._nodes:
            return

        start_node = self._nodes[u_id]
        if getattr(start_node, "marked_deleted", False):
            return

        # Snapshot adjacency of u_id (avoid duplicate edges)
        nbrs_u: Set[int] = set(self._nodes[u_id].neighbors)

        # Helper: get candidate's state (prefer saved_state, else state)
        def _candidate_state(v_id: int) -> Optional[str]:
            n = self._nodes.get(v_id)
            if n is None:
                return None
            if hasattr(n, "saved_state") and n.saved_state is not None:
                return n.saved_state
            return n.state

        # Eligibility predicate
        def eligible(v_id: int) -> bool:
            if v_id == u_id or v_id in nbrs_u:
                return False
            n = self._nodes[v_id]
            if getattr(n, "marked_new", False):
                return False
            if required_state is None:
                return True
            vst = _candidate_state(v_id)
            return (vst is not None) and (str(vst) == str(required_state))

        # BFS to collect all candidates at the *first* depth where any exist
        visited: Set[int] = {u_id}
        q: deque[Tuple[int, int]] = deque([(u_id, 0)])
        found_depth: Optional[int] = None
        found: List[int] = []

        while q:
            nid, d = q.popleft()
            if found_depth is not None and d > found_depth:
                break

            if 0 < d <= max_depth and eligible(nid):
                found_depth = d
                found.append(nid)
                # do not return yet; continue draining queue for all nodes at depth d*
                continue

            if d < max_depth:
                # Deterministic enqueue order
                for nb in sorted(self._nodes[nid].neighbors):
                    if nb not in visited:
                        visited.add(nb)
                        q.append((nb, d + 1))

        if not found:
            return  # nothing to connect

        # Tolerant edge adder
        def _safe_add_edge(a: int, b: int) -> None:
            if a == b:
                return
            if a not in self._nodes or b not in self._nodes:
                return
            if b in self._nodes[a].neighbors:
                return
            self._nodes[a].neighbors.add(b)
            self._nodes[b].neighbors.add(a)

        # Connect to all at minimal depth
        if connect_all:
            for v in found:
                _safe_add_edge(u_id, v)
            return

        # Choose a single candidate by tie-breaker
        mode = (tie_breaker or "stable").lower()
        # In this graph, id order == creation order; if you later add an explicit creation_index,
        # switch "by_creation" to sort by that field instead of id.
        if mode == "random" and rng is not None:
            v_id = rng.choice(found)
        elif mode in ("by_id", "by_creation", "stable"):
            v_id = min(found)  # ids are monotonically assigned by add_vertex
        else:
            v_id = min(found)

        _safe_add_edge(u_id, v_id)


def stats_summary(g: GUMGraph) -> dict:
    states = Counter(n.state for n in g.nodes())
    return {
        "nodes": sum(1 for _ in g.nodes()),
        "edges": sum(1 for _ in g.edges()),
        "states_count": dict(states),
    }

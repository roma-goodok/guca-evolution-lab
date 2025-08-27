from __future__ import annotations
from collections import deque, Counter
from typing import Dict, Iterable, Optional, Set, Tuple
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
            n.saved_degree = n.degree

    def delete_marked(self) -> None:
        for nid in list(self._nodes.keys()):
            if self._nodes[nid].marked_deleted:
                self.remove_vertex(nid)

    # --- helpers -----------------------------------------------------------
    def nearest_with_saved_state(self, start: int, target_state: str, max_distance: int = 2) -> Optional[int]:
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

def stats_summary(g: GUMGraph) -> dict:
    states = Counter(n.state for n in g.nodes())
    return {
        "nodes": sum(1 for _ in g.nodes()),
        "edges": sum(1 for _ in g.edges()),
        "states_count": dict(states),
    }

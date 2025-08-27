from __future__ import annotations
from .graph import GUMGraph
from .node import Node
from .rules import ChangeTable, Rule, OperationKind, TranscriptionWay, CountCompare, rule_matches

class GraphUnfoldingMachine:
    """
    Minimal GUM engine:
    - Snapshot node fields at iteration start.
    - Match rules against snapshots.
    - resettable/continuable transcription; range/exact comparators.
    - Stop on max_steps or two empty iterations.
    """

    def __init__(self, graph: GUMGraph, *,
                 start_state: str = "A",
                 transcription: TranscriptionWay = TranscriptionWay.resettable,
                 count_compare: CountCompare = CountCompare.range,
                 max_vertices: int = 0,
                 max_steps: int = 100) -> None:
        self.graph = graph
        self.transcription = transcription
        self.count_compare = count_compare
        self.max_vertices = max_vertices
        self.max_steps = max_steps
        self.change_table: ChangeTable = ChangeTable()
        if not any(True for _ in self.graph.nodes()):
            self.graph.add_vertex(state=start_state, parents_count=0, mark_new=True)
        self.passed_steps = 0
        self._empty_iters = 0

    def run(self) -> None:
        self.passed_steps = 0
        self._empty_iters = 0
        while self.max_steps < 0 or self.passed_steps < self.max_steps:
            if not self._next_step():
                self._empty_iters += 1
            else:
                self._empty_iters = 0
            self.passed_steps += 1
            if self._empty_iters >= 2:
                break

    def _next_step(self) -> bool:
        self.graph.snapshot_nodes()
        did_anything = False

        node_ids = list(self.graph.node_ids())
        for nid in node_ids:
            if nid not in self.graph._nodes:
                continue
            node = self.graph.node(nid)
            if node.marked_deleted:
                continue

            rule, idx = self._find_rule_for(node)
            if rule:
                self._apply(node, rule)
                did_anything = True
                rule.is_active = True
                rule.was_active = True
                rule.last_activation_index = (rule.last_activation_index + 1) if rule.last_activation_index >= 0 else 0
                if self.transcription == TranscriptionWay.continuable:
                    node.rule_index = idx + 1 if (idx + 1) < len(self.change_table) else 0

            node.prior_state = node.saved_state

        self.graph.delete_marked()
        return did_anything

    def _find_rule_for(self, node: Node):
        start = 0 if self.transcription == TranscriptionWay.resettable else node.rule_index
        n = len(self.change_table)
        # forward
        for i in range(start, n):
            r = self.change_table[i]
            if rule_matches(node.saved_state, node.prior_state, node.saved_degree,
                            node.saved_parents_count, r, self.count_compare):
                return r, i
        # wrap when continuable
        if self.transcription == TranscriptionWay.continuable and start > 0:
            for i in range(0, start):
                r = self.change_table[i]
                if rule_matches(node.saved_state, node.prior_state, node.saved_degree,
                                node.saved_parents_count, r, self.count_compare):
                    return r, i
        return None, -1

    def _apply(self, node: Node, rule: Rule) -> None:
        kind = rule.operation.kind
        opnd = rule.operation.operand

        if kind == OperationKind.TurnToState and opnd:
            node.state = opnd
            return

        if kind == OperationKind.GiveBirth and opnd:
            if self.max_vertices == 0 or sum(1 for _ in self.graph.nodes()) < self.max_vertices:
                self.graph.add_vertex(state=opnd, parents_count=node.parents_count + 1, mark_new=True)
            return

        if kind == OperationKind.GiveBirthConnected and opnd:
            if self.max_vertices == 0 or sum(1 for _ in self.graph.nodes()) < self.max_vertices:
                nid = self.graph.add_vertex(state=opnd, parents_count=node.parents_count + 1, mark_new=True)
                self.graph.add_edge(node.id, nid)
            return

        if kind == OperationKind.TryToConnectWith and opnd:
            for other in list(self.graph.nodes()):
                if other.id == node.id or other.marked_new or other.marked_deleted:
                    continue
                if other.saved_state != opnd:
                    continue
                if other.id not in node.neighbors:
                    self.graph.add_edge(node.id, other.id)
            return

        if kind == OperationKind.TryToConnectWithNearest and opnd:
            nb = self.graph.nearest_with_saved_state(node.id, opnd, max_distance=2)
            if nb is not None and nb not in node.neighbors:
                self.graph.add_edge(node.id, nb)
            return

        if kind == OperationKind.DisconnectFrom and opnd:
            for nb in list(node.neighbors):
                other = self.graph.node(nb)
                if (not other.marked_new) and other.saved_state == opnd and (not node.marked_deleted):
                    self.graph.remove_edge(node.id, nb)
            return

        if kind == OperationKind.Die:
            node.marked_deleted = True
            return

# src/guca/ga/simulate.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx

from guca.ga.encoding import OpKind, decode_gene, sanitize_gene
from guca.core.graph import GUMGraph as CoreGraph
from guca.core.machine import GraphUnfoldingMachine, TranscriptionWay, CountCompare
from guca.core.rules import ChangeTable as CoreChangeTable, Rule as CoreRule, Condition as CoreCond, Operation as CoreOp, OperationKind as CoreOpKind

def _opkind_to_core(k: OpKind) -> CoreOpKind:
    if k == OpKind.TurnToState: return CoreOpKind.TurnToState
    if k in (OpKind.GiveBirthConnected, OpKind.GiveBirth): return CoreOpKind.GiveBirthConnected
    if k == OpKind.DisconnectFrom: return CoreOpKind.DisconnectFrom
    if k in (OpKind.TryToConnectWithNearest, OpKind.TryToConnectNearest): return CoreOpKind.TryToConnectWithNearest
    if k == OpKind.TryToConnectWith: return CoreOpKind.TryToConnectWith
    return CoreOpKind.TurnToState

def _genes_to_core_change_table(genes: List[int], states: List[str], machine_encoding: Dict[str, Any] | None) -> CoreChangeTable:
    n = max(1, len(states))
    enc = machine_encoding or {}
    if enc.get("sanitize_on_decode", False):
        genes = [
            sanitize_gene(
                g,
                state_count=n,
                enforce_semantics=bool(enc.get("enforce_semantics", False)),
                canonicalize_flags=bool(enc.get("canonicalize_flags", False)),
                enforce_bounds_order=bool(enc.get("enforce_bounds_order", False)),
            )
            for g in genes
        ]
    tbl = CoreChangeTable()
    for g in genes:
        r = decode_gene(g, state_count=n)
        cond = CoreCond(
            current=states[int(r.cond_current) % n],
            prior=("any" if r.prior is None else states[int(r.prior) % n]),
            conn_ge=(-1 if r.conn_ge is None else int(r.conn_ge)),
            conn_le=(-1 if r.conn_le is None else int(r.conn_le)),
            parents_ge=(-1 if r.parents_ge is None else int(r.parents_ge)),
            parents_le=(-1 if r.parents_le is None else int(r.parents_le)),
        )
        op = CoreOp(kind=_opkind_to_core(r.op_kind), operand=None if r.operand is None else states[int(r.operand) % n])
        tbl.append(CoreRule(condition=cond, operation=op))
    return tbl

def _core_graph_to_nx(g: CoreGraph, states: List[str]) -> nx.Graph:
    G = nx.Graph()
    for n in g.nodes():
        st = n.state
        sid = states.index(st) if isinstance(st, str) and st in states else 0
        G.add_node(n.id, state=st, state_id=sid)
    for u, v in g.edges():
        G.add_edge(u, v)
    return G

def simulate_genome(
    genes: List[int],
    *,
    states: List[str],
    machine_cfg: Dict[str, Any],
    collect_activity: bool = False,
) -> nx.Graph | Tuple[nx.Graph, List[bool]]:
    """Run core GraphUnfoldingMachine and return nx graph (+optional activity mask)."""
    m = GraphUnfoldingMachine(
        CoreGraph(),
        start_state=str(machine_cfg.get("start_state", "A")),
        transcription=TranscriptionWay(machine_cfg.get("transcription", "resettable")),
        count_compare=CountCompare(machine_cfg.get("count_compare", "range")),
        max_vertices=int(machine_cfg.get("max_vertices", 0)),
        max_steps=int(machine_cfg.get("max_steps", 120)),
        nearest_max_depth=int((machine_cfg.get("nearest_search", {}) or {}).get("max_depth", 2)),
        nearest_tie_breaker=str((machine_cfg.get("nearest_search", {}) or {}).get("tie_breaker", "stable")),
        nearest_connect_all=bool((machine_cfg.get("nearest_search", {}) or {}).get("connect_all", False)),
        rng_seed=machine_cfg.get("rng_seed", None),
    )
    m.change_table = _genes_to_core_change_table(genes, states, dict(machine_cfg.get("encoding", {}) or {}))
    if not any(True for _ in m.graph.nodes()):
        m.graph.add_vertex(state=str(machine_cfg.get("start_state", "A")), parents_count=0, mark_new=True)
    m.run()
    G = _core_graph_to_nx(m.graph, states)
    if not collect_activity:
        return G
    mask = [bool(r.was_active) for r in m.change_table]
    return G, mask

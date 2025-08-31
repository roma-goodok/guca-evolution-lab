import json
from pathlib import Path
import yaml

from guca.core.graph import GUMGraph
from guca.core.machine import GraphUnfoldingMachine
from guca.core.rules import change_table_from_yaml, TranscriptionWay, CountCompare


def _build_machine_from_dict(cfg: dict):
    machine_cfg = cfg.get("machine", {}) or {}
    rules_yaml = cfg.get("rules", []) or []
    init_cfg = cfg.get("init_graph", {}) or {}

    # seed a tiny graph
    g = GUMGraph()
    id_map = {}
    next_id = 0
    nodes = init_cfg.get("nodes") or []
    edges = init_cfg.get("edges") or []

    if not nodes:
        nid = g.add_vertex(state=str(machine_cfg.get("start_state", "A")))
        id_map[0] = nid
    else:
        for item in nodes:
            if isinstance(item, dict):
                state = item.get("state", str(machine_cfg.get("start_state", "A")))
                req_id = item.get("id", None)
            else:
                state = str(item)
                req_id = None
            nid = g.add_vertex(state=state)
            if req_id is None:
                req_id = nid
            id_map[req_id] = nid

    for u, v in edges:
        g.add_edge(id_map[u], id_map[v])

    nearest = machine_cfg.get("nearest_search", {}) or {}

    m = GraphUnfoldingMachine(
        g,
        start_state=str(machine_cfg.get("start_state", "A")),
        transcription=TranscriptionWay(machine_cfg.get("transcription", "resettable")),
        count_compare=CountCompare(machine_cfg.get("count_compare", "range")),
        max_vertices=int(machine_cfg.get("max_vertices", 0)),
        max_steps=int(machine_cfg.get("max_steps", 1)),
        nearest_max_depth=int(nearest.get("max_depth", 2)),
        nearest_tie_breaker=str(nearest.get("tie_breaker", "stable")),
        nearest_connect_all=bool(nearest.get("connect_all", False)),
        rng_seed=machine_cfg.get("rng_seed", 42),
    )
    m.change_table = change_table_from_yaml(rules_yaml)
    return m, g


def _edges_set(g: GUMGraph):
    return {tuple(sorted(e)) for e in g.edges()}


def test_nearest_required_state_stable_depth2_picks_min_id(tmp_path):
    """
    Shape:
        0(A) - 1(B) - 3(C)
           \         /
            2(B) -- 4(C)

    From 0 (A), with operand C and max_depth=2 → candidates {3,4} at depth 2.
    'stable' tie-breaker should pick min id (3).
    """
    d = yaml.safe_load("""
machine:
  start_state: A
  max_vertices: 100
  max_steps: 1
  rng_seed: 123
  nearest_search:
    max_depth: 2
    tie_breaker: stable
    connect_all: false
init_graph:
  nodes:
    - {id: 0, state: A}
    - {id: 1, state: B}
    - {id: 2, state: B}
    - {id: 3, state: C}
    - {id: 4, state: C}
  edges:
    - [0, 1]
    - [0, 2]
    - [1, 3]
    - [2, 4]
rules:
  - condition: { current: A }
    op: { kind: TryToConnectWithNearest, operand: C }
""")
    m, g = _build_machine_from_dict(d)
    m.run()

    es = _edges_set(g)
    assert (0, 3) in es, f"expected edge (0,3) in {es}"


def test_nearest_required_state_connect_all(tmp_path):
    """
    Shape:
        0(A) - 1(X)
               |   \
               2(C) 3(C)

    From 0 with operand C, depth=2 → candidates {2,3}; connect_all=true => add both edges.
    """
    d = yaml.safe_load("""
machine:
  start_state: A
  max_vertices: 100
  max_steps: 1
  rng_seed: 123
  nearest_search:
    max_depth: 2
    tie_breaker: stable
    connect_all: true
init_graph:
  nodes:
    - {id: 0, state: A}
    - {id: 1, state: X}
    - {id: 2, state: C}
    - {id: 3, state: C}
  edges:
    - [0, 1]
    - [1, 2]
    - [1, 3]
rules:
  - condition: { current: A }
    op: { kind: TryToConnectWithNearest, operand: C }
""")
    m, g = _build_machine_from_dict(d)
    m.run()
    es = _edges_set(g)
    assert (0, 2) in es and (0, 3) in es, f"expected edges (0,2) and (0,3) in {es}"


def test_nearest_required_state_respects_max_depth(tmp_path):
    """
    Shape:
      0(A) - 1(B) - 2(B) - 3(C)
    With max_depth=2, no candidate at depth<=2 → no new edge.
    """
    d = yaml.safe_load("""
machine:
  start_state: A
  max_vertices: 100
  max_steps: 1
  rng_seed: 123
  nearest_search:
    max_depth: 2
    tie_breaker: stable
    connect_all: false
init_graph:
  nodes:
    - {id: 0, state: A}
    - {id: 1, state: B}
    - {id: 2, state: B}
    - {id: 3, state: C}
  edges:
    - [0, 1]
    - [1, 2]
    - [2, 3]
rules:
  - condition: { current: A }
    op: { kind: TryToConnectWithNearest, operand: C }
""")
    m, g = _build_machine_from_dict(d)
    before = _edges_set(g)
    m.run()
    after = _edges_set(g)
    assert before == after, f"no new edges expected, got: {after - before}"


def test_nearest_random_is_reproducible(tmp_path):
    """
    Same diamond as first test; random with fixed seed must be reproducible.
    """
    base = yaml.safe_load("""
machine:
  start_state: A
  max_vertices: 100
  max_steps: 1
  rng_seed: 987
  nearest_search:
    max_depth: 2
    tie_breaker: random
    connect_all: false
init_graph:
  nodes:
    - {id: 0, state: A}
    - {id: 1, state: B}
    - {id: 2, state: B}
    - {id: 3, state: C}
    - {id: 4, state: C}
  edges:
    - [0, 1]
    - [0, 2]
    - [1, 3]
    - [2, 4]
rules:
  - condition: { current: A }
    op: { kind: TryToConnectWithNearest, operand: C }
""")

    # run twice with same seed
    m1, g1 = _build_machine_from_dict(base)
    m1.run()
    es1 = _edges_set(g1)

    m2, g2 = _build_machine_from_dict(base)
    m2.run()
    es2 = _edges_set(g2)

    assert es1 == es2, f"random selection should be reproducible with same seed; {es1} != {es2}"
    # also sanity check we connected to either 3 or 4
    assert (0, 3) in es1 or (0, 4) in es1

# src/guca/ga/toolbox.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Sequence
from collections import Counter


import random

import networkx as nx
from deap import base, creator, tools
from multiprocessing import get_context

from guca.ga.encoding import (
    Rule, OpKind, encode_rule, decode_gene, random_gene, labels_to_state_maps,
    sanitize_gene
)
from guca.ga.operators import splice_cx, make_mutate_fn

# soft import for tqdm
try:
    from tqdm import tqdm  # type: ignore
except Exception:          # pragma: no cover
    tqdm = None


# =============================
# Mini engine: genome -> graph
# =============================
def _nearest_candidates(G: nx.Graph, u, operand_sid: Optional[int], *, max_depth: int, tie_breaker: str, connect_all: bool) -> List:
    """
    BFS from u up to max_depth, return nodes at minimal distance that match operand_sid (or any if None).
    Excludes self and current neighbors.
    """
    from collections import deque

    seen = {u}
    q = deque([(u, 0)])
    neighbors_u = set(G.neighbors(u)) if G.has_node(u) else set()
    candidates_at_depth: Dict[int, List] = {}

    while q:
        v, d = q.popleft()
        if d > max_depth:
            break
        # Check eligibility (skip depth==0 which is u)
        if d > 0:
            if v not in neighbors_u and v != u:
                vsid = G.nodes[v].get("state_id")
                if operand_sid is None or vsid == operand_sid:
                    candidates_at_depth.setdefault(d, []).append(v)
        if d == max_depth:
            continue
        for w in G.neighbors(v):
            if w not in seen:
                seen.add(w)
                q.append((w, d + 1))

    if not candidates_at_depth:
        return []
    dstar = min(candidates_at_depth)
    cands = candidates_at_depth[dstar]
    if connect_all:
        return sorted(cands)
    # pick one by tie-breaker
    if tie_breaker in ("stable", "by_id"):
        return [min(cands)]
    elif tie_breaker == "by_creation":
        return [min(cands)]  # we don't track creation index; ids are stable
    else:
        # random choice — caller must seed RNG beforehand if reproducibility is required
        import random
        return [random.choice(cands)]

def _check_rule_condition(G: nx.Graph, u: int, rule: Rule) -> bool:
    """
    Gate a rule by its extended condition:
      - current state matches (enforced by the caller's indexing)
      - prior (if set) equals node.prev_state_id
      - degree bounds (conn_ge/le) on current graph degree
      - parents bounds (parents_ge/le) on node.parents_count (defaults to 0)
    """
    cur_sid = int(G.nodes[u].get("state_id"))
    if int(rule.cond_current) != cur_sid:
        return False

    # prior state
    prior_sid = G.nodes[u].get("prev_state_id")
    if rule.prior is not None and prior_sid is not None:
        if int(prior_sid) != int(rule.prior):
            return False

    # degree bounds
    deg = int(G.degree[u])
    if rule.conn_ge is not None and deg < int(rule.conn_ge):
        return False
    if rule.conn_le is not None and deg > int(rule.conn_le):
        return False

    # parents bounds
    parents = int(G.nodes[u].get("parents_count", 0))
    if rule.parents_ge is not None and parents < int(rule.parents_ge):
        return False
    if rule.parents_le is not None and parents > int(rule.parents_le):
        return False

    return True


def _apply_rules_once(
    G: nx.Graph,
    rules: List[Rule],
    *,
    machine_cfg: Dict[str, Any],
    active_hits: Optional[List[bool]] = None,
    rid_by_cond: Optional[Dict[int, int]] = None,
) -> bool:
    """
    Apply rules to all nodes in a deterministic order; return True if any change occurred.
    If 'active_hits' is provided, mark rules that EFFECTED a change at least once in this step.
    Rule selection is by the first rule for a given cond_current (rid_by_cond).
    """
    # Unpack machine params
    max_vertices = int(machine_cfg.get("max_vertices", 0) or 0)

    # ensure per-node bookkeeping
    for u in G.nodes():
        G.nodes[u].setdefault("parents_count", 0)
        if "prev_state_id" not in G.nodes[u]:
            G.nodes[u]["prev_state_id"] = int(G.nodes[u].get("state_id"))

    changed = False
    to_add_nodes: List[Tuple[int, int]] = []   # (new_id, state_id)
    to_add_edges: List[Tuple[int, int]] = []
    to_remove_edges: List[Tuple[int, int]] = []
    to_remove_nodes: List[int] = []
    state_updates: List[Tuple[int, int]] = []

    for u in sorted(G.nodes()):
        cur_sid = int(G.nodes[u]["state_id"])

        # pick first matching rule by cond_current via table index
        rid = None
        if rid_by_cond is not None:
            rid = rid_by_cond.get(int(cur_sid))
            if rid is None:
                # fallback to linear search (should not happen if rid_by_cond is provided)
                rid = next((i for i, r in enumerate(rules) if int(r.cond_current) == cur_sid), None)
        else:
            # fallback to linear search
            rid = next((i for i, r in enumerate(rules) if int(r.cond_current) == cur_sid), None)
        if rid is None:
            continue

        rule = rules[rid]
        # NEW: full condition gate
        if not _check_rule_condition(G, u, rule):
            continue

        op = rule.op_kind
        operand_sid = rule.operand if rule.operand is not None else None

        rule_effective = False

        if op == OpKind.TurnToState:
            if operand_sid is not None and operand_sid != cur_sid:
                state_updates.append((u, int(operand_sid)))
                rule_effective = True

        elif op == OpKind.GiveBirth:
            if operand_sid is not None:
                # Treat legacy 'GiveBirth' exactly like 'GiveBirthConnected' (C# effective semantics)
                new_id = max(G.nodes()) + 1 if G.number_of_nodes() > 0 else 0
                to_add_nodes.append((new_id, int(operand_sid)))
                to_add_edges.append((u, new_id))  # <-- connect newborn to the active parent
                rule_effective = True


        elif op == OpKind.GiveBirthConnected:
            if operand_sid is not None:
                new_id = max(G.nodes()) + 1 if G.number_of_nodes() > 0 else 0
                to_add_nodes.append((new_id, int(operand_sid)))
                to_add_edges.append((u, new_id))
                rule_effective = True

        elif op == OpKind.TryToConnectWith:
            if operand_sid is not None:
                targets = [v for v in G.nodes()
                           if v != u and G.nodes[v].get("state_id") == int(operand_sid) and not G.has_edge(u, v)]
                if targets:
                    for v in targets:
                        to_add_edges.append((u, v))
                    rule_effective = True

        elif op == OpKind.TryToConnectWithNearest:
            if operand_sid is not None:
                cands = _nearest_candidates(
                    G, u, int(operand_sid),
                    max_depth=int(machine_cfg.get("nearest_search", {}).get("max_depth", 2)),
                    tie_breaker=str(machine_cfg.get("nearest_search", {}).get("tie_breaker", "stable")),
                    connect_all=bool(machine_cfg.get("nearest_search", {}).get("connect_all", False))
                )
                cands = [v for v in cands if not G.has_edge(u, v)]
                if cands:
                    for v in cands:
                        to_add_edges.append((u, v))
                    rule_effective = True

        elif op == OpKind.DisconnectFrom:
            if operand_sid is not None:
                targets = [v for v in list(G.neighbors(u)) if G.nodes[v].get("state_id") == int(operand_sid)]
                if targets:
                    for v in targets:
                        to_remove_edges.append((u, v))
                    rule_effective = True

        elif op == OpKind.Die:
            to_remove_nodes.append(u)
            rule_effective = True

        if rule_effective and active_hits is not None:
            active_hits[rid] = True

    # respect max_vertices by capping additions (best effort)
    if max_vertices and G.number_of_nodes() + len(to_add_nodes) > max_vertices:
        allow = max(0, max_vertices - G.number_of_nodes())
        to_add_nodes = to_add_nodes[:allow]

    # apply structural edits
    # add nodes (and initialize bookkeeping)
    for (new_id, sid) in to_add_nodes:
        G.add_node(new_id, state_id=int(sid), parents_count=1, prev_state_id=int(sid))
    # add edges
    for (u, v) in to_add_edges:
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v)
    # remove edges
    for (u, v) in to_remove_edges:
        if G.has_node(u) and G.has_edge(u, v):
            G.remove_edge(u, v)
    # remove nodes
    if to_remove_nodes:
        for u in sorted(set(to_remove_nodes), reverse=True):
            if G.has_node(u):
                G.remove_node(u)

    # update states
    for (u, sid) in state_updates:
        if G.has_node(u) and int(G.nodes[u]["state_id"]) != int(sid):
            G.nodes[u]["state_id"] = int(sid)
            changed = True
     
    # consider structural edits as changes too
    if (to_add_nodes or to_add_edges or to_remove_edges or to_remove_nodes or state_updates):
        changed = True
    return changed


def _rank_weights(n: int) -> List[float]:
    # linear ranking: n..1 then normalize
    raw = [float(n - i) for i in range(n)]
    s = sum(raw) or 1.0
    return [x/s for x in raw]

def _roulette_weights(fits: List[float]) -> List[float]:
    # shift if any non-positive, then normalize
    mn = min(fits) if fits else 0.0
    shift = 1e-9 - mn if mn <= 0 else 0.0
    raw = [max(0.0, f + shift) for f in fits]
    s = sum(raw) or 1.0
    return [x/s for x in raw]


# --- Selection helpers --------------------------------------------------------
from typing import Callable

def _sel_rank(pop, k: int, rng: random.Random, random_ratio: float = 0.0):
    """
    Linear rank selection (higher fitness -> higher rank).
    Mix in a small fraction of uniform random picks if random_ratio > 0.
    Sampling is with replacement.
    """
    if k <= 0:
        return []

    # Sort by descending fitness
    sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
    n = len(sorted_pop)
    if n == 0:
        return []

    # ranks: n..1 (linear)
    ranks = list(range(n, 0, -1))
    total = sum(ranks)
    probs = [r / total for r in ranks]

    k_rank = int(round(k * max(0.0, 1.0 - random_ratio)))
    k_rand = max(0, k - k_rank)

    selected = []
    if k_rank > 0:
        # rng.choices is deterministic under seeded rng
        selected.extend(rng.choices(sorted_pop, weights=probs, k=k_rank))
    if k_rand > 0:
        selected.extend(rng.choices(pop, k=k_rand))
    return selected


def _make_selector(method: str, tournament_k: int, random_ratio: float, rng: random.Random) -> Callable:
    """
    Return a selection function sel(pop, k) based on GA config.
    """
    m = (method or "rank").lower()
    if m == "rank":
        return lambda pop, k: _sel_rank(pop, k, rng, random_ratio=random_ratio)
    elif m == "tournament":
        # keep DEAP tournament
        return lambda pop, k: tools.selTournament(pop, k=k, tournsize=max(2, int(tournament_k)))
    elif m == "roulette":
        # roulette on raw fitness (DEAP)
        # optional mixing with random picks if random_ratio > 0
        def _roulette(pop, k):
            k_rank = int(round(k * max(0.0, 1.0 - random_ratio)))
            k_rand = max(0, k - k_rank)
            part = tools.selRoulette(pop, k=k_rank) if k_rank > 0 else []
            if k_rand > 0:
                part.extend(rng.choices(pop, k=k_rand))
            return part
        return _roulette
    else:
        # fallback = rank
        return lambda pop, k: _sel_rank(pop, k, rng, random_ratio=random_ratio)




def simulate_genome(
    genes: List[int],
    *,
    states: List[str],
    machine_cfg: Dict[str, Any],
    collect_activity: bool = False,
) -> nx.Graph | Tuple[nx.Graph, List[bool]]:
    """
    Minimal in-process simulator for GA evaluation.
    Starts from one node with start_state unless an init graph is provided by the caller.
    If collect_activity=True, also returns a boolean list mark of rules that EFFECTED changes.
    """
    # initial graph: single node with start_state
    start_label = str(machine_cfg.get("start_state", "A"))
    _, inv = labels_to_state_maps(states)
    try:
        start_sid = inv.index(start_label)
    except ValueError:
        start_sid = 0

    G = nx.Graph()
    G.add_node(0, state_id=int(start_sid), parents_count=0, prev_state_id=int(start_sid))

    # decode rules and stabilize priority (first rule per cond_current wins)
    enc_cfg = dict(machine_cfg.get("encoding", {}))
    sanitize_on_decode = bool(enc_cfg.get("sanitize_on_decode", False))
    enforce_semantics  = bool(enc_cfg.get("enforce_semantics", False))
    canonicalize_flags = bool(enc_cfg.get("canonicalize_flags", False))
    enforce_bounds_ord = bool(enc_cfg.get("enforce_bounds_order", False))

    if sanitize_on_decode:
        genes_eff = [
            sanitize_gene(
                g,
                state_count=len(states),
                enforce_semantics=enforce_semantics,
                canonicalize_flags=canonicalize_flags,
                enforce_bounds_order=enforce_bounds_ord,
            )
            for g in genes
        ]
    else:
        genes_eff = list(genes)

    rules = [decode_gene(g, state_count=len(states)) for g in genes_eff]

    rid_by_cond: Dict[int, int] = {}
    for i, r in enumerate(rules):
        rid_by_cond.setdefault(int(r.cond_current), i)

    active_hits: Optional[List[bool]] = [False] * len(rules) if collect_activity else None

    max_steps = int(machine_cfg.get("max_steps", 120))
    for _ in range(max_steps):
        # snapshot prior state for this step
        for u in list(G.nodes()):
            G.nodes[u]["prev_state_id"] = int(G.nodes[u].get("state_id"))

        stepped = _apply_rules_once(
            G,
            rules,
            machine_cfg=machine_cfg,
            active_hits=active_hits,
            rid_by_cond=rid_by_cond,
        )
        if not stepped:
            break
        if machine_cfg.get("max_vertices", 0) and G.number_of_nodes() >= int(machine_cfg["max_vertices"]):
            break

    if collect_activity:
        return G, (active_hits or [False] * len(rules))
    return G





# =======================
# Checkpoint writers
# =======================
def _genes_to_hex(genes: List[int]) -> List[str]:
    return [f"{g:016x}" for g in genes]


def _genes_to_yaml_rules(
    genes: List[int],
    states: List[str],
    *,
    full_condition: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert encoded genes to YAML-rule dicts.
    If full_condition=True, include a 'condition_meta' block populated from decode_gene.
    """
    out: List[Dict[str, Any]] = []
    n_states = max(1, len(states))
    for g in genes:
        r = decode_gene(g, state_count=n_states)

        cur = states[int(r.cond_current) % n_states]
        op_name = r.op_kind.name
        operand = None
        if getattr(r, "operand", None) is not None:
            operand = states[int(r.operand) % n_states]

        row: Dict[str, Any] = {"condition": {"current": cur}, "op": {"kind": op_name}}
        if operand is not None:
            row["op"]["operand"] = operand

        if full_condition:
            # decode_gene exposes semantic fields; map to human-readable YAML
            prior = getattr(r, "prior", None)
            if prior is not None:
                prior = states[int(prior) % n_states]

            row["condition_meta"] = {
                "prior": prior,  # state label or null
                "conn_ge": getattr(r, "conn_ge", None),
                "conn_le": getattr(r, "conn_le", None),
                "parents_ge": getattr(r, "parents_ge", None),
                "parents_le": getattr(r, "parents_le", None),
            }

        out.append(row)
    return out

def _activity_scheme(mask: Sequence[bool]) -> str:
    """
    Compact, legacy-like visualization:
    - numbers for consecutive inactive runs
    - 'x' for each active gene
    Example: [F,F,T,T,F] -> '2xx1'
    """
    if not mask:
        return ""
    parts: List[str] = []
    inactive_run = 0
    for m in mask:
        if m:
            if inactive_run > 0:
                parts.append(str(inactive_run))
                inactive_run = 0
            parts.append("x")
        else:
            inactive_run += 1
    if inactive_run > 0:
        parts.append(str(inactive_run))
    return "".join(parts)

def _activity_to_yaml(mask: Optional[Sequence[bool]]) -> Optional[Dict[str, Any]]:
    if mask is None:
        return None
    mask_list = [bool(x) for x in mask]
    return {
        "mask": mask_list,
        "active_count": int(sum(mask_list)),
        "scheme": _activity_scheme(mask_list),
    }


def _write_checkpoint_best(
    ckpt_dir: Path,
    genes: List[int],
    fitness: float,
    fmt: str,
    states: List[str],
    *,
    full_condition: bool = False,
    graph_summary: Optional[Dict[str, Any]] = None,
    activity: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Write the 'best' artifact as YAML or JSON.
    YAML: { fitness, rules[, graph_summary][, activity] }
    JSON: { fitness, genes[, graph_summary] } (debug; no activity here by default)
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {}

    if fmt == "yaml":
        try:
            import yaml
            data = {
                "fitness": float(fitness),
                "rules": _genes_to_yaml_rules(genes, states, full_condition=full_condition),
            }
            if graph_summary:
                data["graph_summary"] = graph_summary
            if activity:
                data["activity"] = activity

            p = ckpt_dir / "best.yaml"
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            out["best"] = str(p)
            return out
        except Exception:
            fmt = "json"

    # fallback / JSON path
    payload = {"fitness": float(fitness), "genes": _genes_to_hex(genes)}
    if graph_summary:
        payload["graph_summary"] = graph_summary
    p = ckpt_dir / "best.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    out["best"] = str(p)
    return out




def _graph_summary(G: nx.Graph, states: List[str]) -> Dict[str, Any]:
    """
    Build a compact summary identical in shape to run_gum:
    { "edges": int, "nodes": int, "states_count": {label: count, ...} }
    """
    counts = Counter(states[int(G.nodes[n].get("state_id", 0))] for n in G.nodes())
    return {
        "edges": int(G.number_of_edges()),
        "nodes": int(G.number_of_nodes()),
        "states_count": dict(counts),
    }



def _render_best_png_inproc(
    ckpt_dir: Path,
    G: nx.Graph,
    *,
    out_name: str = "best.png",
) -> Optional[str]:
    """
    Render an already-evolved graph G to PNG (no subprocess).
    Visuals aligned with run_gum:
      - black background
      - per-state 16-tone color wheel (via node 'state_id')
      - edges-only by default; if no edges, draw tiny colored dots
    """
    import colorsys
    import matplotlib.pyplot as plt

    out_png = ckpt_dir / out_name

    # ---- layout ----
    try:
        pos = nx.spring_layout(G, seed=42)
    except Exception:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.circular_layout(G)

    # ---- color mapping by state_id ----
    def state_color(sid: int):
        hue = (int(sid) % 16) / 16.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        return (r, g, b)

    node_colors = {
        n: state_color(int(G.nodes[n].get("state_id", 0)))
        for n in G.nodes()
    }

    # ---- figure / axes (black bg) ----
    fig = plt.figure(figsize=(6, 6), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    N = max(1, G.number_of_nodes())
    E = G.number_of_edges()

    # ---- edges-only; fallback to tiny dots if no edges ----
    if E > 0:
        edges = list(G.edges())
        edge_colors = [
            (
                (node_colors[u][0] + node_colors[v][0]) / 2.0,
                (node_colors[u][1] + node_colors[v][1]) / 2.0,
                (node_colors[u][2] + node_colors[v][2]) / 2.0,
            )
            for (u, v) in edges
        ]
        edge_width = max(0.6, 1.8 / max(1.0, (N ** 0.5) / 6.0))
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_width, ax=ax)
    else:
        dot_size = max(6.0, 80.0 / (N ** 0.5))
        nx.draw_networkx_nodes(G, pos, node_size=dot_size, node_color=[node_colors[n] for n in G.nodes()], ax=ax)

    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(out_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return str(out_png)



def _write_checkpoint_pop(ckpt_dir: Path, pop: List[List[int]], fits: List[float], mode: str) -> Optional[str]:
    if mode not in ("best", "all"):
        return None
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / ("pop_best.jsonl" if mode == "best" else "pop_all.jsonl")
    order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
    with open(p, "w", encoding="utf-8") as f:
        if mode == "best" and order:
            i = order[0]
            json.dump({"fitness": float(fits[i]), "genes": _genes_to_hex(pop[i])}, f)
            f.write("\n")
        else:
            for i in order:
                json.dump({"fitness": float(fits[i]), "genes": _genes_to_hex(pop[i])}, f)
                f.write("\n")
    return str(p)


def _write_checkpoint_epoch(ckpt_dir: Path, gen: int, pop: List[List[int]], fits: List[float]) -> str:
    p = ckpt_dir / f"epoch_{gen}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({
            "generation": gen,
            "population": [{"fitness": float(fits[i]), "genes": _genes_to_hex(pop[i])} for i in range(len(pop))]
        }, f, indent=2)
    return str(p)


# ===================================
# Parallel eval: worker context
# ===================================
_WFIT = None
_WSTATES = None
_WMCFG = None

def _init_worker(fitness, states, machine_cfg):
    global _WFIT, _WSTATES, _WMCFG
    _WFIT = fitness
    _WSTATES = states
    _WMCFG = machine_cfg

def _eval_only(genes: List[int]) -> float:
    G = simulate_genome(genes, states=_WSTATES, machine_cfg=_WMCFG)
    return float(_WFIT.score(G))


def _select_parents(pop, k, method: str, rng: random.Random, tourn_k: int):
    m = method.lower()
    if m == "elite":
        return tools.selBest(pop, k)
    elif m == "roulette":
        return tools.selRoulette(pop, k)
    elif m == "rank":
        # linear rank weights: n..1
        order = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
        weights = list(reversed(range(1, len(pop) + 1)))  # 1..n (ascending)
        # align to 'order' (largest weight for best)
        weights = list(range(len(pop), 0, -1))
        return rng.choices(order, weights=weights, k=k)
    else:
        return tools.selTournament(pop, k=k, tournsize=tourn_k)



# ======================
# Evolve loop
# ======================
def evolve(
    *,
    fitness,                         # instantiated fitness object (TriangleMesh, HexMesh, BySample, ...)
    machine_cfg: Dict[str, Any],     # see MachineConfig in engine/config.py
    ga_cfg: Dict[str, Any],
    states: List[str],
    seed: int,
    n_workers: int,
    checkpoint_cfg: Dict[str, Any],
    run_dir: Path,
    progress: bool = False,
) -> Dict[str, Any]:
    r = random.Random(seed)
    state_count = len(states)
    min_len = int(ga_cfg.get("min_len", 1))
    max_len = int(ga_cfg.get("max_len", 64))
    init_len = int(ga_cfg.get("init_len", 6))


    # DEAP setup
    try:
        creator.FitnessMax  # type: ignore[attr-defined]
    except Exception:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.Individual  # type: ignore[attr-defined]
    except Exception:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("rand_gene", random_gene, state_count=state_count, rng=r)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.rand_gene, n=init_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # operators
    def cx(ind1, ind2):
        L1, L2 = len(ind1), len(ind2)
        if L1 == 0 or L2 == 0:
            return ind1, ind2
        # Robust cuts: if a parent has length 1, use cut=0 (no-op on that side).
        c1 = (r.randrange(1, L1) if L1 > 1 else 0)
        c2 = (r.randrange(1, L2) if L2 > 1 else 0)

        child1 = ind1[:c1] + ind2[c2:]
        child2 = ind2[:c2] + ind1[c1:]

        # clamp to max_len (like legacy: min(max_len, c + tail_len))
        if len(child1) > max_len:
            child1 = child1[:max_len]
        if len(child2) > max_len:
            child2 = child2[:max_len]

        # splice active masks if present
        m1 = getattr(ind1, "active_mask", [False]*L1)
        m2 = getattr(ind2, "active_mask", [False]*L2)
        child1_mask = (m1[:c1] + m2[c2:])[:len(child1)]
        child2_mask = (m2[:c2] + m1[c1:])[:len(child2)]

        ind1[:] = child1
        ind2[:] = child2
        setattr(ind1, "active_mask", child1_mask)
        setattr(ind2, "active_mask", child2_mask)
        return ind1, ind2

    toolbox.register("mate", cx)


    mutate = make_mutate_fn(
        state_count=state_count,
        min_len=min_len,
        max_len=max_len,
        structural_cfg=ga_cfg.get("structural", {}),
        field_cfg=ga_cfg.get("field", {}),
        active_cfg=ga_cfg.get("active", {}),
        passive_cfg=ga_cfg.get("passive", {}),
        structuralx_cfg=ga_cfg.get("structuralx", {}),
        rng=r,
    )

    toolbox.register("mutate", mutate)

    # selection policy (C# alignment default = rank)
    sel_cfg = ga_cfg.get("selection", {}) or {}
    selection_method = str(sel_cfg.get("method", "rank"))
    random_ratio = float(sel_cfg.get("random_ratio", 0.0))

    # Build and register selector into toolbox
    selector_fn = _make_selector(selection_method, tournament_k=int(ga_cfg.get("tournament_k", 3)),
                                 random_ratio=random_ratio, rng=r)
    toolbox.register("select", selector_fn)

    def evaluate(individual: List[int]) -> Tuple[float]:
        # simulate genome -> graph, also collect which rules truly effected changes
        G, active_mask = simulate_genome(
            individual,
            states=states,
            machine_cfg=machine_cfg,
            collect_activity=True,
        )
        # stash mask on the individual for the next mutation call
        setattr(individual, "active_mask", list(active_mask))
        # score
        score = float(fitness.score(G))
        return (score,)



    toolbox.register("evaluate", evaluate)

    # population
    pop_size = int(ga_cfg.get("pop_size", 40))
    cx_pb = float(ga_cfg.get("cx_pb", 0.7))
    mut_pb = float(ga_cfg.get("mut_pb", 0.3))
    ngen = int(ga_cfg.get("generations", 20))
    elitism = int(ga_cfg.get("elitism", 2))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(max(1, elitism))

    # --------- parallel map setup ----------
    global _WFIT, _WSTATES, _WMCFG
    _WFIT, _WSTATES, _WMCFG = fitness, states, machine_cfg

    pool = None
    if n_workers and n_workers > 0:
        try:
            ctx = get_context("spawn")
            pool = ctx.Pool(processes=int(n_workers), initializer=_init_worker,
                            initargs=(fitness, states, machine_cfg))
            map_fn = pool.map
        except Exception as e:
            print(f"[WARN] parallel init failed ({e}); falling back to serial.")
            map_fn = map
    else:
        map_fn = map

    # evaluate initial pop
    fits = list(map_fn(_eval_only, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = (fit,)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda xs: sum(xs) / max(1, len(xs)))
    stats.register("max", max)
    stats.register("min", min)

    ckpt = checkpoint_cfg or {}
    ckpt_dir = (run_dir / ckpt.get("out_dir", "checkpoints"))
    fmt = str(ckpt.get("fmt", "json")).lower()

    # parse checkpoint options
    save_best = bool(ckpt.get("save_best", True))
    save_last = bool(ckpt.get("save_last", True))
    save_every = int(ckpt.get("save_every", 0))
    save_population = str(ckpt.get("save_population", "best"))
    full_cond = bool(ckpt.get("export_full_condition_shape", False))
    save_best_png = bool(ckpt.get("save_best_png", False))    


    # initial checkpoints (generation 0)
    hof.update(pop)
    best0 = hof[0]    
    
    _write_checkpoint_best(ckpt_dir, list(best0), best0.fitness.values[0], fmt, states, full_condition=full_cond)


    if save_population in ("best", "all"):
        _write_checkpoint_pop(ckpt_dir, [list(ind) for ind in pop], [ind.fitness.values[0] for ind in pop], save_population)
    if save_every and save_every > 0:
        _write_checkpoint_epoch(ckpt_dir, 0, [list(ind) for ind in pop], [ind.fitness.values[0] for ind in pop])

    record0 = stats.compile(pop)
    best_len0 = len(hof[0]) if len(hof) else 0

    # --- progress bar setup ---
    pbar = None
    if progress and tqdm is not None:
        pbar = tqdm(total=ngen, desc="epochs", leave=True, dynamic_ncols=True)
        pbar.set_postfix(max=f"{record0['max']:.4f}",
                         avg=f"{record0['avg']:.4f}",
                         len=best_len0)
    elif progress:
        print(f"gen 0/{ngen}  max={record0['max']:.4f}  avg={record0['avg']:.4f}  len={best_len0}")

    # evolve
    for gen in range(1, ngen + 1):
        # elitism: copy best E
        elites = tools.selBest(pop, k=elitism)
        

        # offspring via variation
        offspring = toolbox.select(pop, k=pop_size - elitism)   # <— use our selector here
        offspring = list(map(toolbox.clone, offspring))
       

        # optional random immigrants
        k_off = len(offspring)

        # read from new config (ga.selection.random_ratio) with a legacy fallback (ga.random_selection_ratio)
        sel_cfg = ga_cfg.get("selection", {})
        rand_ratio = float(sel_cfg.get("random_ratio", ga_cfg.get("random_selection_ratio", 0.0)))
        rand_ratio = max(0.0, min(1.0, rand_ratio))

        n_rand = int(rand_ratio * k_off)
        if n_rand > 0:
            # Create fresh individuals using the registered DEAP constructors
            try:
                immigrants = list(toolbox.population(n=n_rand))
            except Exception:
                immigrants = [toolbox.individual() for _ in range(n_rand)]
            # Overwrite the first n_rand offspring slots (length preserved; deterministic under fixed seed)
            offspring[:n_rand] = immigrants



        
        # mate
        for i in range(1, len(offspring), 2):
            if r.random() < cx_pb:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # mutate
        for i in range(len(offspring)):
            if r.random() < mut_pb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # new population
        pop = elites + offspring

        # evaluate invalid
        invalid = [ind for ind in pop if not ind.fitness.valid]
        if invalid:
            newfits = list(map_fn(_eval_only, invalid))
            for ind, fit in zip(invalid, newfits):
                ind.fitness.values = (fit,)

        # update HOF & checkpoints
        hof.update(pop)
        record = stats.compile(pop)

        best_len = len(hof[0]) if len(hof) else 0

        if save_every and gen % save_every == 0:
            _write_checkpoint_epoch(ckpt_dir, gen,
                                    [list(ind) for ind in pop],
                                    [ind.fitness.values[0] for ind in pop])

        # progress update
        if pbar is not None:
            pbar.set_postfix(max=f"{record['max']:.4f}",
                             avg=f"{record['avg']:.4f}",
                             len=best_len)
            pbar.update(1)
        elif progress:
            print(f"gen {gen}/{ngen}  max={record['max']:.4f}  avg={record['avg']:.4f}  len={best_len}")

    if pbar is not None:
        pbar.close()

    # final checkpoints
    best = hof[0]
    paths = {}

    # NEW: single simulate_genome call – both graph and activity
    try:
        sim = simulate_genome(
            list(best),
            states=states,          # use the same states you pass into evolve(...)
            machine_cfg=machine_cfg,# use the same machine_cfg you pass into evolve(...)
            collect_activity=True
        )
        if isinstance(sim, tuple) and len(sim) == 2:
            G_best, mask = sim
        else:
            G_best, mask = sim, None
    except Exception:
        G_best, mask = None, None

    gsum = _graph_summary(G_best, states) if G_best is not None else None
    activity_yaml = _activity_to_yaml(mask)



    if save_best:
        paths.update(_write_checkpoint_best(
            ckpt_dir, list(best), best.fitness.values[0], fmt, states,
            full_condition=full_cond, graph_summary=gsum, activity=activity_yaml
        ))




    if save_last:
        last_path = ckpt_dir / "last.json"
        with open(last_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_fitness": float(best.fitness.values[0]),
                "best_genes": _genes_to_hex(list(best)),
                "pop_size": pop_size,
                "generations": ngen,
                "graph_summary": gsum,
            }, f, indent=2)
        paths["last"] = str(last_path)

    # optional in-process PNG (now using the already-simulated G_best)
    if save_best_png:
        png = _render_best_png_inproc(ckpt_dir, G_best)
        if png:
            paths["best_png"] = png


    # close the pool if any
    if 'pool' in locals() and pool is not None:
        pool.close()
        pool.join()

    return {
        "status": "ok",
        "best_fitness": float(best.fitness.values[0]),
        "best_length": len(best),
        "pop_size": pop_size,
        "generations": ngen,
        "graph_summary": gsum,
        "checkpoints": paths,
    }

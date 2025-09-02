# src/guca/ga/toolbox.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import networkx as nx
from deap import base, creator, tools
from multiprocessing import get_context

from guca.ga.encoding import (
    Rule, OpKind, encode_rule, decode_gene, random_gene, labels_to_state_maps
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


def _apply_rules_once(G: nx.Graph, rules: List[Rule], *, machine_cfg: Dict[str, Any]) -> bool:
    """
    Apply rules to all nodes in a deterministic order; return True if any change occurred.
    """
    # Unpack machine params
    max_vertices = int(machine_cfg.get("max_vertices", 0) or 0)
    ns = machine_cfg.get("nearest_search", {}) or {}
    max_depth = int(ns.get("max_depth", 2))
    tie_breaker = str(ns.get("tie_breaker", "stable"))
    connect_all = bool(ns.get("connect_all", False))

    changed = False
    to_add_nodes: List[Tuple[int, int]] = []   # (new_id, state_id)
    to_add_edges: List[Tuple[int, int]] = []
    to_remove_edges: List[Tuple[int, int]] = []
    to_remove_nodes: List[int] = []
    state_updates: List[Tuple[int, int]] = []

    # deterministic iteration
    for u in sorted(G.nodes()):
        cur_sid = int(G.nodes[u]["state_id"])
        # pick first matching rule by cond_current
        rule = next((r for r in rules if r.cond_current == cur_sid), None)
        if rule is None:
            continue

        op = rule.op_kind
        operand_sid = rule.operand if rule.operand is not None else None

        if op == OpKind.TurnToState:
            if operand_sid is not None and operand_sid != cur_sid:
                state_updates.append((u, operand_sid))

        elif op == OpKind.GiveBirth:
            # create a node but don't connect
            if operand_sid is not None:
                new_id = max(G.nodes()) + 1 if G.number_of_nodes() > 0 else 0
                to_add_nodes.append((new_id, operand_sid))

        elif op == OpKind.GiveBirthConnected:
            if operand_sid is not None:
                new_id = max(G.nodes()) + 1 if G.number_of_nodes() > 0 else 0
                to_add_nodes.append((new_id, operand_sid))
                to_add_edges.append((u, new_id))

        elif op == OpKind.TryToConnectWith:
            if operand_sid is not None:
                # connect to ALL nodes that have operand state and are not already neighbors
                targets = [v for v in G.nodes()
                           if v != u and G.nodes[v].get("state_id") == operand_sid and not G.has_edge(u, v)]
                for v in targets:
                    to_add_edges.append((u, v))

        elif op == OpKind.TryToConnectWithNearest:
            if operand_sid is not None:
                cands = _nearest_candidates(G, u, operand_sid, max_depth=max_depth, tie_breaker=tie_breaker, connect_all=connect_all)
                for v in cands:
                    if not G.has_edge(u, v):
                        to_add_edges.append((u, v))

        elif op == OpKind.DisconnectFrom:
            if operand_sid is not None:
                targets = [v for v in list(G.neighbors(u)) if G.nodes[v].get("state_id") == operand_sid]
                for v in targets:
                    to_remove_edges.append((u, v))

        elif op == OpKind.Die:
            to_remove_nodes.append(u)

    # apply removals first
    for (u, v) in to_remove_edges:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            changed = True
    if to_remove_nodes:
        for u in sorted(set(to_remove_nodes), reverse=True):
            if G.has_node(u):
                G.remove_node(u)
                changed = True

    # add nodes (guard max_vertices)
    for (nid, sid) in to_add_nodes:
        if max_vertices and G.number_of_nodes() >= max_vertices:
            break
        if not G.has_node(nid):
            G.add_node(nid, state_id=int(sid))
            changed = True

    # add edges
    for (u, v) in to_add_edges:
        if G.has_node(u) and G.has_node(v) and not G.has_edge(u, v):
            G.add_edge(u, v)
            changed = True

    # update states
    for (u, sid) in state_updates:
        if G.has_node(u):
            if int(G.nodes[u]["state_id"]) != int(sid):
                G.nodes[u]["state_id"] = int(sid)
                changed = True

    return changed


def simulate_genome(genes: List[int], *, states: List[str], machine_cfg: Dict[str, Any]) -> nx.Graph:
    """
    Minimal in-process simulator for GA evaluation.
    Starts from one node with machine.start_state if init graph is omitted.
    """
    # initial graph: single node with start_state
    start_label = str(machine_cfg.get("start_state", "A"))
    _, inv = labels_to_state_maps(states)
    try:
        start_sid = inv.index(start_label)
    except ValueError:
        start_sid = 0  # fallback

    G = nx.Graph()
    G.add_node(0, state_id=int(start_sid))

    # decode rules (first-match semantics by cond_current)
    rules = [decode_gene(g, state_count=len(states)) for g in genes]
    # ensure stable order: sort by cond then op id to create a predictable priority
    rules.sort(key=lambda r: (int(r.cond_current), int(r.op_kind)))

    max_steps = int(machine_cfg.get("max_steps", 120))
    for _ in range(max_steps):
        if not _apply_rules_once(G, rules, machine_cfg=machine_cfg):
            break  # converged / no changes

        if machine_cfg.get("max_vertices", 0) and G.number_of_nodes() >= int(machine_cfg["max_vertices"]):
            break

    return G


# =======================
# Checkpoint writers
# =======================
def _genes_to_hex(genes: List[int]) -> List[str]:
    return [f"{g:016x}" for g in genes]


def _genes_to_yaml_rules(genes: List[int], states: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for g in genes:
        r = decode_gene(g, state_count=len(states))
        cur = states[int(r.cond_current)]
        op = r.op_kind.name
        operand = None if r.operand is None else states[int(r.operand)]
        row = {"condition": {"current": cur}, "op": {"kind": op}}
        if operand is not None:
            row["op"]["operand"] = operand
        out.append(row)
    return out


def _write_checkpoint_best(ckpt_dir: Path, genes: List[int], fitness: float, fmt: str, states: List[str]) -> Dict[str, str]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    if fmt == "yaml":
        try:
            import yaml
            data = {"fitness": float(fitness), "rules": _genes_to_yaml_rules(genes, states)}
            p = ckpt_dir / "best.yaml"
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            out["best"] = str(p)
            return out
        except Exception as e:
            print(f"[WARN] YAML requested but unavailable ({e}); falling back to JSON.")

    p = ckpt_dir / "best.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"fitness": float(fitness), "genes": _genes_to_hex(genes)}, f, indent=2)
    out["best"] = str(p)
    return out


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
        c1, c2 = splice_cx(ind1, ind2, rng=r, unaligned=True)
        ind1[:] = c1
        ind2[:] = c2
        return ind1, ind2

    mutate = make_mutate_fn(
        state_count=state_count,
        min_len=min_len,
        max_len=max_len,
        structural_cfg=ga_cfg.get("structural", {}),
        field_cfg=ga_cfg.get("field", {}),
        rng=r,
    )

    toolbox.register("mate", cx)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=int(ga_cfg.get("tournament_k", 3)))

    # evaluation wrapper (DEAP API); real work in _eval_only
    def evaluate(individual: List[int]) -> Tuple[float]:
        return (_eval_only(individual),)

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
    save_best = bool(ckpt.get("save_best", True))
    save_last = bool(ckpt.get("save_last", True))
    save_every = int(ckpt.get("save_every", 0))
    save_population = str(ckpt.get("save_population", "best"))

    # initial checkpoints (generation 0)
    hof.update(pop)
    best0 = hof[0]
    _write_checkpoint_best(ckpt_dir, list(best0), best0.fitness.values[0], fmt, states)
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
        offspring = tools.selTournament(pop, k=pop_size - elitism, tournsize=int(ga_cfg.get("tournament_k", 3)))
        offspring = list(map(toolbox.clone, offspring))

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
    if save_best:
        paths.update(_write_checkpoint_best(ckpt_dir, list(best), best.fitness.values[0], fmt, states))
    if save_last:
        last_path = ckpt_dir / "last.json"
        with open(last_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_fitness": float(best.fitness.values[0]),
                "best_genes": _genes_to_hex(list(best)),
                "pop_size": pop_size,
                "generations": ngen
            }, f, indent=2)
        paths["last"] = str(last_path)

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
        "checkpoints": paths,
    }

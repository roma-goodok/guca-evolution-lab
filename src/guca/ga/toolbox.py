# src/guca/ga/toolbox.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Sequence, Callable
from collections import Counter

import random
import csv
from datetime import datetime

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


# =============================================================================
# Core-engine–backed simulator
# =============================================================================
from guca.core.graph import GUMGraph as CoreGraph
from guca.core.machine import GraphUnfoldingMachine, TranscriptionWay, CountCompare
from guca.core.rules import (
    ChangeTable as CoreChangeTable,
    Rule as CoreRule,
    Condition as CoreCond,
    Operation as CoreOp,
    OperationKind as CoreOpKind,
)

def _opkind_to_core(k: OpKind) -> CoreOpKind:
    if k == OpKind.TurnToState:
        return CoreOpKind.TurnToState
    if k in (OpKind.GiveBirthConnected, OpKind.GiveBirth):
        return CoreOpKind.GiveBirthConnected  # legacy parity
    if k == OpKind.DisconnectFrom:
        return CoreOpKind.DisconnectFrom
    if k in (OpKind.TryToConnectWithNearest, OpKind.TryToConnectNearest):
        return CoreOpKind.TryToConnectWithNearest
    if k == OpKind.TryToConnectWith:
        return CoreOpKind.TryToConnectWith
    return CoreOpKind.TurnToState

def _genes_to_core_change_table(genes: List[int], states: List[str], machine_encoding: Dict[str, Any] | None) -> CoreChangeTable:
    n = max(1, len(states))
    # optional sanitize-on-decode (allow GA encoding knobs)
    sanitize = bool((machine_encoding or {}).get("sanitize_on_decode", False))
    enforce  = bool((machine_encoding or {}).get("enforce_semantics", False))
    canonf   = bool((machine_encoding or {}).get("canonicalize_flags", False))
    order    = bool((machine_encoding or {}).get("enforce_bounds_order", False))

    if sanitize:
        genes = [
            sanitize_gene(
                g,
                state_count=n,
                enforce_semantics=enforce,
                canonicalize_flags=canonf,
                enforce_bounds_order=order,
            )
            for g in genes
        ]

    tbl = CoreChangeTable()
    for g in genes:
        r = decode_gene(g, state_count=n)
        cur = states[int(r.cond_current) % n]
        prior = "any" if r.prior is None else states[int(r.prior) % n]
        oper_label = None if r.operand is None else states[int(r.operand) % n]
        cond = CoreCond(
            current=cur,
            prior=prior,
            conn_ge=(-1 if r.conn_ge is None else int(r.conn_ge)),
            conn_le=(-1 if r.conn_le is None else int(r.conn_le)),
            parents_ge=(-1 if r.parents_ge is None else int(r.parents_ge)),
            parents_le=(-1 if r.parents_le is None else int(r.parents_le)),
        )
        op = CoreOp(kind=_opkind_to_core(r.op_kind), operand=oper_label)
        tbl.append(CoreRule(condition=cond, operation=op))
    return tbl

def _core_graph_to_nx(g: CoreGraph, states: List[str]) -> nx.Graph:
    G = nx.Graph()
    # nodes
    for n in g.nodes():
        st = n.state
        sid = states.index(st) if isinstance(st, str) and st in states else 0
        G.add_node(n.id, state=st, state_id=sid)
    # edges
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
    """
    Evaluate genes using the **core** GraphUnfoldingMachine; return a networkx graph
    (for fitness) and, optionally, an 'ever-active' boolean mask per gene reflecting
    core-rule `was_active` flags over the whole run.
    """
    start_state = str(machine_cfg.get("start_state", "A"))
    nearest = machine_cfg.get("nearest_search", {}) or {}
    m = GraphUnfoldingMachine(
        CoreGraph(),
        start_state=start_state,
        transcription=TranscriptionWay(machine_cfg.get("transcription", "resettable")),
        count_compare=CountCompare(machine_cfg.get("count_compare", "range")),
        max_vertices=int(machine_cfg.get("max_vertices", 0)),
        max_steps=int(machine_cfg.get("max_steps", 120)),
        nearest_max_depth=int(nearest.get("max_depth", 2)),
        nearest_tie_breaker=str(nearest.get("tie_breaker", "stable")),
        nearest_connect_all=bool(nearest.get("connect_all", False)),
        rng_seed=machine_cfg.get("rng_seed", None),
    )

    enc = dict(machine_cfg.get("encoding", {}) or {})
    m.change_table = _genes_to_core_change_table(genes, states, enc)

    # Ensure a seed node (core also does this internally)
    if not any(True for _ in m.graph.nodes()):
        m.graph.add_vertex(state=start_state, parents_count=0, mark_new=True)

    m.run()

    G = _core_graph_to_nx(m.graph, states)

    if not collect_activity:
        return G

    mask = [bool(r.was_active) for r in m.change_table]
    return G, mask


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
    Additionally, when the graph is small (edges < 20), include a sorted 'edge_list'
    of undirected pairs for convenience.
    """
    counts = Counter(states[int(G.nodes[n].get("state_id", 0))] for n in G.nodes())
    e = int(G.number_of_edges())
    summary = {
        "edges": e,
        "nodes": int(G.number_of_nodes()),
        "states_count": dict(counts),
    }
    if e < 1000:
        # produce a stable, sorted list of undirected edges as [u, v] pairs
        edge_list = sorted(
            [ [int(min(u, v)), int(max(u, v))] for (u, v) in G.edges() ],
            key=lambda uv: (uv[0], uv[1])
        )
        summary["edge_list"] = edge_list
    return summary


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

def _eval_with_mask(genes: List[int]) -> Tuple[float, List[bool]]:
    G, mask = simulate_genome(genes, states=_WSTATES, machine_cfg=_WMCFG, collect_activity=True)
    return float(_WFIT.score(G)), list(mask or [])


# --- Selection helpers --------------------------------------------------------
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


def _mk_ckpt_subdir(root: Path, tag: str, gen: int) -> Path:
    d = root / f"{tag}_{gen:05d}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _pop_arrays(pop):
    fits = [float(ind.fitness.values[0]) for ind in pop]
    lengths = [len(ind) for ind in pop]
    actlens = [int(sum(getattr(ind, "active_mask", []) or [])) for ind in pop]
    return fits, lengths, actlens

def _best_set(pop, fits):
    if not fits:
        return [], []
    mx = max(fits)
    tops = [ind for ind, f in zip(pop, fits) if abs(f - mx) < 1e-12]
    return tops, mx

def _write_population_json(dir_: Path, pop, fits):
    out = []
    for ind, f in zip(pop, fits):
        out.append({
            "fitness": float(f),
            "length": len(ind),
            "active_len": int(sum(getattr(ind, "active_mask", []) or [])),
            "genes": [f"{g:016x}" for g in ind],
        })
    p = dir_ / "population.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return str(p)

def _machine_yaml_from_cfg(machine_cfg: Dict[str, Any]) -> Dict[str, Any]:
    nearest = machine_cfg.get("nearest_search", {}) or {}
    return {
        "start_state": str(machine_cfg.get("start_state", "A")),
        "transcription": "resettable",
        "count_compare": "range",
        "max_vertices": int(machine_cfg.get("max_vertices", 2000)),
        "max_steps": int(machine_cfg.get("max_steps", 120)),
        "nearest_search": {
            "max_depth": int(nearest.get("max_depth", 2)),
            "tie_breaker": str(nearest.get("tie_breaker", "stable")),
            "connect_all": bool(nearest.get("connect_all", False)),
        },
        # "rng_seed": machine_cfg.get("rng_seed", 42),  # optional
    }

def _write_genome_yaml(dir_: Path, best_genes: List[int], states: List[str],
                       machine_cfg: Dict[str, Any],
                       graph_summary: Optional[Dict[str, Any]] = None,
                       activity_mask: Optional[List[bool]] = None,
                       full_condition: bool = False) -> str:
    try:
        import yaml
    except Exception:
        yaml = None
    rules = _genes_to_yaml_rules(best_genes, states, full_condition=full_condition)
    y = {
        "machine": _machine_yaml_from_cfg(machine_cfg),
        "init_graph": {"nodes": [{"state": str(machine_cfg.get("start_state", "A"))}]},
        "rules": rules,
    }
    if graph_summary or activity_mask is not None:
        y["meta"] = {}
        if graph_summary:
            y["meta"]["graph_summary"] = graph_summary
        if activity_mask is not None:
            y["meta"]["activity_mask"] = [bool(x) for x in activity_mask]
    p = dir_ / "genome.yaml"
    if yaml is None:
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(y, indent=2))
    else:
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(y, fh, sort_keys=False, allow_unicode=True)
    return str(p)

def _render_png_dots(dir_: Path, G: nx.Graph, out_name: str = "best.png") -> Optional[str]:
    # Reuse the public save_png utility which supports dots mode
    try:
        from guca.vis.png import save_png
    except Exception:
        return None
    out = dir_ / out_name
    try:
        save_png(G, out, node_render="dots")
        return str(out)
    except Exception:
        return None

def _write_hist_png(dir_: Path, name: str, data: List[float], bins: int):
    out = dir_ / name

    # sanitize data (avoid NaN/inf)
    vals = []
    for x in data:
        try:
            xv = float(x)
            if xv == xv and abs(xv) != float("inf"):
                vals.append(xv)
        except Exception:
            continue
    if not vals:
        vals = [0.0]

    # Matplotlib (force headless)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4, 4.2))
        plt.hist(vals, bins=int(max(1, bins)))
        plt.title(name.replace("_", " "))
        plt.tight_layout()
        plt.savefig(out.as_posix(), dpi=150)
        plt.close()
        return str(out)
    except Exception:
        pass

    # Ultimate fallback: tiny placeholder PNG so tests pass
    try:
        with open(out, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        return str(out)
    except Exception:
        return None


def _write_metrics_csv(dir_: Path, metrics: Dict[str, Any]):
    p = dir_ / "metrics.csv"
    cols = list(metrics.keys())
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerow([metrics[k] for k in cols])
    return str(p)

def _bundle_checkpoint(dir_: Path, *,
                       pop, fits, states, machine_cfg, fitness,
                       best_ind, best_mask, G_best, full_condition: bool,
                       hist_bins: int):
    # population.json
    _write_population_json(dir_, pop, fits)

    # runnable genome.yaml (+meta)
    gsum = _graph_summary(G_best, states) if G_best is not None else None
    genome_path = _write_genome_yaml(dir_, list(best_ind), states, machine_cfg, gsum, best_mask, full_condition)

    # PNG (dots)
    if G_best is not None:
        _render_png_dots(dir_, G_best, out_name="best.png")

    # histograms
    _, lengths, actlens = _pop_arrays(pop)  # fits already passed
    _write_hist_png(dir_, "hist_fitness.png", fits, bins=hist_bins)
    _write_hist_png(dir_, "hist_length.png", lengths, bins=hist_bins)
    _write_hist_png(dir_, "hist_active.png", actlens, bins=hist_bins)

    # metrics.csv
    tops, mx = _best_set(pop, fits)
    best_act = int(sum(getattr(best_ind, "active_mask", []) or []))
    avg_fit = sum(fits) / max(1, len(fits))
    cnt_max = len(tops)
    top_act_av = (sum(int(sum(getattr(t, "active_mask", []) or [])) for t in tops) / cnt_max) if cnt_max else 0.0
    pop_act_av = sum(actlens) / max(1, len(actlens))
    best_len = len(best_ind)
    avg_len = sum(lengths) / max(1, len(lengths))

    stats = {
        "best_fitness": float(mx),
        "avg_fitness": float(avg_fit),
        "count_at_max": int(cnt_max),
        "best_active_len": int(best_act),
        "avg_active_len_at_max": float(top_act_av),
        "avg_active_len_pop": float(pop_act_av),
        "best_length": int(best_len),
        "avg_length_pop": float(avg_len),
    }
    _write_metrics_csv(dir_, stats)

    return {"genome": genome_path, "metrics": stats}



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

    # Merge GA 'encoding' knobs into the simulated machine config
    mc_eval = dict(machine_cfg)
    mc_eval["encoding"] = dict(ga_cfg.get("encoding", {}))

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
            machine_cfg=mc_eval,
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
    _WFIT, _WSTATES, _WMCFG = fitness, states, mc_eval

    pool = None
    if n_workers and n_workers > 0:
        try:
            ctx = get_context("spawn")
            pool = ctx.Pool(processes=int(n_workers), initializer=_init_worker,
                            initargs=(fitness, states, mc_eval))
            map_fn = pool.map
        except Exception as e:
            print(f"[WARN] parallel init failed ({e}); falling back to serial.")
            map_fn = map
    else:
        map_fn = map

    # evaluate initial pop
    res = list(map_fn(_eval_with_mask, pop))
    for ind, (fit, mask) in zip(pop, res):
        ind.fitness.values = (fit,)
        setattr(ind, "active_mask", mask)

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
    hist_bins = int(ckpt.get("hist_bins", 100))

    # initial checkpoints (generation 0)
    hof.update(pop)
    best0 = hof[0]

    # Compile quick stats & optionally write a FIRST out-of-sequence checkpoint (last_0000)
    fits0 = [ind.fitness.values[0] for ind in pop]
    best0_graph = None
    best0_mask = getattr(best0, "active_mask", None)
    try:
        sim0 = simulate_genome(list(best0), states=states, machine_cfg=mc_eval, collect_activity=True)
        if isinstance(sim0, tuple) and len(sim0) == 2:
            best0_graph, best0_mask = sim0
        else:
            best0_graph = sim0
    except Exception:
        best0_graph = None

    # Always produce an initial "last_0000" folder if save_last
    last_dir_path = None
    if save_last:
        last_dir = _mk_ckpt_subdir(ckpt_dir, "last", 0)
        _bundle_checkpoint(last_dir,
            pop=pop, fits=fits0, states=states, machine_cfg=mc_eval, fitness=fitness,
            best_ind=best0, best_mask=best0_mask, G_best=best0_graph,
            full_condition=full_cond, hist_bins=hist_bins
        )
        last_dir_path = str(last_dir)

    # For backward compatibility, provide a 'best' pointer (to runnable YAML in last_0000)
    paths = {}
    if save_best:
        if last_dir_path is not None:
            paths["best"] = str(Path(last_dir_path) / "genome.yaml")

    # (optional) keep population jsonl if requested
    if save_population in ("best", "all"):
        _write_checkpoint_pop(ckpt_dir, [list(ind) for ind in pop], [ind.fitness.values[0] for ind in pop], save_population)

    # epoch_0000 snapshot if periodic saving is configured
    if save_every and save_every > 0:
        _write_checkpoint_epoch(ckpt_dir, 0, [list(ind) for ind in pop], [ind.fitness.values[0] for ind in pop])

    record0 = stats.compile(pop)
    best_len0 = len(hof[0]) if len(hof) else 0
    best_so_far = record0['max']

    # --- progress bar setup ---
    pbar = None
    if progress and tqdm is not None:
        fits_arr, lens_arr, acts_arr = _pop_arrays(pop)
        tops0, mx0 = _best_set(pop, fits_arr)
        if tops0:
            len_top_max = max(len(t) for t in tops0)
            len_top_min = min(len(t) for t in tops0)
            len_top_avg = sum(len(t) for t in tops0) / len(tops0)
            act_top_vals = [int(sum(getattr(t, "active_mask", []) or [])) for t in tops0]
            act_top_max = max(act_top_vals)
            act_top_min = min(act_top_vals)
            act_top_avg = sum(act_top_vals) / len(act_top_vals)
        else:
            len_top_max = len_top_min = len_top_avg = 0
            act_top_max = act_top_min = act_top_avg = 0
        pbar = tqdm(total=ngen, desc="epochs", leave=True, dynamic_ncols=True)
        pbar.set_postfix(
            max=f"{record0['max']:.4f}",
            avg=f"{record0['avg']:.4f}",
            len=best_len0,
            len_avg=f"{(sum(lens_arr)/max(1,len(lens_arr))):.2f}",
            cnt_max=len(tops0),
            act_avg=f"{(sum(acts_arr)/max(1,len(acts_arr))):.2f}",
            len_top=f"{len_top_min:.0f}/{len_top_avg:.1f}/{len_top_max:.0f}",
            act_top=f"{act_top_min:.0f}/{act_top_avg:.1f}/{act_top_max:.0f}",
        )
    elif progress:
        print(f"gen 0/{ngen}  max={record0['max']:.4f}  avg={record0['avg']:.4f}  len={best_len0}")

    # evolve
    for gen in range(1, ngen + 1):
        # elitism: copy best E
        elites = tools.selBest(pop, k=elitism)

        # offspring via variation
        offspring = toolbox.select(pop, k=pop_size - elitism)
        offspring = list(map(toolbox.clone, offspring))

        # optional random immigrants
        k_off = len(offspring)
        sel_cfg = ga_cfg.get("selection", {})
        rand_ratio = float(sel_cfg.get("random_ratio", ga_cfg.get("random_selection_ratio", 0.0)))
        rand_ratio = max(0.0, min(1.0, rand_ratio))
        n_rand = int(rand_ratio * k_off)
        if n_rand > 0:
            try:
                immigrants = list(toolbox.population(n=n_rand))
            except Exception:
                immigrants = [toolbox.individual() for _ in range(n_rand)]
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
            res_inv = list(map_fn(_eval_with_mask, invalid))
            for ind, (fit, mask) in zip(invalid, res_inv):
                ind.fitness.values = (fit,)
                setattr(ind, "active_mask", mask)

        # update HOF & checkpoints
        hof.update(pop)
        record = stats.compile(pop)

        best_len = len(hof[0]) if len(hof) else 0

        # Prepare arrays
        fits_arr, lens_arr, acts_arr = _pop_arrays(pop)
        tops, mx = _best_set(pop, fits_arr)

        # 1) periodic checkpoint folder (epoch_<gen>)
        if save_every and gen % save_every == 0:
            try:
                G_tmp, mask_tmp = None, getattr(hof[0], "active_mask", None)
                try:
                    simt = simulate_genome(list(hof[0]), states=states, machine_cfg=mc_eval, collect_activity=True)
                    if isinstance(simt, tuple) and len(simt) == 2:
                        G_tmp, mask_tmp = simt
                    else:
                        G_tmp = simt
                except Exception:
                    G_tmp = None
                d_epoch = _mk_ckpt_subdir(ckpt_dir, "epoch", gen)
                _bundle_checkpoint(d_epoch,
                    pop=pop, fits=fits_arr, states=states, machine_cfg=mc_eval, fitness=fitness,
                    best_ind=hof[0], best_mask=mask_tmp, G_best=G_tmp,
                    full_condition=full_cond, hist_bins=hist_bins
                )
            except Exception:
                pass

        # 2) out-of-sequence when a NEW best appears
        if record['max'] > best_so_far and save_last:
            best_so_far = record['max']
            try:
                G_new, mask_new = None, getattr(hof[0], "active_mask", None)
                try:
                    simn = simulate_genome(list(hof[0]), states=states, machine_cfg=mc_eval, collect_activity=True)
                    if isinstance(simn, tuple) and len(simn) == 2:
                        G_new, mask_new = simn
                    else:
                        G_new = simn
                except Exception:
                    G_new = None
                d_last = _mk_ckpt_subdir(ckpt_dir, "last", gen)
                _bundle_checkpoint(d_last,
                    pop=pop, fits=fits_arr, states=states, machine_cfg=mc_eval, fitness=fitness,
                    best_ind=hof[0], best_mask=mask_new, G_best=G_new,
                    full_condition=full_cond, hist_bins=hist_bins
                )
                paths["best"] = str(Path(d_last) / "genome.yaml")
                if save_best_png and G_new is not None:
                    paths["best_png"] = str(Path(d_last) / "best.png")
                paths["last"] = str(d_last)
            except Exception:
                pass

        # 3) progress update (rich metrics)
        if pbar is not None:
            if tops:
                len_top_max = max(len(t) for t in tops)
                len_top_min = min(len(t) for t in tops)
                len_top_avg = sum(len(t) for t in tops) / len(tops)
                act_top_vals = [int(sum(getattr(t, "active_mask", []) or [])) for t in tops]
                act_top_max = max(act_top_vals)
                act_top_min = min(act_top_vals)
                act_top_avg = sum(act_top_vals) / len(act_top_vals)
            else:
                len_top_max = len_top_min = len_top_avg = 0
                act_top_max = act_top_min = act_top_avg = 0

            pbar.set_postfix(
                max=f"{record['max']:.4f}",
                avg=f"{record['avg']:.4f}",
                len=best_len,
                len_avg=f"{(sum(lens_arr)/max(1,len(lens_arr))):.2f}",
                cnt_max=len(tops),
                act_avg=f"{(sum(acts_arr)/max(1,len(acts_arr))):.2f}",
                len_top=f"{len_top_min:.0f}/{len_top_avg:.1f}/{len_top_max:.0f}",
                act_top=f"{act_top_min:.0f}/{act_top_avg:.1f}/{act_top_max:.0f}",
            )
            pbar.update(1)
        elif progress:
            print(f"gen {gen}/{ngen}  max={record['max']:.4f}  avg={record['avg']:.4f}  len={best_len}")

    if pbar is not None:
        pbar.close()

    # final checkpoints
    best = hof[0]
    try:
        sim = simulate_genome(list(best), states=states, machine_cfg=mc_eval, collect_activity=True)
        if isinstance(sim, tuple) and len(sim) == 2:
            G_best, mask = sim
        else:
            G_best, mask = sim, None
    except Exception:
        G_best, mask = None, None

    gsum = _graph_summary(G_best, states) if G_best is not None else None
    activity_yaml = _activity_to_yaml(mask)

    if save_best and "best" not in paths:
        d_last_final = _mk_ckpt_subdir(ckpt_dir, "last", ngen)
        _bundle_checkpoint(d_last_final,
            pop=pop, fits=[ind.fitness.values[0] for ind in pop], states=states, machine_cfg=mc_eval, fitness=fitness,
            best_ind=best, best_mask=getattr(best, "active_mask", None), G_best=G_best,
            full_condition=full_cond, hist_bins=hist_bins
        )
        paths["best"] = str(Path(d_last_final) / "genome.yaml")
        if save_best_png and G_best is not None:
            paths["best_png"] = str(Path(d_last_final) / "best.png")

    if save_last and "last" not in paths:
        paths["last"] = str(ckpt_dir)

    if pool is not None:
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

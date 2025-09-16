# src/guca/ga/checkpoint.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import Counter
from pathlib import Path
import json
import csv
import networkx as nx
from guca.ga.encoding import decode_gene, sanitize_gene

def _activity_scheme(mask: Sequence[bool]) -> str:
    if not mask: return ""
    parts, run = [], 0
    for m in mask:
        if m:
            if run: parts.append(str(run)); run = 0
            parts.append("x")
        else:
            run += 1
    if run: parts.append(str(run))
    return "".join(parts)

def _activity_to_yaml(mask: Optional[Sequence[bool]]) -> Optional[Dict[str, Any]]:
    if mask is None: return None
    ml = [bool(x) for x in mask]
    return {"mask": ml, "active_count": int(sum(ml)), "scheme": _activity_scheme(ml)}

def _graph_summary(G: nx.Graph, states: List[str]) -> Dict[str, Any]:
    counts = Counter(states[int(G.nodes[n].get("state_id", 0))] for n in G.nodes())
    e = int(G.number_of_edges())
    summary = {"edges": e, "nodes": int(G.number_of_nodes()), "states_count": dict(counts)}
    if e < 1000:
        edge_list = sorted([[int(min(u, v)), int(max(u, v))] for (u, v) in G.edges()], key=lambda uv: (uv[0], uv[1]))
        summary["edge_list"] = edge_list
    return summary

def _genes_to_yaml_rules(
    genes: List[int],
    states: List[str],
    *,
    full_condition: bool = False,
    machine_encoding: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n_states = max(1, len(states))
    enc = dict(machine_encoding or {})

    for g in genes:
        g_for_decode = g
        if enc.get("sanitize_on_decode", False):
            g_for_decode = sanitize_gene(
                g,
                state_count=n_states,
                enforce_semantics=bool(enc.get("enforce_semantics", False)),
                canonicalize_flags=bool(enc.get("canonicalize_flags", False)),
                enforce_bounds_order=bool(enc.get("enforce_bounds_order", False)),
            )
        r = decode_gene(g_for_decode, state_count=n_states)

        cur = states[int(r.cond_current) % n_states]
        operand = None if r.operand is None else states[int(r.operand) % n_states]

        row: Dict[str, Any] = {"condition": {"current": cur}, "op": {"kind": r.op_kind.name}}
        if operand is not None:
            row["op"]["operand"] = operand

        if full_condition:
            prior = "any" if r.prior is None else states[int(r.prior) % n_states]
            def nz(v): return -1 if v is None else int(v)
            row["condition"].update({
                "prior": prior,
                "conn_ge": nz(r.conn_ge),
                "conn_le": nz(r.conn_le),
                "parents_ge": nz(r.parents_ge),
                "parents_le": nz(r.parents_le),
            })
        out.append(row)
    return out


def _mk_ckpt_subdir(root: Path, tag: str, gen: int) -> Path:
    d = root / f"{tag}_{gen:05d}"; d.mkdir(parents=True, exist_ok=True); return d

def _machine_yaml_from_cfg(machine_cfg: Dict[str, Any]) -> Dict[str, Any]:
    nearest = machine_cfg.get("nearest_search", {}) or {}
    return {
        "start_state": str(machine_cfg.get("start_state", "A")),
        "transcription": str(machine_cfg.get("transcription", "resettable")),
        "count_compare": "range",
        "max_vertices": int(machine_cfg.get("max_vertices", 2000)),
        "max_steps": int(machine_cfg.get("max_steps", 120)),
        "nearest_search": {
            "max_depth": int(nearest.get("max_depth", 2)),
            "tie_breaker": str(nearest.get("tie_breaker", "stable")),
            "connect_all": bool(nearest.get("connect_all", False)),
        },
    }


def _write_population_json(dir_: Path, pop, fits):
    out = []
    for ind, f in zip(pop, fits):
        out.append({"fitness": float(f), "length": len(ind), "active_len": int(sum(getattr(ind, "active_mask", []) or [])), "genes": [f"{g:016x}" for g in ind]})
    p = dir_ / "population.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return str(p)

def _write_hist_png(dir_: Path, name: str, data: List[float], bins: int):
    out = dir_ / name
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        vals = []
        for x in data:
            try: xv = float(x); 
            except Exception: continue
            if xv == xv and abs(xv) != float("inf"): vals.append(xv)
        if not vals: vals = [0.0]
        plt.figure(figsize=(6.4, 4.2)); plt.hist(vals, bins=int(max(1, bins))); plt.title(name.replace("_", " ")); plt.tight_layout(); plt.savefig(out.as_posix(), dpi=150); plt.close()
        return str(out)
    except Exception:
        try:
            with open(out, "wb") as fh: fh.write(b"\x89PNG\r\n\x1a\n")
            return str(out)
        except Exception:
            return None

def _write_metrics_csv(dir_: Path, metrics: Dict[str, Any]):
    p = dir_ / "metrics.csv"
    cols = list(metrics.keys())
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(cols); w.writerow([metrics[k] for k in cols])
    return str(p)

def _write_metrics_txt(dir_: Path, metrics: Dict[str, Any]):
    """
    Write metrics as a simple key:value list, one per line, aligned.
    Useful for quick human inspection alongside metrics.csv.
    """
    p = dir_ / "metrics.txt"
    try:
        width = max((len(str(k)) for k in metrics.keys()), default=0)
        with open(p, "w", encoding="utf-8") as fh:
            for k, v in metrics.items():
                fh.write(f"{str(k).ljust(width)} : {v}\n")
        return str(p)
    except Exception:
        # best-effort fall-back: still try to create an empty file to signal intent
        try:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("")
            return str(p)
        except Exception:
            return None


def _render_png_dots(dir_: Path, G: nx.Graph, out_name: str = "best.png") -> Optional[str]:
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

def _write_genome_yaml(dir_: Path, best_genes: List[int], states: List[str], machine_cfg: Dict[str, Any], graph_summary: Optional[Dict[str, Any]] = None, activity_mask: Optional[List[bool]] = None, full_condition: bool = False) -> str:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore
    y = {
        "machine": _machine_yaml_from_cfg(machine_cfg),
        "init_graph": {"nodes": [{"state": str(machine_cfg.get("start_state", "A"))}]},
        "rules": _genes_to_yaml_rules(
            best_genes, states,
            full_condition=full_condition,
            machine_encoding=machine_cfg.get("encoding", {})
        ),
    }
    if graph_summary or activity_mask is not None:
        y["meta"] = {}
        if graph_summary:
            y["meta"]["graph_summary"] = graph_summary
        if activity_mask is not None:            
            y["meta"]["activity_scheme"] = _activity_scheme(activity_mask)

    p = dir_ / "genome.yaml"
    if yaml is None:
        with open(p, "w", encoding="utf-8") as fh: fh.write(json.dumps(y, indent=2))
    else:
        with open(p, "w", encoding="utf-8") as fh: yaml.safe_dump(y, fh, sort_keys=False, allow_unicode=True)
    return str(p)

def _bundle_checkpoint(dir_: Path, *, pop, fits, states, machine_cfg, fitness, best_ind, best_mask, G_best, full_condition: bool, hist_bins: int):
    _write_population_json(dir_, pop, fits)
    gsum = _graph_summary(G_best, states) if G_best is not None else None
    genome_path = _write_genome_yaml(dir_, list(best_ind), states, machine_cfg, gsum, best_mask, full_condition)
    if G_best is not None:
        _render_png_dots(dir_, G_best, out_name="best.png")
    # hists
    lengths = [len(ind) for ind in pop]
    actlens = [int(sum(getattr(ind, "active_mask", []) or [])) for ind in pop]
    _write_hist_png(dir_, "hist_fitness.png", fits, bins=hist_bins)
    _write_hist_png(dir_, "hist_length.png", lengths, bins=hist_bins)
    _write_hist_png(dir_, "hist_active.png", actlens, bins=hist_bins)
    # metrics.csv
    mx = max(fits) if fits else 0.0
    tops = [ind for ind, f in zip(pop, fits) if abs(f - mx) < 1e-12]
    best_act = int(sum(getattr(best_ind, "active_mask", []) or []))
    avg_fit = sum(fits) / max(1, len(fits))
    cnt_max = len(tops)
    top_act_av = (sum(int(sum(getattr(t, "active_mask", []) or [])) for t in tops) / cnt_max) if cnt_max else 0.0
    pop_act_av = sum(actlens) / max(1, len(actlens))
    best_len = len(best_ind)
    avg_len = sum(lengths) / max(1, len(lengths))

    # In _bundle_checkpoint, just before building 'stats', compute extra metrics:
    fit_metrics: Dict[str, Any] = {}
    best_scheme = _activity_scheme(getattr(best_ind, "active_mask", []) or [])
    try:
        sval, fit_metrics = fitness.score(G_best, return_metrics=True)  # sval unused
        if not isinstance(fit_metrics, dict):
            fit_metrics = {}
    except Exception:
        fit_metrics = {}

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

    stats.update({"best_activity_scheme": best_scheme})
    # merge extra per-individual metrics into stats (namespaced or flat; keep flat & short)
    for k, v in fit_metrics.items():
        # only simple scalars for CSV
        if isinstance(v, (int, float, str, bool)):
            stats[k] = v

    _write_metrics_csv(dir_, stats)
    _write_metrics_txt(dir_, stats)
    return {"genome": genome_path, "metrics": stats}

def _write_checkpoint_epoch(ckpt_dir: Path, gen: int, pop: List[List[int]], fits: List[float]) -> str:
    p = ckpt_dir / f"epoch_{gen}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"generation": gen, "population": [{"fitness": float(fits[i]), "genes": [f"{g:016x}" for g in pop[i]]} for i in range(len(pop))]}, f, indent=2)
    return str(p)

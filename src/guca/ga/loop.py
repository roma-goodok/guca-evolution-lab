# src/guca/ga/loop.py
from __future__ import annotations
import json, random, csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from multiprocessing import get_context

import networkx as nx
from deap import base, creator, tools

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

from .simulate import simulate_genome
from .selection import _make_selector
from .checkpoint import (
    _activity_scheme, _activity_to_yaml, _graph_summary, _mk_ckpt_subdir,
    _bundle_checkpoint, _write_checkpoint_epoch, _write_population_json
)

# --- worker globals for multiprocessing ---
_WFIT = None
_WSTATES = None
_WMCFG = None

def _init_worker(fitness, states, machine_cfg):
    global _WFIT, _WSTATES, _WMCFG
    _WFIT = fitness; _WSTATES = states; _WMCFG = machine_cfg

def _eval_only(genes: List[int]) -> float:
    G = simulate_genome(genes, states=_WSTATES, machine_cfg=_WMCFG)
    meta = {"genome_len": len(genes), "max_steps": _WMCFG.get("max_steps")}
    return float(_WFIT.score(G, meta=meta))

def _eval_with_mask(genes: List[int]) -> Tuple[float, List[bool], Dict[str, Any]]:
    """Evaluation helper for pools: returns (fitness, activity_mask, metrics_dict)."""
    G, mask = simulate_genome(genes, states=_WSTATES, machine_cfg=_WMCFG, collect_activity=True)
    meta = {"genome_len": len(genes), "max_steps": _WMCFG.get("max_steps")}
    # Prefer rich return (score, metrics); fallback to plain score
    try:
        s, metrics = _WFIT.score(G, meta=meta, return_metrics=True)  # type: ignore[call-arg]
    except TypeError:
        s, metrics = float(_WFIT.score(G, meta=meta)), {}
    return float(s), list(mask or []), dict(metrics or {})

def evolve(
    *,
    fitness,
    machine_cfg: Dict[str, Any],
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

    mc_eval = dict(machine_cfg)
    mc_eval["encoding"] = dict(ga_cfg.get("encoding", {}))

    try: creator.FitnessMax  # type: ignore[attr-defined]
    except Exception: creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try: creator.Individual  # type: ignore[attr-defined]
    except Exception: creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    from guca.ga.encoding import random_gene
    toolbox.register("rand_gene", random_gene, state_count=state_count, rng=r)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.rand_gene, n=init_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # crossover (length-1 safe tail swap)
    def cx(ind1, ind2):
        L1, L2 = len(ind1), len(ind2)
        if L1 == 0 or L2 == 0: return ind1, ind2
        c1 = (r.randrange(1, L1) if L1 > 1 else 0)
        c2 = (r.randrange(1, L2) if L2 > 1 else 0)
        child1 = ind1[:c1] + ind2[c2:]
        child2 = ind2[:c2] + ind1[c1:]
        if len(child1) > max_len: child1 = child1[:max_len]
        if len(child2) > max_len: child2 = child2[:max_len]
        m1 = getattr(ind1, "active_mask", [False]*L1)
        m2 = getattr(ind2, "active_mask", [False]*L2)
        setattr(ind1, "active_mask", (m1[:c1] + m2[c2:])[:len(child1)])
        setattr(ind2, "active_mask", (m2[:c2] + m1[c1:])[:len(child2)])
        ind1[:] = child1; ind2[:] = child2
        return ind1, ind2
    toolbox.register("mate", cx)

    from guca.ga.operators import make_mutate_fn
    mutate = make_mutate_fn(
        state_count=state_count, min_len=min_len, max_len=max_len,
        structural_cfg=ga_cfg.get("structural", {}),
        field_cfg=ga_cfg.get("field", {}),
        active_cfg=ga_cfg.get("active", {}),
        passive_cfg=ga_cfg.get("passive", {}),
        structuralx_cfg=ga_cfg.get("structuralx", {}),
        rng=r,
    )
    toolbox.register("mutate", mutate)

    sel_cfg = ga_cfg.get("selection", {}) or {}
    selector_fn = _make_selector(
        str(sel_cfg.get("method", "rank")),
        tournament_k=int(ga_cfg.get("tournament_k", 3)),
        random_ratio=float(sel_cfg.get("random_ratio", 0.0)),
        rng=r
    )
    toolbox.register("select", selector_fn)

    def evaluate(individual: List[int]) -> Tuple[float]:
        G, active_mask = simulate_genome(individual, states=states, machine_cfg=mc_eval, collect_activity=True)
        setattr(individual, "active_mask", list(active_mask))
        meta = {"genome_len": len(individual), "max_steps": mc_eval.get("max_steps")}
        try:
            s, metrics = fitness.score(G, meta=meta, return_metrics=True)  # type: ignore[call-arg]
        except TypeError:
            s, metrics = float(fitness.score(G, meta=meta)), {}
        setattr(individual, "metrics", metrics)
        return (float(s),)
    toolbox.register("evaluate", evaluate)

    pop_size = int(ga_cfg.get("pop_size", 40))
    cx_pb = float(ga_cfg.get("cx_pb", 0.7))
    mut_pb = float(ga_cfg.get("mut_pb", 0.3))
    ngen = int(ga_cfg.get("generations", 20))
    elitism = int(ga_cfg.get("elitism", 2))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(max(1, elitism))

    # parallel map
    global _WFIT, _WSTATES, _WMCFG
    _WFIT, _WSTATES, _WMCFG = fitness, states, mc_eval
    pool = None
    if n_workers and n_workers > 0:
        try:
            ctx = get_context("spawn")
            pool = ctx.Pool(processes=int(n_workers), initializer=_init_worker, initargs=(fitness, states, mc_eval))
            map_fn = pool.map
        except Exception as e:
            print(f"[WARN] parallel init failed ({e}); falling back to serial.")
            map_fn = map
    else:
        map_fn = map

    # evaluate init (fitness, mask, metrics)
    res = list(map_fn(_eval_with_mask, pop))
    for ind, (fit, mask, metrics) in zip(pop, res):
        ind.fitness.values = (fit,)
        setattr(ind, "active_mask", mask)
        setattr(ind, "metrics", metrics)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda xs: sum(xs) / max(1, len(xs)))
    stats.register("max", max)
    stats.register("min", min)

    ckpt = checkpoint_cfg or {}
    ckpt_dir = (run_dir / ckpt.get("out_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fmt = str(ckpt.get("fmt", "json")).lower()
    save_best = bool(ckpt.get("save_best", True))
    save_last = bool(ckpt.get("save_last", True))
    save_every = int(ckpt.get("save_every", 0))
    save_population = str(ckpt.get("save_population", "best"))
    full_cond = bool(ckpt.get("export_full_condition_shape", False))
    save_best_png = bool(ckpt.get("save_best_png", False))
    hist_bins = int(ckpt.get("hist_bins", 100))

    hof.update(pop)
    best0 = hof[0]
    fits0 = [ind.fitness.values[0] for ind in pop]

    # Helper: append a row into a single progress.csv for the whole run
    progress_csv = ckpt_dir / "progress.csv"
    def _append_progress(gen_idx: int, pop_list: List[Any]) -> None:
        progress_csv.parent.mkdir(parents=True, exist_ok=True)
        fits_arr = [float(ind.fitness.values[0]) for ind in pop_list]
        lens_arr = [len(ind) for ind in pop_list]
        acts_arr = [int(sum(getattr(ind, "active_mask", []) or [])) for ind in pop_list]
        best = hof[0]
        best_scheme = _activity_scheme(getattr(best, "active_mask", []) or [])

        tops = [ind for ind in pop_list if abs(ind.fitness.values[0] - max(fits_arr)) < 1e-12]
        if tops:
            len_top_max = max(len(t) for t in tops); len_top_min = min(len(t) for t in tops); len_top_avg = sum(len(t) for t in tops)/len(tops)
            act_top_vals = [int(sum(getattr(t, "active_mask", []) or [])) for t in tops]
            act_top_max = max(act_top_vals); act_top_min = min(act_top_vals); act_top_avg = sum(act_top_vals)/len(act_top_vals)
        else:
            len_top_max = len_top_min = len_top_avg = act_top_max = act_top_min = act_top_avg = 0

        metrics_json = "{}"
        best_metrics = {}
        if hasattr(best, "metrics") and isinstance(best.metrics, dict):
            # keep JSON for compatibility
            try:
                metrics_json = json.dumps(best.metrics, ensure_ascii=False)
            except Exception:
                metrics_json = "{}"
            # numeric-only view for columns
            for k, v in best.metrics.items():
                if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
                    best_metrics[k] = float(v)

        # choose stable set of metric keys to emit as columns
        KNOWN_KEYS = [
            "tri_count", "quad_count", "hex_count",
            "interior_deg3", "interior_deg4", "interior_deg6",
            "nodes", "edges", "shell_len", "faces_total", "faces_interior",
            "tri_no_shell_edges",  # present for TriangleMesh
            "forbidden_penalty", "longface_penalty", "allowed_max_face_len",  # QuadMesh extras
        ]
        metric_cols = {f"m_{k}": float(best_metrics.get(k, 0.0)) for k in KNOWN_KEYS}

        row = {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "gen": gen_idx,
            "best_fitness": max(fits_arr) if fits_arr else 0.0,
            "avg_fitness": (sum(fits_arr)/max(1,len(fits_arr))) if fits_arr else 0.0,
            "cnt_max": len(tops),
            "best_length": len(best) if len(hof) else 0,
            "len_avg": (sum(lens_arr)/max(1,len(lens_arr))) if lens_arr else 0.0,
            "act_avg": (sum(acts_arr)/max(1,len(acts_arr))) if acts_arr else 0.0,
            "len_top_min": len_top_min, "len_top_avg": len_top_avg, "len_top_max": len_top_max,
            "act_top_min": act_top_min, "act_top_avg": act_top_avg, "act_top_max": act_top_max,
            "best_activity_scheme": best_scheme,
            "best_metrics_json": metrics_json,
        }
        row.update(metric_cols)

        write_header = not progress_csv.exists()
        with open(progress_csv, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)


    # initial last_00000
    last_dir_path = None
    if save_last:
        last_dir = _mk_ckpt_subdir(ckpt_dir, "last", 0)
        # simulate best for artefacts
        try:
            sim0 = simulate_genome(list(best0), states=states, machine_cfg=mc_eval, collect_activity=True)
            G0, mask0 = (sim0 if isinstance(sim0, tuple) else (sim0, getattr(best0, "active_mask", None)))
        except Exception:
            G0, mask0 = None, getattr(best0, "active_mask", None)
        _bundle_checkpoint(last_dir, pop=pop, fits=fits0, states=states, machine_cfg=mc_eval, fitness=fitness, best_ind=best0, best_mask=mask0, G_best=G0, full_condition=full_cond, hist_bins=hist_bins)
        last_dir_path = str(last_dir)

    paths: Dict[str, str] = {}
    if save_best and last_dir_path is not None:
        paths["best"] = str(Path(last_dir_path) / "genome.yaml")

    if save_population in ("best", "all"):
        _write_population_json(ckpt_dir, [list(ind) for ind in pop], [ind.fitness.values[0] for ind in pop])

    if save_every and save_every > 0:
        _write_checkpoint_epoch(ckpt_dir, 0, [list(ind) for ind in pop], [ind.fitness.values[0] for ind in pop])

    record0 = stats.compile(pop)
    best_len0 = len(hof[0]) if len(hof) else 0
    best_so_far = record0["max"]

    # append gen-0 progress row
    _append_progress(0, pop)

    pbar = None
    if progress and tqdm is not None:
        fits_arr = [float(ind.fitness.values[0]) for ind in pop]
        lens_arr = [len(ind) for ind in pop]
        acts_arr = [int(sum(getattr(ind, "active_mask", []) or [])) for ind in pop]
        tops0 = [ind for ind in pop if abs(ind.fitness.values[0] - max(fits_arr)) < 1e-12]
        if tops0:
            len_top_max = max(len(t) for t in tops0); len_top_min = min(len(t) for t in tops0); len_top_avg = sum(len(t) for t in tops0)/len(tops0)
            act_top_vals = [int(sum(getattr(t, "active_mask", []) or [])) for t in tops0]
            act_top_max = max(act_top_vals); act_top_min = min(act_top_vals); act_top_avg = sum(act_top_vals)/len(act_top_vals)
        else:
            len_top_max = len_top_min = len_top_avg = act_top_max = act_top_min = act_top_avg = 0
        pbar = tqdm(total=ngen, desc="epochs", leave=True, dynamic_ncols=True)
        pbar.set_postfix(
            _max=f"{record0['max']:.4f}", avg=f"{record0['avg']:.4f}", len=best_len0,
            len_avg=f"{(sum(lens_arr)/max(1,len(lens_arr))):.2f}", cnt_max=len(tops0),
            act_avg=f"{(sum(acts_arr)/max(1,len(acts_arr))):.2f}",
            len_top=f"{len_top_min:.0f}/{len_top_avg:.1f}/{len_top_max:.0f}",
            act_top=f"{act_top_min:.0f}/{act_top_avg:.1f}/{act_top_max:.0f}",
        )
    elif progress:
        print(f"gen 0/{ngen}  max={record0['max']:.4f}  avg={record0['avg']:.4f}  len={best_len0}")

    for gen in range(1, ngen + 1):
        elites = tools.selBest(pop, k=elitism)
        offspring = toolbox.select(pop, k=pop_size - elitism)
        offspring = list(map(toolbox.clone, offspring))

        # random immigrants
        rand_ratio = float((ga_cfg.get("selection", {}) or {}).get("random_ratio", ga_cfg.get("random_selection_ratio", 0.0)))
        rand_ratio = max(0.0, min(1.0, rand_ratio))
        n_rand = int(rand_ratio * len(offspring))
        if n_rand > 0:
            try: immigrants = list(toolbox.population(n=n_rand))
            except Exception: immigrants = [toolbox.individual() for _ in range(n_rand)]
            offspring[:n_rand] = immigrants

        # mate
        for i in range(1, len(offspring), 2):
            if r.random() < cx_pb:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values; del offspring[i].fitness.values
        # mutate
        for i in range(len(offspring)):
            if r.random() < mut_pb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        pop = elites + offspring

        invalid = [ind for ind in pop if not ind.fitness.valid]
        if invalid:
            res_inv = list(map_fn(_eval_with_mask, invalid))
            for ind, (fit, mask, metrics) in zip(invalid, res_inv):
                ind.fitness.values = (fit,)
                setattr(ind, "active_mask", mask)
                setattr(ind, "metrics", metrics)

        hof.update(pop)
        record = stats.compile(pop)

        # periodic checkpoint
        if save_every and gen % save_every == 0:
            try:
                simt = simulate_genome(list(hof[0]), states=states, machine_cfg=mc_eval, collect_activity=True)
                G_tmp, mask_tmp = (simt if isinstance(simt, tuple) else (simt, getattr(hof[0], "active_mask", None)))
                d_epoch = _mk_ckpt_subdir(ckpt_dir, "epoch", gen)
                _bundle_checkpoint(d_epoch, pop=pop, fits=[float(ind.fitness.values[0]) for ind in pop], states=states, machine_cfg=mc_eval, fitness=fitness, best_ind=hof[0], best_mask=mask_tmp, G_best=G_tmp, full_condition=full_cond, hist_bins=hist_bins)
            except Exception:
                pass

        # periodic checkpoint
        if gen % 10 == 0:
            try:
                _append_progress(gen, pop)                
            except Exception:
                pass

        # out-of-sequence last when a new best appears
        if record["max"] > best_so_far and save_last:
            best_so_far = record["max"]
            try:
                simn = simulate_genome(list(hof[0]), states=states, machine_cfg=mc_eval, collect_activity=True)
                G_new, mask_new = (simn if isinstance(simn, tuple) else (simn, getattr(hof[0], "active_mask", None)))
                d_last = _mk_ckpt_subdir(ckpt_dir, "last", gen)
                _bundle_checkpoint(d_last, pop=pop, fits=[float(ind.fitness.values[0]) for ind in pop], states=states, machine_cfg=mc_eval, fitness=fitness, best_ind=hof[0], best_mask=mask_new, G_best=G_new, full_condition=full_cond, hist_bins=hist_bins)
                paths["best"] = str(Path(d_last) / "genome.yaml")
                if save_best_png and G_new is not None:
                    paths["best_png"] = str(Path(d_last) / "best.png")
                paths["last"] = str(d_last)

                # append row to progress.csv for this generation
                _append_progress(gen, pop)
            except Exception:
                pass

        # progress display
        if pbar is not None:
            fits_arr = [float(ind.fitness.values[0]) for ind in pop]
            lens_arr = [len(ind) for ind in pop]
            acts_arr = [int(sum(getattr(ind, "active_mask", []) or [])) for ind in pop]
            tops = [ind for ind in pop if abs(ind.fitness.values[0] - max(fits_arr)) < 1e-12]
            if tops:
                len_top_max = max(len(t) for t in tops); len_top_min = min(len(t) for t in tops); len_top_avg = sum(len(t) for t in tops)/len(tops)
                act_top_vals = [int(sum(getattr(t, "active_mask", []) or [])) for t in tops]
                act_top_max = max(act_top_vals); act_top_min = min(act_top_vals); act_top_avg = sum(act_top_vals)/len(act_top_vals)
            else:
                len_top_max = len_top_min = len_top_avg = act_top_max = act_top_min = act_top_avg = 0
            pbar.set_postfix(
                _max=f"{record['max']:.4f}", avg=f"{record['avg']:.4f}",
                len=len(hof[0]) if len(hof) else 0,
                len_avg=f"{(sum(lens_arr)/max(1,len(lens_arr))):.2f}",
                cnt_max=len(tops),
                act_avg=f"{(sum(acts_arr)/max(1,len(acts_arr))):.2f}",
                len_top=f"{len_top_min:.0f}/{len_top_avg:.1f}/{len_top_max:.0f}",
                act_top=f"{act_top_min:.0f}/{act_top_avg:.1f}/{act_top_max:.0f}",
            )
            pbar.update(1)
        elif progress:
            print(f"gen {gen}/{ngen}  max={record['max']:.4f}  avg={record['avg']:.4f}  len={len(hof[0]) if len(hof) else 0}")

        

    if pbar is not None:
        pbar.close()

    # final
    best = hof[0]
    try:
        sim = simulate_genome(list(best), states=states, machine_cfg=mc_eval, collect_activity=True)
        G_best, mask = (sim if isinstance(sim, tuple) else (sim, None))
    except Exception:
        G_best, mask = None, None
    gsum = _graph_summary(G_best, states) if G_best is not None else None
    if save_best and "best" not in paths:
        d_last_final = _mk_ckpt_subdir(ckpt_dir, "last", ngen)
        _bundle_checkpoint(d_last_final, pop=pop, fits=[ind.fitness.values[0] for ind in pop], states=states, machine_cfg=mc_eval, fitness=fitness, best_ind=best, best_mask=getattr(best, "active_mask", None), G_best=G_best, full_condition=full_cond, hist_bins=hist_bins)
        paths["best"] = str(Path(d_last_final) / "genome.yaml")
        if G_best is not None:
            paths["best_png"] = str(Path(d_last_final) / "best.png")
    if save_last and "last" not in paths:
        paths["last"] = str(ckpt_dir)

    if pool is not None:
        pool.close(); pool.join()

    return {
        "status": "ok",
        "best_fitness": float(best.fitness.values[0]),
        "best_length": len(best),
        "pop_size": pop_size,
        "generations": ngen,
        "graph_summary": gsum,
        "checkpoints": paths,
    }

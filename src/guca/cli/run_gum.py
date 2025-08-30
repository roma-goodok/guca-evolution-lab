from __future__ import annotations
import os
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from typing import Any, Dict
import yaml

from guca.core.graph import GUMGraph, stats_summary
from guca.core.machine import GraphUnfoldingMachine
from guca.core.rules import change_table_from_yaml, TranscriptionWay, CountCompare

def _save_placeholder_png(out_path: Path, *, title: str = "", body: str = "") -> float:
    """
    Write a simple PNG with a title and a body text. Returns plotting time (seconds).
    This avoids depending on graph internals; good enough for Week 1.5 screenshots/tests.
    """
    import matplotlib.pyplot as plt  # local import to keep CLI import cheap
    t0 = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.9
    if title:
        ax.text(0.02, y, title, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        y -= 0.06
    if body:
        ax.text(0.02, y, body, transform=ax.transAxes, fontsize=10, va="top", family="monospace")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return time.perf_counter() - t0


def _save_graph_png(graph, out_path: Path, *, layout_seed: int | None = 42) -> float:
    """
    Render the resulting graph to a PNG using networkx + matplotlib.
    We try to be resilient to different graph shapes:
      - If graph has .nodes() / .edges() yielding ids/tuples, we use them directly.
      - If edges are objects, we try common attributes (u/v, source/target, a/b).
    Returns plotting time (seconds).
    """
    import time
    t0 = time.perf_counter()

    # Late imports keep CLI import-time light
    import matplotlib.pyplot as plt
    try:
        import networkx as nx
    except Exception:
        # Soft fallback: blank placeholder if networkx missing
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.05, 0.95, "networkx not installed", va="top", transform=ax.transAxes)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return time.perf_counter() - t0

    # --- Build a networkx.Graph from "graph" ---
    G = nx.Graph()

    # Collect nodes
    nodes_added = False
    for attr in ("nodes", "Vertices", "vertices"):
        if hasattr(graph, attr):
            obj = getattr(graph, attr)
            try:
                it = obj() if callable(obj) else obj
                for n in it:
                    # n could be an object; use itself as id or its id/int index if present
                    nid = getattr(n, "id", None)
                    G.add_node(nid if nid is not None else n)
                nodes_added = True
            except Exception:
                pass
            break  # stop at first workable attr

    # Collect edges
    def _edge_endpoints(e):
        # Try common fields in order
        for (a, b) in (
            ("u", "v"),
            ("source", "target"),
            ("src", "dst"),
            ("a", "b"),
            ("from_", "to"),
            ("From", "To"),
        ):
            if hasattr(e, a) and hasattr(e, b):
                return getattr(e, a), getattr(e, b)
        # Try methods
        for m in ("endpoints", "tuple", "as_tuple"):
            if hasattr(e, m) and callable(getattr(e, m)):
                try:
                    return tuple(getattr(e, m)())
                except Exception:
                    pass
        return None

    edges_added = False
    for attr in ("edges", "Edges", "adjacent_edges", "AdjacentEdges"):
        if hasattr(graph, attr):
            obj = getattr(graph, attr)
            try:
                it = obj() if callable(obj) else obj
                for e in it:
                    if isinstance(e, tuple) and len(e) >= 2:
                        u, v = e[0], e[1]
                    else:
                        pair = _edge_endpoints(e)
                        if pair is None:
                            # Try QuickGraph-like: e.Source, e.Target
                            if hasattr(e, "Source") and hasattr(e, "Target"):
                                pair = (getattr(e, "Source"), getattr(e, "Target"))
                            elif hasattr(e, "source") and hasattr(e, "target"):
                                pair = (getattr(e, "source"), getattr(e, "target"))
                        if pair is None:
                            continue
                        u, v = pair
                    # If nodes are objects, normalize to ids consistently with nodes above
                    uid = getattr(u, "id", None)
                    vid = getattr(v, "id", None)
                    G.add_edge(uid if uid is not None else u, vid if vid is not None else v)
                edges_added = True
            except Exception:
                pass
            break  # stop at first workable attr

    # If we didn't get nodes but did add edges, add nodes from edges
    if not nodes_added:
        G.add_nodes_from(list(G.nodes()))

    # --- Draw ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111)
    ax.axis("off")

    if len(G) == 0:
        ax.text(0.05, 0.95, "empty graph", va="top", transform=ax.transAxes)
    else:
        try:
            pos = nx.spring_layout(G, seed=layout_seed)
        except Exception:
            pos = None

        if pos:
            nx.draw_networkx(
                G, pos,
                with_labels=False,
                node_size=28,
                width=0.6,
                linewidths=0.2,
                alpha=0.95,
                ax=ax,
            )
        else:
            # Fallback: rely on networkx defaults
            nx.draw(G, ax=ax)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return time.perf_counter() - t0



def _derive_genome_name(genome_path: Path) -> str:
    p = Path(genome_path)
    return (p.stem or "genome").replace(os.sep, "_") if hasattr(p, "stem") else "genome"


def _load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(argv: Optional[list[str]] = None) -> int:

    ap = argparse.ArgumentParser(description="Run a GUM genome and print stats (JSON).")
    ap.add_argument("--genome", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--save-png", action="store_true", help="Save final PNG of the result (placeholder).")
    ap.add_argument("--run-dir", default="runs", help="Base output directory (default: runs).")
    ap.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")

    args = ap.parse_args()

    cfg = _load_yaml(args.genome)
    machine_cfg = cfg.get("machine", {})
    rules_yaml = cfg.get("rules", [])

    graph = GUMGraph()
    m = GraphUnfoldingMachine(
        graph,
        start_state=str(machine_cfg.get("start_state", "A")),
        transcription=TranscriptionWay(machine_cfg.get("transcription", "resettable")),
        count_compare=CountCompare(machine_cfg.get("count_compare", "range")),
        max_vertices=int(machine_cfg.get("max_vertices", 0)),
        max_steps=(args.steps if args.steps is not None else int(machine_cfg.get("max_steps", 100))),
    )
    m.change_table = change_table_from_yaml(rules_yaml)

    # --- logging setup + output layout ---
    genome_name = _derive_genome_name(args.genome)
    run_base = Path(args.run_dir)
    out_genome_dir = run_base / genome_name
    vis_dir = out_genome_dir / "vis"
    logs_dir = out_genome_dir / "logs"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # configure logging (console + file)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile = logs_dir / f"run-{ts}.log"
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logfile, encoding="utf-8"),
        ],
    )
    logging.info("GUM run: genome=%s steps_param=%s genome_path=%s run_dir=%s",
                genome_name, getattr(args, "steps", None), args.genome, args.run_dir)

    # timings
    t0_total = time.perf_counter()
    t0_engine = time.perf_counter()

    m.run()
    t_engine = time.perf_counter() - t0_engine


    summary = stats_summary(graph)
    
    # Save PNG if requested (placeholder includes summary text)
    t_plot = 0.0
    saved_png = None
    if getattr(args, "save_png", False):
        saved_png = vis_dir / f"step{getattr(m, 'passed_steps', getattr(args, 'steps', 0))}.png"
        body = json.dumps(summary, indent=2, sort_keys=True)        
        t_plot = _save_graph_png(graph, saved_png)

    # Log stats and timings
    logging.info("GUM result graph stats: %s", json.dumps(summary, sort_keys=True))
    t_total = time.perf_counter() - t0_total
    logging.info("Timing: engine=%.3fs, plot=%.3fs, total=%.3fs", t_engine, t_plot, t_total)
    if saved_png is not None:
        logging.info("Saved PNG: %s", saved_png.as_posix())
        logging.info("Run log: %s", logfile.as_posix())


    if args.do_assert and "expected" in cfg:
        exp = cfg["expected"] or {}
        if "nodes" in exp:
            assert summary["nodes"] == int(exp["nodes"])
        if "edges" in exp:
            assert summary["edges"] == int(exp["edges"])
        if "states_count" in exp and isinstance(exp["states_count"], dict):
            for k, v in exp["states_count"].items():
                assert summary["states_count"].get(k, 0) == int(v)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


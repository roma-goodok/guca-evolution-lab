# src/guca/cli/draw_edge_list.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import argparse, json, yaml
import networkx as nx
from datetime import datetime

from guca.vis.png import save_png

def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _pick_edge_list(cfg: Any) -> list[list[int]]:
    if not isinstance(cfg, dict):
        return []
    # 1) top-level edge_list
    if isinstance(cfg.get("edge_list"), list):
        return cfg["edge_list"]
    meta = cfg.get("meta") or {}
    # 2) meta.edge_list
    if isinstance(meta.get("edge_list"), list):
        return meta["edge_list"]
    # 3) meta.graph_summary.edge_list
    gs = (meta.get("graph_summary") or {})
    if isinstance(gs.get("edge_list"), list):
        return gs["edge_list"]
    return []

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render a graph from a YAML edge_list.")
    ap.add_argument("--yaml", required=True, type=Path, help="YAML with edge_list (top-level or meta.*)")
    ap.add_argument("--run-dir", default="examples/runs", help="Output base directory")
    ap.add_argument("--vis-node-render", choices=["full", "ids", "dots", "none"], default="ids")
    ap.add_argument("--vis-dot-size", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = _load_yaml(args.yaml)
    el = _pick_edge_list(cfg)
    if not el:
        raise SystemExit(f"No edge_list found in {args.yaml}")

    G = nx.Graph()
    G.add_edges_from([(int(u), int(v)) for (u, v) in el])

    stem = args.yaml.stem
    out_base = Path(args.run_dir) / stem
    vis_dir = out_base / "vis"
    logs_dir = out_base / "logs"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    out_png = vis_dir / "edge_list.png"
    t = save_png(
        G,
        out_png,
        node_render=args.vis_node_render,
        dots_node_size=args.vis_dot_size,
        text_color_override=("#ffffff" if args.vis_node_render == "ids" else None),
    )

    log = logs_dir / f"draw-edges-{datetime.now():%Y%m%d-%H%M%S}.log"
    log.write_text(json.dumps({"png": out_png.as_posix(), "edges": len(el), "render_s": t}, indent=2), encoding="utf-8")

    print(json.dumps({"png": out_png.as_posix(), "edges": len(el)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

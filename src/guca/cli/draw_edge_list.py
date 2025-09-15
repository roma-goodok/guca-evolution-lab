# src/guca/cli/draw_edge_list.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import argparse, json, yaml
import networkx as nx
from datetime import datetime

from guca.fitness.meshes import TriangleMesh
from guca.fitness.planar_basic import PlanarBasic
import copy


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

    # Triangle-legacy metrics + faces/shell on the reconstructed graph
    tm = TriangleMesh()
    tl_score, tl_metrics = tm.score(G, return_metrics=True)

    pb = PlanarBasic()
    emb = pb.compute_embedding_info(G)
    faces_all = [list(f) for f in emb.faces]
    shell_seq = list(emb.shell)


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

    # Enrich and write YAML (do not overwrite the source file)
    enriched = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
    meta = dict(enriched.get("meta") or {})

    meta["graph_summary"] = {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "edge_list": el,
    }
    meta["triangle_legacy_score"] = float(tl_score)
    meta["triangle_legacy_metrics"] = {k: (int(v) if isinstance(v, bool) or isinstance(v, int) else v)
                                    for k, v in tl_metrics.items()}
    meta["faces_all"] = faces_all
    meta["shell"] = shell_seq

    enriched["meta"] = meta
    out_yaml = out_base / "edge_list_enriched.yaml"
    out_yaml.write_text(yaml.safe_dump(enriched, sort_keys=False, allow_unicode=True), encoding="utf-8")


    log = logs_dir / f"draw-edges-{datetime.now():%Y%m%d-%H%M%S}.log"
    log.write_text(json.dumps({"png": out_png.as_posix(), "edges": len(el), "render_s": t}, indent=2), encoding="utf-8")

    result = {
        "png": out_png.as_posix(),
        "yaml": out_yaml.as_posix(),
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "shell_len": len(shell_seq),
        "faces_total": len(faces_all),
        "faces_interior": len([f for f in faces_all if f != shell_seq]),
        "triangle_legacy_score": float(tl_score),
        "triangle_legacy_metrics": {
            "tri_count": int(tl_metrics.get("tri_count", 0)),
            "interior_deg6": int(tl_metrics.get("interior_deg6", 0)),
            "tri_no_shell_edges": int(tl_metrics.get("tri_no_shell_edges", 0)),
            "nodes": int(tl_metrics.get("nodes", G.number_of_nodes())),
            "edges": int(tl_metrics.get("edges", G.number_of_edges())),
            "shell_len": int(tl_metrics.get("shell_len", len(shell_seq))),
        },
    }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

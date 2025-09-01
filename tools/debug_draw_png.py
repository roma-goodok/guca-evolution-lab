# tools/debug_draw_png.py
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

from guca.utils.lattices import make_tri_patch, make_quad_patch, make_hex_patch
from guca.fitness.planar_basic import PlanarBasic


def save_graph_png(G: nx.Graph, path: Path, title: str = "") -> None:
    """Save a PNG of the graph, highlighting the outer shell."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # positions: prefer embedded 'pos', else planar layout
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        pos = nx.planar_layout(G)

    # compute shell for overlay
    pb = PlanarBasic()
    emb = pb.compute_embedding_info(G)
    shell_edges = set()
    for i in range(len(emb.shell)):
        a, b = emb.shell[i], emb.shell[(i + 1) % len(emb.shell)]
        if G.has_edge(a, b) or G.has_edge(b, a):
            shell_edges.add((a, b) if a <= b else (b, a))

    # draw base
    plt.figure(figsize=(3, 3))
    nx.draw_networkx_nodes(G, pos=pos, node_size=10)
    # non-shell edges
    non_shell = []
    for u, v in G.edges():
        e = (u, v) if u <= v else (v, u)
        if e not in shell_edges:
            non_shell.append((u, v))
    nx.draw_networkx_edges(G, pos=pos, edgelist=non_shell, width=1)

    # shell overlay
    nx.draw_networkx_edges(G, pos=pos, edgelist=list(shell_edges), width=2)

    plt.title(title)
    plt.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Render small lattice patches to PNG.")
    ap.add_argument("--out-dir", default="artifacts/mesh_previews", help="Output directory for PNGs.")
    ap.add_argument("--families", nargs="+", default=["tri", "quad", "hex"],
                    choices=["tri", "quad", "hex"], help="Which families to render.")
    ap.add_argument("--tri-kind", default="block", choices=["block", "strip", "compact"])
    ap.add_argument("--hex-kind", default="block", choices=["block", "strip", "compact"])
    ap.add_argument("--faces", nargs="+", type=int, default=[1, 2, 4, 6, 10],
                    help="Face counts to render for tri/hex families.")
    ap.add_argument("--quads", nargs="+", default=["1x1", "1x2", "2x2", "3x2"],
                    help="rowsxcols specs for quad family.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if "tri" in args.families:
        for f in args.faces:
            G = make_tri_patch(args.tri_kind, f)
            save_graph_png(G, out_dir / f"tri_{args.tri_kind}_{f}.png", f"tri {args.tri_kind} {f}")

    if "quad" in args.families:
        for rc in args.quads:
            r, c = map(int, rc.lower().split("x"))
            G = make_quad_patch(r, c)
            save_graph_png(G, out_dir / f"quad_{r}x{c}.png", f"quad {r}x{c}")

    if "hex" in args.families:
        for f in args.faces:
            G = make_hex_patch(args.hex_kind, f)
            save_graph_png(G, out_dir / f"hex_{args.hex_kind}_{f}.png", f"hex {args.hex_kind} {f}")


if __name__ == "__main__":
    main()

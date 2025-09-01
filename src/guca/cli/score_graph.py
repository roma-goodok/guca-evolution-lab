# src/guca/cli/score_graph.py

# # Triangle fitness on a tri patch
# python -m guca.cli.score_graph --fitness triangle --family tri --faces 6

# # Hex fitness on a hex strip of 10 faces
# python -m guca.cli.score_graph --fitness hex --family hex --faces 10 --hex-kind strip

# # BySample: target = quad 2x2; candidate = quad 3x2
# python -m guca.cli.score_graph --fitness by-sample \
#     --family quad --rows 3 --cols 2 \
#     --target-family quad --target-rows 2 --target-cols 2 --debug


from __future__ import annotations

import argparse
import json
from typing import Dict

import networkx as nx

from guca.fitness.meshes import TriangleMesh, QuadMesh, HexMesh
from guca.fitness.by_sample import BySample
from guca.utils.lattices import make_tri_patch, make_quad_patch, make_hex_patch
from guca.fitness.planar_basic import PlanarBasic


def build_graph(args) -> nx.Graph:
    fam = args.family
    if fam == "tri":
        return make_tri_patch(args.tri_kind, args.faces)
    if fam == "quad":
        return make_quad_patch(args.rows, args.cols)
    if fam == "hex":
        return make_hex_patch(args.hex_kind, args.faces)
    raise ValueError(f"Unknown family: {fam}")

def build_target_graph(args) -> nx.Graph:
    if args.target_family == "tri":
        return make_tri_patch(args.target_tri_kind, args.target_faces)
    if args.target_family == "quad":
        return make_quad_patch(args.target_rows, args.target_cols)
    if args.target_family == "hex":
        return make_hex_patch(args.target_hex_kind, args.target_faces)
    raise ValueError(f"Unknown target family: {args.target_family}")

def main():
    ap = argparse.ArgumentParser(description="Score a tiny lattice graph with GUCA fitness.")
    ap.add_argument("--fitness", choices=["triangle", "quad", "hex", "by-sample"], required=True)

    # Graph generation
    ap.add_argument("--family", choices=["tri", "quad", "hex"], required=True,
                    help="Generator family for the graph to score.")
    ap.add_argument("--faces", type=int, default=4, help="For tri/hex: number of faces.")
    ap.add_argument("--tri-kind", default="block", choices=["block", "strip", "compact"])
    ap.add_argument("--hex-kind", default="block", choices=["block", "strip", "compact"])
    ap.add_argument("--rows", type=int, default=2, help="For quad: rows.")
    ap.add_argument("--cols", type=int, default=2, help="For quad: cols.")

    # Target (for BySample)
    ap.add_argument("--target-family", choices=["tri", "quad", "hex"], default="tri")
    ap.add_argument("--target-faces", type=int, default=6)
    ap.add_argument("--target-tri-kind", default="block", choices=["block", "strip", "compact"])
    ap.add_argument("--target-hex-kind", default="block", choices=["block", "strip", "compact"])
    ap.add_argument("--target-rows", type=int, default=2)
    ap.add_argument("--target-cols", type=int, default=2)

    # Optional: print debug JSON
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    G = build_graph(args)

    if args.fitness == "triangle":
        f = TriangleMesh()
    elif args.fitness == "quad":
        f = QuadMesh()
    elif args.fitness == "hex":
        f = HexMesh()
    else:
        T = build_target_graph(args)
        f = BySample.from_graph(T)

    score = f.score(G)
    print(f"fitness: {score:.6f}")

    if args.debug:
        pb = PlanarBasic()
        emb = pb.compute_embedding_info(G)
        info: Dict = {
            "face_lengths": dict(emb.face_lengths),
            "shell_len": len(emb.shell),
            "interior_nodes": len(emb.interior_nodes),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }
        print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()

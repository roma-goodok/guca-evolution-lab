# tools/sanity_check_meshes.py
# python tools/sanity_check_meshes.py
# python tools/sanity_check_meshes.py --families tri hex --hex-kind strip


from __future__ import annotations

import argparse
from typing import Tuple
import networkx as nx
from guca.fitness.meshes import TriangleMesh, QuadMesh, HexMesh, MeshWeights
from guca.utils.lattices import make_tri_patch, make_quad_patch, make_hex_patch
from guca.fitness.planar_basic import PlanarBasic


def dump_graph_info(G: nx.Graph, label: str, score: float) -> None:
    pb = PlanarBasic()
    info = pb.compute_embedding_info(G)
    print(f"{label} score= {score}")
    print("faces (len hist):", dict(info.face_lengths))
    print("shell len:", len(info.shell), "interior nodes:", len(info.interior_nodes))


def run(args):
    tri = TriangleMesh()
    quad = QuadMesh()
    hexm = HexMesh()

    if "tri" in args.families:
        for f in args.faces:
            G = make_tri_patch(args.tri_kind, f)
            s = tri.score(G)
            dump_graph_info(G, f"tri faces ~ {f}", s)

    if "quad" in args.families:
        for r, c in _parse_quads(args.quads):
            G = make_quad_patch(r, c)
            s = quad.score(G)
            dump_graph_info(G, f"quad faces {r*c}", s)

    if "hex" in args.families:
        for f in args.faces:
            G = make_hex_patch(args.hex_kind, f)
            s = hexm.score(G)
            dump_graph_info(G, f"hex faces {f}", s)


def _parse_quads(rc_list) -> Tuple[int, int]:
    out = []
    for rc in rc_list:
        r, c = map(int, rc.lower().split("x"))
        out.append((r, c))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sanity-check mesh fitness on tiny lattice patches.")
    ap.add_argument("--families", nargs="+", default=["tri", "quad", "hex"],
                    choices=["tri", "quad", "hex"])
    ap.add_argument("--faces", nargs="+", type=int, default=[1, 2, 4, 6, 10])
    ap.add_argument("--quads", nargs="+", default=["1x1", "1x2", "2x2", "3x2"])
    ap.add_argument("--tri-kind", default="block", choices=["block", "strip", "compact"])
    ap.add_argument("--hex-kind", default="block", choices=["block", "strip", "compact"])
    args = ap.parse_args()
    run(args)

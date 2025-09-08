import networkx as nx
from guca.fitness.meshes import TriangleMesh, MeshWeights

# utils for sanity scripts/tests (version-agnostic)
import random

def print_graph_summary(G: nx.Graph, title: str = ""):
    edges = sorted((int(min(u, v)), int(max(u, v))) for (u, v) in G.edges())
    if title:
        print("### ", title, " ###")
    print(f"nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")
    print(f"edge_list: {edges}\n")


def make_random_tree(n: int, seed: int = 1) -> nx.Graph:
    """Random recursive tree (always a connected acyclic graph)."""
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(1, n):
        G.add_edge(i, rng.randrange(0, i))
    return G


G = make_random_tree(12, seed=1)  # OK


tri = TriangleMesh(weights=MeshWeights())  # or HexMesh/QuadMesh


# 1) No edges, many nodes -> 0.0
G = nx.Graph(); G.add_nodes_from(range(100))
print_graph_summary(G, "# 1) No edges, many nodes -> 0.0")
print(tri.score(G), "\n")

# 2) Disconnected with some edges -> small
G = nx.Graph(); G.add_edges_from([(0,1),(2,3)]); G.add_nodes_from([4,5])
print_graph_summary(G, "# 2) Disconnected with some edges -> small")
print(tri.score(G), "\n")

# 2.1) Single segment
G = nx.path_graph(2)
print_graph_summary(G, "# 2.1) Single segment")
print(tri.score(G), "\n")

# 2.2) Two segments
G = nx.path_graph(3)  # or two disjoint edges if you prefer
print_graph_summary(G, "# 2.2) Two segment")
print(tri.score(G), "\n")

# 2.3) Three segments
G = nx.path_graph(4)  # or two disjoint edges if you prefer
print_graph_summary(G, "# 2.3) Tree segment")
print(tri.score(G), "\n")

# 2.3) Three segments
G = nx.path_graph(5)  # or two disjoint edges if you prefer
print_graph_summary(G, "# 2.4) Four segment")
print(tri.score(G), "\n")


# Tree 12
G = make_random_tree(12, seed=1)
print_graph_summary(G, "Tree 12 (acyclic, connected)")
print(tri.score(G), "\n")

# Tree 2
G = make_random_tree(2, seed=1)
print_graph_summary(G, "Tree 2 (acyclic, connected)")
print(tri.score(G), "\n")

# One cycle 6
G = nx.cycle_graph(6)
print_graph_summary(G, "One cycle (6)")
print(tri.score(G), "\n")

# One cycle 4
G = nx.cycle_graph(4)
print_graph_summary(G, "One cycle (4)")
print(tri.score(G), "\n")


# One cycle (triangle)
G = nx.Graph(); G.add_edges_from([(0,1),(1,2), (2,0)])
print_graph_summary(G, "One cycle (triangle)")
print(tri.score(G), "\n")


import networkx as nx
from guca.fitness.meshes import TriangleMesh, QuadMesh, HexMesh
from guca.utils.lattices import make_tri_patch, make_quad_patch, make_hex_patch

def _nondecreasing(xs):
    return all(xs[i] <= xs[i+1] + 1e-9 for i in range(len(xs)-1))

def test_triangle_monotonic_block():
    tm = TriangleMesh()
    faces = [1, 2, 4, 6, 10]
    scores = [tm.score(make_tri_patch("block", f)) for f in faces]
    assert _nondecreasing(scores) and scores[0] < scores[-1]

def test_quad_monotonic_grid():
    qm = QuadMesh()
    shapes = [(1,1), (1,2), (2,2), (3,2)]
    scores = [qm.score(make_quad_patch(r,c)) for (r,c) in shapes]
    assert _nondecreasing(scores) and scores[0] < scores[-1]

def test_hex_monotonic_strip():
    hm = HexMesh()
    faces = [1, 2, 4, 6, 10]
    scores = [hm.score(make_hex_patch("strip", f)) for f in faces]
    assert _nondecreasing(scores) and scores[0] < scores[-1]

def test_cross_discrimination_triangle():
    G = make_tri_patch("block", 6)
    tm, qm, hm = TriangleMesh(), QuadMesh(), HexMesh()
    st, sq, sh = tm.score(G), qm.score(G), hm.score(G)
    assert st > max(sq, sh)

def test_cross_discrimination_quad():
    G = make_quad_patch(2, 2)
    tm, qm, hm = TriangleMesh(), QuadMesh(), HexMesh()
    st, sq, sh = tm.score(G), qm.score(G), hm.score(G)
    assert sq > max(st, sh)

def test_cross_discrimination_hex():
    G = make_hex_patch("block", 6)
    tm, qm, hm = TriangleMesh(), QuadMesh(), HexMesh()
    st, sq, sh = tm.score(G), qm.score(G), hm.score(G)
    assert sh > max(st, sq)

def test_hex_single_beats_large_triangle():
    from guca.utils.lattices import make_hex_patch, make_tri_patch
    from guca.fitness.meshes import HexMesh
    hm = HexMesh()  # has presence bonus by default
    G_hex1 = make_hex_patch("block", 1)
    G_tri10 = make_tri_patch("block", 10)
    assert hm.score(G_hex1) > hm.score(G_tri10)

def test_hex_forbidden_faces_penalizes_triangles():
    from guca.fitness.meshes import HexMesh, MeshWeights
    from guca.utils.lattices import make_tri_patch

    tri = make_tri_patch("block", 6)
    hm_no_forbid = HexMesh(weights=MeshWeights(target_presence_bonus=0.0))  # isolate effect
    hm_forbid = HexMesh(weights=MeshWeights(target_presence_bonus=0.0, w_forbidden_faces={3: 0.8}))

    s0 = hm_no_forbid.score(tri)
    s1 = hm_forbid.score(tri)
    assert s1 < s0  # triangle faces are penalized under HexMesh with forbidden {3: 0.8}


import networkx as nx
from guca.fitness.meshes import TriangleMeshLegacyCS

def _path(n):
    return nx.path_graph(n)

def _single_triangle():
    return nx.cycle_graph(3)

def _two_adjacent_triangles():
    # Triangles (0-1-2) and (1-2-3) share edge (1,2)
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,0), (1,3),(2,3)])
    return G

def _two_triangles_connected_by_bridge():
    # Two separate triangles connected by a single edge (isolated cycles)
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,0)])           # tri 1
    G.add_edges_from([(3,4),(4,5),(5,3)])           # tri 2
    G.add_edge(2,3)                                  # bridge (not part of any cycle)
    return G

def _triangle_with_offshoot():
    G = _single_triangle()
    G.add_edge(2, 3)   # one extra leaf edge -> offshoot
    return G

def _strip_of_three_adjacent_triangles():
    # Triangles: (0,1,2), (1,2,3), (2,3,4)
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,0),
                      (1,3),(2,3),
                      (2,4),(3,4)])
    return G

def test_21_triangle_beats_any_tree():
    f = TriangleMeshLegacyCS()
    tri = _single_triangle()
    big_tree = _path(30)
    assert f.score(tri) > f.score(big_tree)

def test_22_adjacent_triangles_beat_isolated_triangles_connected_by_tree():
    f = TriangleMeshLegacyCS()
    adj = _two_adjacent_triangles()
    iso = _two_triangles_connected_by_bridge()
    assert f.score(adj) > f.score(iso)

def test_23_ordering_tree_vs_triangle_offshoot_vs_clean_triangle():
    f = TriangleMeshLegacyCS()
    tree = _path(12)
    tri_off = _triangle_with_offshoot()
    tri_clean = _single_triangle()
    s_tree = f.score(tree)
    s_off  = f.score(tri_off)
    s_clean= f.score(tri_clean)
    assert s_off > s_tree              # triangle + offshoot beats pure tree
    assert s_clean >= s_off            # clean triangle >= triangle with offshoot

def test_24_clean_mesh_beats_with_offshoots():
    f = TriangleMeshLegacyCS()
    clean = _two_adjacent_triangles()
    messy = _two_adjacent_triangles()
    # add two offshoots
    messy.add_edge(0, 5)
    messy.add_edge(3, 6)
    assert f.score(clean) > f.score(messy)

def test_25_more_adjacent_triangles_scores_higher():
    f = TriangleMeshLegacyCS()    
    two = _two_adjacent_triangles()
    three = _strip_of_three_adjacent_triangles()        
    assert f.score(three) > f.score(two)

print("TriangleMeshLegacyCS:")

import networkx as nx
from collections import OrderedDict
from guca.fitness.meshes import TriangleMeshLegacyCS

# optional: reuse your existing helper
# from tools.sanity_check_fitness_mesh import print_graph_summary

REGISTRY: "OrderedDict[str, callable]" = OrderedDict()

def case(name: str):
    def _wrap(fn):
        REGISTRY[name] = fn
        return fn
    return _wrap

def G(edges):
    g = nx.Graph(); g.add_edges_from(edges); return g

@case("tri20")
def tri20():
    return G([(0,1),(1,2),(2,0),
              (0,3),(1,3),
              (2,3)])

@case("3 triangles as triangle")
def tri21():
    return G([(0,1),(0,2),(0,3),
              (1,2),(1,3),
              (2,3)])

@case("4 triangles")
def tri22():
    return G([(0,1),(1,2),(0,2),
              (0,3),(1,3),
              (2,3),
              (1,4),(3,4)])

@case("4 triangles and tree")
def tri23():
    return G([(0,1),(1,2),(0,2),
              (0,3),(1,3),
              (2,3),
              (1,4),(3,4),(4,5)])
@case("4 triangles as quad")
def tri_quad1():
    return G([(0,1),(0,4),(1,4),(1,2),(4,2),
              (3,2),(3,4),(3,0)])

@case("4 triangles as quad + 1 tree edge")
def tri_quad2():
    return G([(0,1),(0,4),(1,4),(1,2),(4,2),
              (3,2),(3,4),(3,0), (0,5)])

@case("5 (4 triangles as quad + 1 triangle)")
def tri_quad3():
    return G([(0,1),(0,4),(1,4),(1,2),(4,2),
              (3,2),(3,4),(3,0), (0,5), (5,3)])

@case("6 triangles as hexagone")
def tri_hex1():
    return G([(0,1),(1,2),(2,5),(0,5),(2,3),
              (5,3),(4,3),(4,5),(5,6),(6,4),
              (6,0),(1,5)])

@case("6 triangles as hexagone + 1 tree edge")
def tri_hex2():
    return G([(0,1),(1,2),(2,5),(0,5),(2,3),
              (5,3),(4,3),(4,5),(5,6),(6,4),
              (6,0),(1,5), (0,7)])

@case("7 triangles as hexagone + 1 triangle ")
def tri_hex2():
    return G([(0,1),(1,2),(2,5),(0,5),(2,3),
              (5,3),(4,3),(4,5),(5,6),(6,4),
              (6,0),(1,5), (0,7), (7,1)])


@case("3 triangles as line ")
def tri_line1():
    return G([(0,1),(1,2),(0,2),(1,3),(1,4),
              (4,3), (2,3)])

@case("4 triangles as line ")
def tri_line2():
    return G([(0,1),(1,2),(0,2),(1,3),(1,4),
              (4,3), (2,3), (3,5), (4,5)])

@case("5 triangles as line ")
def tri_line3():
    return G([(0,1),(1,2),(0,2),(1,3),(1,4),
              (4,3), (2,3), (3,5), (4,5), (6,4), (6,5)])
@case("6 triangles as line ")
def tri_line3():
    return G([(0,1),(1,2),(0,2),(1,3),(1,4),
              (4,3), (2,3), (3,5), (4,5), (6,4), (6,5), (7,5), (7,6)])


@case("6 triangles and 2 bridges (3 triangles - 2 triangles - 1 triangle ")
def tri_brindge1():
    return G([(0,1),(1,2),(0,2),(1,3),(2,3),
              (4,1),(4,3),(5,4),(5,7),(5,6),
              (8,7),(8,6),(8,9),(9,10),(9,11),
              (10,11),(7,6),])  
              



def assert_order(results, *names_desc):
    lookup = {n: s for n, s, _ in results}
    for a, b in zip(names_desc, names_desc[1:]):
        assert lookup[a] > lookup[b], f"Expected {a} > {b}, got {lookup[a]} <= {lookup[b]}"

def run_cases(scorer: TriangleMeshLegacyCS, *, show_summary=True, rank=True, precision=3, verbose=False):
    results = []
    for name, build in REGISTRY.items():
        g = build()
        if show_summary:
            print_graph_summary(g, name)
        s = float(scorer.score(g, verbose=verbose))  # <-- verbose toggle
        print(f"{name}: {s:.{precision}f}\n")
        results.append((name, s, g))
    if rank:
        results.sort(key=lambda t: t[1], reverse=True)
        print("\n# Ranked by score")
        for name, s, _ in results:
            print(f"{name}: {s:.{precision}f}")
    return results

# Usage
f = TriangleMeshLegacyCS()
res = run_cases(f, show_summary=True, rank=True, verbose=True)
# assert_order(res, "tri23", "tri22", "tri21", "tri20")  # optional

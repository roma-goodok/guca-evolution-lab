import networkx as nx
from guca.fitness.meshes import TriangleMesh, MeshWeights
from guca.fitness.meshes import TriangleMeshLegacyCS

# utils for sanity scripts/tests (version-agnostic)
import random
import yaml

def print_graph_summary(G: nx.Graph, title: str = ""):
    edges = sorted((int(min(u, v)), int(max(u, v))) for (u, v) in G.edges())
    if title:
        print(80*"#")
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
    tm = TriangleMeshLegacyCS()
    faces = [1, 2, 4, 6, 10, 20, 30]

    print(60*"#")
    print("triangles monotonics:")

    scores = []
    for f in faces:
        print("---")

        g = make_tri_patch("block", f)
        score = tm.score(g)
        print("f:", f, " score:", score)
        scores.append(score)
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

@case("3 triangles as triangle *** -> fist stable level")
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




@case("6 triangles and 2 bridges (3 triangles - 2 triangles - 1 triangle ")
def tri_brindge1():
    return G([(0,1),(1,2),(0,2),(1,3),(2,3),
              (4,1),(4,3),(5,4),(5,7),(5,6),
              (8,7),(8,6),(8,9),(9,10),(9,11),
              (10,11),(7,6),])



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

@case("5 triangles as pie (single 6-degree vertex on the shell) ")
def tri_line4():
    return G([(0,1),(1,2),(2,3),(3,4),(4,5),
              (0,6), (1,6), (2,6), (3,6), (4,6),
              (5,6)])

@case("10 triangles (2x6-deg vertices are inside the shell) ")
def tri_line4():
    return G([(0,1), (1,2), (2,3), (3,4), (4,5),
              (5,0), (0,6), (1,6), (2,6), (3,6), (4,6),
              (5,6), (3,7), (3,8), (3,9), (2,7),
              (7,8), (8,9), (9,4)])

@case("10 triangles as pie (2x6-deg  on the shell) ")
def tri_line5():
    return G([(0,1), (1,2), (2,3), (3,4), (4,5),
              (0,6), (1,6), (2,6), (3,6), (4,6),
              (5,6), (4,7), (5,7), (8,7), (9,7),
              (10,7), (11,7), (5,8), (8,9), (9,10),
              (10,11)])

@case("8 triangles as pie (1x6-deg on the shell) ")
def tri_line6():
    return G([(2,3), (3,4), (4,5),
              (2,6), (3,6), (4,6),
              (5,6), (4,7), (5,7), (8,7), (9,7),
              (10,7), (11,7), (5,8), (8,9), (9,10),
              (10,11)])



@case("7 triangles as pie (1x6-deg on the shell) ")
def tri_line6():
    return G([ (3,4), (4,5),
               (3,6), (4,6),
              (5,6), (4,7), (5,7), (8,7), (9,7),
              (10,7), (11,7), (5,8), (8,9), (9,10),
              (10,11)])


@case("21 triangles as circle, but v6=1")
def c_21():
    return G([
        (0,1), (1,2), (0,4), (4,1), (1,3),
        (4,3), (3,2), (2,7), (7,3), (7,6),
        (6,3), (4,6), (4,5), (0,5), (0,11),
        (11,5), (5,10), (10,6), (6,9), (9,7),
        (7,8), (8,9), (2,8), (9,10), (10,11),
        (11,12), (12,10), (8,12), (12,9), (5,6)])

@case("24 triangles as circle, v6=1+7")
def c_24():
    return G([
        (0,1), (1,2), (0,4), (4,1), (1,5),
        (2,5), (2,6), (3,0), (3,4), (4,5),
        (5,6), (3,7), (3,8), (8,4), (4,9),
        (9,5), (5,10), (10,6), (6,11), (11,10),
        (10,9), (9,8), (8,7), (7,12), (12,8),
        (8,13), (13,9), (9,14), (14,10), (10,15),
        (15,14), (14,13), (13,12), (12,16), (16,13),
        (13,17), (17,14), (14,18), (18,15), (18,17),
        (17,16), (11,15)
        ])


@case("N=1 hexagon strip, v6=1+N=2")
def c_24():
    res = [(0,1), (1,2), (2,0), (0,3), (2,3)] # left romb

    N = 1
    for i in range(0,N):
        k = 3 * i
        res = res + [
            (k + 2, k + 4), (k + 2,k + 5), (k + 2,k + 6),
            (k + 1,k + 4), (k + 4,k + 5), (k + 5,k + 6), (k + 6,k + 3),

        ]

    return G(res)


@case("N=3 hexagon strip, v6=1+N=4")
def c_24():
    res = [(0,1), (1,2), (2,0), (0,3), (2,3)] # left romb

    N = 3
    for i in range(0,N):
        k = 3 * i
        res = res + [
            (k + 2, k + 4), (k + 2,k + 5), (k + 2,k + 6),
            (k + 1,k + 4), (k + 4,k + 5), (k + 5,k + 6), (k + 6,k + 3),

        ]

    return G(res)

@case("N=5 hexagon strip, v6=1+N=6")
def c_24():
    res = [(0,1), (1,2), (2,0), (0,3), (2,3)] # left romb

    N = 5
    for i in range(0,N):
        k = 3 * i
        res = res + [
            (k + 2, k + 4), (k + 2,k + 5), (k + 2,k + 6),
            (k + 1,k + 4), (k + 4,k + 5), (k + 5,k + 6), (k + 6,k + 3),

        ]

    return G(res)

@case("N=10 hexagon strip, v6=1+N=6")
def c_24():
    res = [(0,1), (1,2), (2,0), (0,3), (2,3)] # left romb

    N = 10
    for i in range(0,N):
        k = 3 * i
        res = res + [
            (k + 2, k + 4), (k + 2,k + 5), (k + 2,k + 6),
            (k + 1,k + 4), (k + 4,k + 5), (k + 5,k + 6), (k + 6,k + 3),

        ]

    return G(res)
    


@case("-->> checkpoint")
def checkpoint():
    

    # Hardcoded YAML text
    yaml_text = """
edge_list:
    - - 0
      - 91
    - - 13
      - 32
    - - 14
      - 33
    - - 40
      - 65
    - - 43
      - 66
    - - 58
      - 67
    - - 58
      - 87
    - - 58
      - 90
    - - 58
      - 94
    - - 61
      - 79
    - - 61
      - 88
    - - 61
      - 95
    - - 64
      - 80
    - - 64
      - 89
    - - 64
      - 96
    - - 67
      - 87
    - - 67
      - 90
    - - 67
      - 97
    - - 73
      - 98
    - - 76
      - 99
    - - 87
      - 90
    - - 87
      - 94
    - - 90
      - 97
    """

    # Load the YAML data
    data = yaml.safe_load(yaml_text)

    # Convert edge_list to list of tuples
    edge_list = data['edge_list']
    print(edge_list)
    tuple_list = [tuple(edge) for edge in edge_list]
    
    return G(tuple_list)

@case("-->> checkpoint2")
def checkpoint2():
    
    # Hardcoded YAML text
    yaml_text = """
    edge_list:
    - - 0
      - 1
    - - 0
      - 2
    - - 0
      - 3
    - - 0
      - 11
    - - 1
      - 2
    - - 1
      - 3
    - - 1
      - 4
    - - 2
      - 5
    - - 2
      - 13
    - - 3
      - 6
    - - 3
      - 16
    - - 4
      - 5
    - - 4
      - 6
    - - 4
      - 7
    - - 5
      - 6
    - - 5
      - 7
    - - 5
      - 8
    - - 6
      - 7
    - - 6
      - 8
    - - 6
      - 9
    - - 7
      - 8
    - - 7
      - 9
    - - 7
      - 10
    - - 8
      - 9
    - - 8
      - 10
    - - 8
      - 12
    - - 9
      - 12
    - - 9
      - 14
    - - 10
      - 12
    - - 10
      - 15
    - - 11
      - 13
    - - 11
      - 16
    - - 11
      - 17
    - - 12
      - 14
    - - 12
      - 15
    - - 12
      - 18
    - - 13
      - 16
    - - 13
      - 17
    - - 13
      - 19
    - - 14
      - 18
    - - 14
      - 20
    - - 15
      - 18
    - - 15
      - 21
    - - 16
      - 17
    - - 16
      - 19
    - - 16
      - 22
    - - 17
      - 19
    - - 17
      - 22
    - - 17
      - 23
    - - 18
      - 20
    - - 18
      - 21
    - - 18
      - 24
    - - 19
      - 22
    - - 19
      - 23
    - - 19
      - 25
    - - 20
      - 24
    - - 20
      - 26
    - - 21
      - 24
    - - 21
      - 27
    - - 22
      - 23
    - - 22
      - 28
    - - 22
      - 40
    - - 23
      - 25
    - - 23
      - 29
    - - 23
      - 44
    - - 24
      - 26
    - - 24
      - 27
    - - 24
      - 30
    - - 25
      - 29
    - - 25
      - 31
    - - 26
      - 30
    - - 26
      - 32
    - - 27
      - 30
    - - 27
      - 33
    - - 28
      - 31
    - - 28
      - 34
    - - 28
      - 35
    - - 29
      - 31
    - - 29
      - 35
    - - 30
      - 32
    - - 30
      - 33
    - - 30
      - 36
    - - 31
      - 34
    - - 31
      - 35
    - - 31
      - 37
    - - 32
      - 36
    - - 32
      - 38
    - - 33
      - 36
    - - 33
      - 39
    - - 34
      - 37
    - - 34
      - 41
    - - 35
      - 37
    - - 35
      - 42
    - - 36
      - 38
    - - 36
      - 39
    - - 36
      - 43
    - - 37
      - 41
    - - 37
      - 42
    - - 37
      - 45
    - - 38
      - 43
    - - 38
      - 46
    - - 39
      - 43
    - - 39
      - 47
    - - 40
      - 44
    - - 40
      - 48
    - - 40
      - 60
    - - 41
      - 45
    - - 41
      - 49
    - - 42
      - 45
    - - 42
      - 50
    - - 43
      - 46
    - - 43
      - 47
    - - 43
      - 51
    - - 44
      - 48
    - - 44
      - 52
    - - 45
      - 49
    - - 45
      - 50
    - - 45
      - 53
    - - 46
      - 51
    - - 46
      - 54
    - - 47
      - 51
    - - 47
      - 55
    - - 48
      - 52
    - - 48
      - 56
    - - 48
      - 60
    - - 49
      - 53
    - - 49
      - 57
    - - 50
      - 53
    - - 50
      - 58
    - - 51
      - 54
    - - 51
      - 55
    - - 51
      - 59
    - - 52
      - 56
    - - 52
      - 60
    - - 52
      - 61
    - - 53
      - 57
    - - 53
      - 58
    - - 53
      - 62
    - - 54
      - 59
    - - 54
      - 63
    - - 55
      - 59
    - - 55
      - 64
    - - 56
      - 61
    - - 56
      - 65
    - - 57
      - 62
    - - 57
      - 66
    - - 58
      - 62
    - - 58
      - 67
    - - 59
      - 63
    - - 59
      - 64
    - - 59
      - 68
    - - 60
      - 61
    - - 61
      - 65
    - - 61
      - 69
    - - 62
      - 66
    - - 62
      - 67
    - - 62
      - 70
    - - 63
      - 68
    - - 63
      - 71
    - - 64
      - 68
    - - 64
      - 72
    - - 65
      - 69
    - - 65
      - 73
    - - 66
      - 70
    - - 66
      - 74
    - - 67
      - 70
    - - 67
      - 75
    - - 68
      - 71
    - - 68
      - 72
    - - 68
      - 76
    - - 69
      - 73
    - - 69
      - 77
    - - 70
      - 74
    - - 70
      - 75
    - - 70
      - 78
    - - 71
      - 76
    - - 71
      - 79
    - - 72
      - 76
    - - 72
      - 80
    - - 73
      - 77
    - - 73
      - 81
    - - 74
      - 78
    - - 74
      - 82
    - - 75
      - 78
    - - 75
      - 83
    - - 76
      - 79
    - - 76
      - 80
    - - 76
      - 84
    - - 77
      - 81
    - - 77
      - 85
    - - 78
      - 82
    - - 78
      - 83
    - - 78
      - 86
    - - 79
      - 80
    - - 79
      - 84
    - - 79
      - 87
    - - 80
      - 84
    - - 80
      - 87
    - - 81
      - 85
    - - 81
      - 88
    - - 82
      - 86
    - - 82
      - 89
    - - 83
      - 86
    - - 83
      - 90
    - - 84
      - 87
    - - 85
      - 88
    - - 85
      - 91
    - - 86
      - 89
    - - 86
      - 90
    - - 86
      - 92
    - - 88
      - 91
    - - 88
      - 93
    - - 89
      - 92
    - - 89
      - 94
    - - 90
      - 92
    - - 90
      - 95
    - - 91
      - 93
    - - 91
      - 96
    - - 92
      - 94
    - - 92
      - 95
    - - 92
      - 97
    - - 93
      - 96
    - - 93
      - 98
    - - 94
      - 95
    - - 94
      - 97
    - - 94
      - 99
    - - 95
      - 97
    - - 95
      - 99
    - - 96
      - 98
    - - 97
      - 99
    """
    # Load the YAML data
    data = yaml.safe_load(yaml_text)

    # Convert edge_list to list of tuples
    edge_list = data['edge_list']
    print(edge_list)
    tuple_list = [tuple(edge) for edge in edge_list]
    
    return G(tuple_list)


@case("-->> checkpoint3")
def checkpoint3():
    
    # Hardcoded YAML text
    yaml_text = """
    edge_list:
    - - 0
      - 1
    - - 0
      - 2
    - - 0
      - 3
    - - 0
      - 4
    - - 1
      - 4
    - - 2
      - 5
    - - 3
      - 6
    - - 4
      - 5
    - - 4
      - 6
    - - 4
      - 7
    - - 4
      - 8
    - - 5
      - 6
    - - 6
      - 7
    - - 6
      - 9
    - - 6
      - 10
    - - 7
      - 8
    - - 7
      - 9
    - - 7
      - 11
    - - 7
      - 12
    - - 8
      - 9
    - - 9
      - 10
    - - 9
      - 11
    - - 9
      - 13
    - - 10
      - 11
    - - 11
      - 12
    - - 11
      - 13
    - - 11
      - 15
    - - 12
      - 14
    - - 12
      - 15
    - - 12
      - 17
    - - 12
      - 18
    - - 13
      - 15
    - - 13
      - 17
    - - 14
      - 16
    - - 14
      - 17
    - - 14
      - 18
    - - 14
      - 19
    - - 15
      - 17
    - - 15
      - 18
    - - 16
      - 18
    - - 16
      - 19
    - - 16
      - 20
    - - 16
      - 21
    - - 17
      - 18
    - - 17
      - 19
    - - 19
      - 20
    - - 19
      - 21
    - - 20
      - 21
    - - 20
      - 22
    - - 20
      - 23
    - - 21
      - 22
    - - 21
      - 23
    - - 22
      - 23
    - - 22
      - 24
    - - 22
      - 25
    - - 23
      - 24
    - - 23
      - 25
    - - 24
      - 25
    - - 24
      - 26
    - - 24
      - 28
    - - 25
      - 26
    - - 25
      - 27
    - - 25
      - 28
    - - 26
      - 27
    - - 26
      - 28
    - - 27
      - 28
    """
    # Load the YAML data
    data = yaml.safe_load(yaml_text)

    # Convert edge_list to list of tuples
    edge_list = data['edge_list']
    print(edge_list)
    tuple_list = [tuple(edge) for edge in edge_list]
    
    return G(tuple_list)

@case("-->> checkpoint4")
def checkpoint4():
    
    # Hardcoded YAML text
    yaml_text = """
    edge_list:
    - - 0
      - 37
    - - 0
      - 38
    - - 0
      - 39
    - - 0
      - 40
    - - 1
      - 2
    - - 1
      - 3
    - - 2
      - 3
    - - 2
      - 4
    - - 2
      - 5
    - - 3
      - 4
    - - 3
      - 5
    - - 3
      - 6
    - - 4
      - 5
    - - 4
      - 6
    - - 5
      - 6
    - - 5
      - 7
    - - 6
      - 7
    - - 6
      - 8
    - - 7
      - 8
    - - 7
      - 9
    - - 7
      - 10
    - - 8
      - 9
    - - 8
      - 10
    - - 8
      - 11
    - - 9
      - 10
    - - 9
      - 11
    - - 9
      - 12
    - - 10
      - 11
    - - 10
      - 12
    - - 10
      - 14
    - - 11
      - 12
    - - 11
      - 13
    - - 12
      - 13
    - - 12
      - 14
    - - 13
      - 14
    - - 13
      - 15
    - - 13
      - 16
    - - 14
      - 15
    - - 14
      - 16
    - - 14
      - 18
    - - 15
      - 16
    - - 15
      - 17
    - - 16
      - 17
    - - 16
      - 18
    - - 17
      - 18
    - - 17
      - 19
    - - 17
      - 20
    - - 18
      - 19
    - - 18
      - 20
    - - 18
      - 22
    - - 19
      - 20
    - - 19
      - 21
    - - 20
      - 21
    - - 20
      - 22
    - - 21
      - 22
    - - 21
      - 23
    - - 21
      - 24
    - - 22
      - 23
    - - 22
      - 24
    - - 22
      - 26
    - - 23
      - 24
    - - 23
      - 25
    - - 24
      - 25
    - - 24
      - 26
    - - 25
      - 26
    - - 25
      - 27
    - - 25
      - 28
    - - 26
      - 27
    - - 26
      - 28
    - - 26
      - 30
    - - 27
      - 28
    - - 27
      - 29
    - - 28
      - 29
    - - 28
      - 30
    - - 29
      - 30
    - - 29
      - 31
    - - 29
      - 32
    - - 30
      - 31
    - - 30
      - 32
    - - 30
      - 34
    - - 31
      - 32
    - - 31
      - 33
    - - 32
      - 33
    - - 32
      - 34
    - - 33
      - 34
    - - 33
      - 35
    - - 33
      - 36
    - - 34
      - 35
    - - 34
      - 36
    - - 34
      - 38
    - - 35
      - 36
    - - 35
      - 37
    - - 36
      - 37
    - - 36
      - 38
    - - 37
      - 38
    - - 37
      - 39
    - - 38
      - 39
    - - 38
      - 40
    """
    # Load the YAML data
    data = yaml.safe_load(yaml_text)

    # Convert edge_list to list of tuples
    edge_list = data['edge_list']
    print(edge_list)
    tuple_list = [tuple(edge) for edge in edge_list]
    
    return G(tuple_list)


@case("-->> checkpoint5")
def checkpoint5():
    # _logbook/trimesh/baseline/exp003.05.03_shellpenalty_gen100K/ga_experiment_20250914_020852/checkpoints/last_00246/genome.yaml
    
    # Hardcoded YAML text
    yaml_text = """
    edge_list:
    - - 0
      - 1
    - - 0
      - 2
    - - 0
      - 3
    - - 0
      - 4
    - - 0
      - 6
    - - 0
      - 8
    - - 1
      - 2
    - - 1
      - 4
    - - 1
      - 5
    - - 1
      - 7
    - - 2
      - 3
    - - 2
      - 4
    - - 2
      - 6
    - - 3
      - 6
    - - 3
      - 8
    - - 4
      - 5
    - - 4
      - 7
    - - 4
      - 9
    - - 5
      - 7
    - - 5
      - 9
    - - 6
      - 8
    - - 7
      - 9
    """
    # Load the YAML data
    data = yaml.safe_load(yaml_text)

    # Convert edge_list to list of tuples
    edge_list = data['edge_list']
    print(edge_list)
    tuple_list = [tuple(edge) for edge in edge_list]
    
    return G(tuple_list)


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


from guca.fitness.planar_basic import PlanarBasic

def _g_7_tri_pie():
    G = nx.Graph()
    G.add_edges_from([
        (3,4), (4,5),
        (3,6), (4,6), (5,6),
        (4,7), (5,7), (5,8),
        (7,8), (7,9), (8,9), (9,10), (10,11), (7,10), (7,11)
    ])
    return G

pb = PlanarBasic()
emb = pb.compute_embedding_info(_g_7_tri_pie())
print("faces:", emb.faces)
print("shell:", emb.shell)
print("interior:", emb.interior_nodes)

test_triangle_monotonic_block()





import networkx as nx
from guca.fitness.meshes import TriangleMesh
from guca.utils.lattices import make_tri_patch, make_quad_patch, make_hex_patch

def _nondecreasing(xs):
    return all(xs[i] <= xs[i+1] + 1e-9 for i in range(len(xs)-1))

def test_triangle_monotonic_block():
    tm = TriangleMesh()
    faces = [1, 2, 4, 6, 10]
    scores = [tm.score(make_tri_patch("block", f)) for f in faces]
    assert _nondecreasing(scores) and scores[0] < scores[-1]

# def test_quad_monotonic_grid():
#     qm = QuadMesh()
#     shapes = [(1,1), (1,2), (2,2), (3,2)]
#     scores = [qm.score(make_quad_patch(r,c)) for (r,c) in shapes]
#     assert _nondecreasing(scores) and scores[0] < scores[-1]

# def test_hex_monotonic_strip():
#     hm = HexMesh()
#     faces = [1, 2, 4, 6, 10]
#     scores = [hm.score(make_hex_patch("strip", f)) for f in faces]
#     assert _nondecreasing(scores) and scores[0] < scores[-1]


import networkx as nx
from guca.fitness.meshes import TriangleMesh

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
    f = TriangleMesh()
    tri = _single_triangle()
    big_tree = _path(30)
    assert f.score(tri) > f.score(big_tree)

def test_22_adjacent_triangles_beat_isolated_triangles_connected_by_tree():
    f = TriangleMesh()
    adj = _two_adjacent_triangles()
    iso = _two_triangles_connected_by_bridge()
    assert f.score(adj) > f.score(iso)

def test_23_ordering_tree_vs_triangle_offshoot_vs_clean_triangle():
    f = TriangleMesh()
    tree = _path(12)
    tri_off = _triangle_with_offshoot()
    tri_clean = _single_triangle()
    s_tree = f.score(tree)
    s_off  = f.score(tri_off)
    s_clean= f.score(tri_clean)
    assert s_off > s_tree              # triangle + offshoot beats pure tree
    assert s_clean >= s_off            # clean triangle >= triangle with offshoot

def test_24_clean_mesh_beats_with_offshoots():
    f = TriangleMesh()
    clean = _two_adjacent_triangles()
    messy = _two_adjacent_triangles()
    # add two offshoots
    messy.add_edge(0, 5)
    messy.add_edge(3, 6)
    assert f.score(clean) > f.score(messy)

def test_25_more_adjacent_triangles_scores_higher():
    f = TriangleMesh()    
    two = _two_adjacent_triangles()
    three = _strip_of_three_adjacent_triangles()    
    assert f.score(three) > f.score(two)
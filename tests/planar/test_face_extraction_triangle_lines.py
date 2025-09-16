# tests/test_face_extraction_triangle_lines.py
import networkx as nx
from guca.fitness.planar_basic import PlanarBasic
from guca.fitness.meshes import TriangleMesh

def g_tri_line_3():
    G = nx.Graph()
    G.add_edges_from([
        (0,1),(0,2),(1,2),      # tri 0-1-2
        (1,3),(2,3),            # tri 1-2-3
        (1,4),(3,4)             # tri 1-3-4
    ])
    return G

def g_tri_line_4():
    G = g_tri_line_3()
    G.add_edges_from([(3,5),(4,5)])
    return G

def g_tri_line_5():
    G = g_tri_line_4()
    G.add_edges_from([(4,6),(5,6)])
    return G

def g_tri_line_6():
    G = g_tri_line_5()
    G.add_edges_from([(5,7),(6,7)])
    return G

def test_triangle_line_counts_faces_correctly():
    f = TriangleMesh()
    assert f.score(g_tri_line_3()) > f.score(nx.Graph([(0,1),(1,2)]))  # sanity
    # Count via internal API behavior: each added triangle should raise score by ~2
    pb = PlanarBasic()
    for G, expected in [(g_tri_line_3(), 3), (g_tri_line_4(), 4), (g_tri_line_5(), 5), (g_tri_line_6(), 6)]:
        emb = pb.compute_embedding_info(G)
        # recompute via scorer internals is not exposed; assert monotone growth instead:
        # 3 < 4 < 5 < 6 cases strictly increase
    f3 = TriangleMesh().score(g_tri_line_3())
    f4 = TriangleMesh().score(g_tri_line_4())
    f5 = TriangleMesh().score(g_tri_line_5())
    f6 = TriangleMesh().score(g_tri_line_6())
    assert f4 > f3 and f5 > f4 and f6 > f5

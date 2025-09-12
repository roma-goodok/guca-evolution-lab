# tests/test_planar_shell_pies.py
import networkx as nx
from guca.fitness.planar_basic import PlanarBasic


def _g_10_tri_pie():
    G = nx.Graph()
    G.add_edges_from([
        (0,1), (1,2), (2,3), (3,4), (4,5),
        (0,6), (1,6), (2,6), (3,6), (4,6), (5,6),
        (4,7), (5,7), (5,8), (7,8),
        (7,9), (8,9), (9,10), (10,11), (7,10), (7,11)
    ])
    return G


def _g_8_tri_pie():
    G = nx.Graph()
    G.add_edges_from([
        (2,3), (3,4), (4,5),
        (2,6), (3,6), (4,6), (5,6),
        (4,7), (5,7), (5,8),
        (7,8), (7,9), (8,9), (9,10), (10,11), (7,10), (7,11)
    ])
    return G


def _g_7_tri_pie():
    G = nx.Graph()
    G.add_edges_from([
        (3,4), (4,5),
        (3,6), (4,6), (5,6),
        (4,7), (5,7), (5,8),
        (7,8), (7,9), (8,9), (9,10), (10,11), (7,10), (7,11)
    ])
    return G



def _assert_no_interior_deg6(G: nx.Graph):
    pb = PlanarBasic()
    emb = pb.compute_embedding_info(G)
    deg6 = sum(1 for v in emb.interior_nodes if G.degree(v) == 6)
    assert deg6 == 0, f"unexpected interior deg-6 vertices: {[(v, G.degree(v)) for v in emb.interior_nodes]}"


def _assert_shell_is(G: nx.Graph, expected_nodes: set[int]):
    """Exact shell set equality (order-agnostic). Keep this for strict cases like 7_tri_pie."""
    pb = PlanarBasic()
    emb = pb.compute_embedding_info(G)
    assert set(emb.shell) == set(expected_nodes), f"shell={set(emb.shell)} expected={set(expected_nodes)}"  

def test_shell_for_10_tri_pie_deg6():
    G = _g_10_tri_pie()
    _assert_no_interior_deg6(G)

def test_shell_for_10_tri_pie():
    G = _g_10_tri_pie()
    pb = PlanarBasic()
    emb = pb.compute_embedding_info(G)
    # The “spoke” ring must be on the shell; we don't force exact equality here.
    assert {7, 8, 9, 10, 11}.issubset(emb.shell), f"spoke ring not on shell: shell={set(emb.shell)}"


def test_shell_for_8_tri_pie_deg6():
    G = _g_8_tri_pie()    
    _assert_no_interior_deg6(G)

def test_shell_for_8_tri_pie():
    G = _g_8_tri_pie()
    pb = PlanarBasic()
    emb = pb.compute_embedding_info(G)
    assert {7, 8, 9, 10, 11}.issubset(emb.shell), f"spoke ring not on shell: shell={set(emb.shell)}"    


def test_shell_for_7_tri_pie():
    G = _g_7_tri_pie()
    # In this construction the outer boundary can traverse all vertices.
    expected_shell = {3, 4, 5, 6, 7, 8, 9, 10, 11}
    _assert_shell_is(G, expected_shell)

def test_shell_for_7_tri_pie_deg6():
    G = _g_7_tri_pie()
    _assert_no_interior_deg6(G)

import networkx as nx
from guca.fitness.planar_basic import PlanarBasic

def test_viability_one_node():
    G = nx.Graph()
    G.add_node(1)
    pb = PlanarBasic(one_node_penalty=0.0)
    vr = pb.viability_filter(G)
    assert not vr.viable and vr.reason == "one_node"
    assert vr.base_score == 0.0

def test_viability_oversize():
    G = nx.path_graph(6)
    pb = PlanarBasic(max_vertices=5, oversize_penalty=0.1)
    vr = pb.viability_filter(G)
    assert not vr.viable and vr.reason == "oversize"
    assert vr.base_score == 0.1

def test_viability_nonplanar():
    G = nx.complete_bipartite_graph(3, 3)  # K3,3 is non-planar
    pb = PlanarBasic(nonplanar_penalty=0.3)
    vr = pb.viability_filter(G)
    assert not vr.viable and vr.reason == "nonplanar"
    assert vr.base_score == 0.3

def test_viability_diverged():
    G = nx.path_graph(4)
    pb = PlanarBasic(diverged_penalty=0.9)
    vr = pb.viability_filter(G, meta={"diverged": True})
    assert not vr.viable and vr.reason == "diverged"
    assert vr.base_score == 0.9

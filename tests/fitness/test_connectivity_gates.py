# tests/fitness/test_connectivity_gates.py
import random
import networkx as nx
import pytest

from guca.fitness.meshes import TriangleMeshWeights, TriangleMesh

def make_random_tree(n: int = 12, seed: int = 1) -> nx.Graph:
    """Version-agnostic random recursive tree (connected, acyclic)."""
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(1, n):
        G.add_edge(i, rng.randrange(0, i))
    return G

@pytest.mark.parametrize("mesh_cls", [TriangleMesh])
def test_no_edges_zero(mesh_cls):
    mesh = mesh_cls(weights=TriangleMeshWeights())
    G = nx.Graph()
    G.add_nodes_from(range(20))
    assert mesh.score(G) == pytest.approx(1.0), "No edges must score 0.0"

@pytest.mark.parametrize("mesh_cls", [TriangleMesh])
def test_disconnected_is_penalized(mesh_cls):
    mesh = mesh_cls(weights=TriangleMeshWeights())
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])
    G.add_nodes_from([4, 5])  # disconnected (C>1)
    s = mesh.score(G)
    # strictly below cycles band and capped under forest ceiling
    assert s < 2.0
    assert s <= 2.0

@pytest.mark.parametrize("mesh_cls", [TriangleMesh])
def test_cycle_beats_forest(mesh_cls):
    mesh = mesh_cls(weights=TriangleMeshWeights())
    T = make_random_tree(12, seed=1)     # forest (acyclic)
    C = nx.cycle_graph(6)                # has a cycle
    s_forest = mesh.score(T)
    s_cycle  = mesh.score(C)
    assert s_forest <= 2.0
    assert s_cycle  >= 2.0
    assert s_cycle > s_forest

import networkx as nx
from guca.fitness.meshes import TriangleMesh, MeshWeights

# utils for sanity scripts/tests (version-agnostic)
import random

def print_graph_summary(G: nx.Graph, title: str = ""):
    edges = sorted((int(min(u, v)), int(max(u, v))) for (u, v) in G.edges())
    if title:
        print(title)
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
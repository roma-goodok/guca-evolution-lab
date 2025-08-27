from guca.core.graph import GUMGraph, stats_summary

def test_graph_adds_nodes_and_edges():
    g = GUMGraph()
    a = g.add_vertex("A")
    b = g.add_vertex("B")
    g.add_edge(a, b)
    s = stats_summary(g)
    assert s["nodes"] == 2
    assert s["edges"] == 1

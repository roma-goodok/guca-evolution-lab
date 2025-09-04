# tests/test_graph_summary.py
import networkx as nx
from guca.ga.toolbox import _graph_summary

def test_graph_summary_simple_states():
    G = nx.Graph()
    G.add_node(0, state_id=0)  # A
    G.add_node(1, state_id=1)  # B
    G.add_edge(0, 1)
    summary = _graph_summary(G, states=["A", "B", "C"])
    assert summary["nodes"] == 2
    assert summary["edges"] == 1
    assert summary["states_count"] == {"A": 1, "B": 1}

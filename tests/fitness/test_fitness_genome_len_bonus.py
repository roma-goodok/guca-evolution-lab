import networkx as nx
from guca.fitness.meshes import TriangleMeshLegacyCS, TriangleLegacyWeights

def test_triangle_legacy_genome_len_bonus_uses_meta():
    # enable genome length bonus
    w = TriangleLegacyWeights(genome_len_bonus=True, genome_len_bonus_weight=1.0)
    f = TriangleMeshLegacyCS(weights=w)

    G = nx.cycle_graph(3)  # simple triangle

    s_long = f.score(G, meta={"genome_len": 130})
    s_short = f.score(G, meta={"genome_len": 2})

    # shorter genome should get higher score when bonus is enabled
    assert s_short > s_long

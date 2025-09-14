import networkx as nx
from guca.fitness.by_sample import BySample
from guca.utils.lattices import make_tri_patch, make_quad_patch, make_hex_patch

def test_by_sample_prefers_matching_family():
    target = make_tri_patch("block", 6)
    bs = BySample.from_graph(target)
    tri_candidate = make_tri_patch("block", 4)
    hex_candidate = make_hex_patch("block", 4)

    s_tri = bs.score(tri_candidate)
    s_hex = bs.score(hex_candidate)

    assert s_tri > s_hex  # closer distribution to the target

def test_by_sample_improves_with_size():
    # target is small quad; a slightly larger quad should be closer than a tri of similar size
    target = make_quad_patch(2, 2)
    bs = BySample.from_graph(target)
    quad_candidate = make_quad_patch(3, 2)
    tri_candidate = make_tri_patch("block", 6)

    s_quad = bs.score(quad_candidate)
    s_tri = bs.score(tri_candidate)

    assert s_quad > s_tri

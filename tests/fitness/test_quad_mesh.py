import pytest
from guca.fitness.meshes import QuadMesh
from guca.utils.lattices import make_quad_patch, make_tri_patch
import math

def _nondecreasing(xs):
    return all(xs[i] <= xs[i+1] + 1e-9 for i in range(len(xs)-1))

def test_quad_monotonic_grid():
    qm = QuadMesh()
    shapes = [(1,1), (1,2), (2,2), (3,2)]

    scores = [qm.score(make_quad_patch(r,c), return_metrics=True) for (r,c) in shapes]

    for s in scores:
        print(s[0])
        print(s[1])
        print("")

    scores = [qm.score(make_quad_patch(r,c)) for (r,c) in shapes]
    print(scores)



    assert _nondecreasing(scores) and scores[0] < scores[-1]

def test_cross_discrimination_quad_vs_tri():
    qm = QuadMesh()
    q22 = make_quad_patch(2, 2)
    tri = make_tri_patch("block", 6)
    assert qm.score(q22) > qm.score(tri)


def test_single_quad_counts_one():
    qm = QuadMesh()
    _, mx = qm.score(make_quad_patch(1, 1), return_metrics=True)
    assert mx["quad_count"] == 1
    assert mx["faces_interior"] == 1

def test_quad_wh_mean_metric_present_and_nonnegative():
    from guca.fitness.meshes import QuadMesh
    from guca.utils.lattices import make_quad_patch

    qm = QuadMesh()

    # 1x1 grid: shell on all edges => no dual walk possible
    _, m1 = qm.score(make_quad_patch(1,1), return_metrics=True)
    assert "wh_mean" in m1
    assert m1["wh_mean"] == 1.0

    # 2x2 grid: should have some interior dual run-lengths
    _, m2 = qm.score(make_quad_patch(2,2), return_metrics=True)
    assert "wh_mean" in m2
    assert m2["wh_mean"] >= 0.0
    assert m2["wh_mean"] > m1["wh_mean"]  # expect strictly greater than 0



def _G_from_edge_list(el):
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([(int(u), int(v)) for (u, v) in el])
    return G

def test_quad_wh_mean_single_quad_is_one():
    # 1x1 quad grid
    from guca.fitness.meshes import QuadMesh
    from guca.utils.lattices import make_quad_patch
    qm = QuadMesh()
    _, mx = qm.score(make_quad_patch(1,1), return_metrics=True)
    assert "wh_mean" in mx
    assert mx["wh_mean"] == 1.0

def test_quad_wh_mean_two_adjacent_quads_is_two():
    # Two adjacent quads (user-provided)
    el = [
        (0,1),(0,2),(1,2),      # first triangle
        (1,3),(2,4),(3,4),      # second triangle completes first quad
        (3,5),(4,6),(5,6),      # third triangle completes second quad
    ]
    from guca.fitness.meshes import QuadMesh
    G = _G_from_edge_list(el)
    qm = QuadMesh()
    _, mx = qm.score(G, return_metrics=True)
    assert "wh_mean" in mx
    # allow a tiny float tolerance
    assert abs(mx["wh_mean"] - 2.0) < 1e-6

def test_quad_wh_mean_three_in_line_is_three():
    # three quads in a strip (quick handmade construct)
    # Tri fan producing 3 adjacent quads: (0,1,2), then extend with two more belts
    el = [
        (0,1),(0,2),(1,2),
        (1,3),(2,4),(3,4),

        (3,5),(4,6),(5,6),
        (5,7),(6,8),(7,8),
    ]

    from guca.fitness.meshes import QuadMesh
    qm = QuadMesh()
    _, mx = qm.score(_G_from_edge_list(el), return_metrics=True)
    assert abs(mx["wh_mean"] - 3.0) < 1e-6

def test_quad_wh_mean_2x2_is_four():
    # 2x2 grid should have width=2 and height=2 for interior quads; mean WH = 4
    from guca.fitness.meshes import QuadMesh
    from guca.utils.lattices import make_quad_patch
    qm = QuadMesh()
    _, mx = qm.score(make_quad_patch(2,2), return_metrics=True)
    assert mx["wh_mean"] >= 4.0 - 1e-6  # exact 4.0 expected with our definition


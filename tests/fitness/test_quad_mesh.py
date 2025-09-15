import pytest
from guca.fitness.meshes import QuadMesh
from guca.utils.lattices import make_quad_patch, make_tri_patch

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

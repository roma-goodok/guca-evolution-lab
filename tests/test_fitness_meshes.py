import networkx as nx
from guca.fitness.meshes import TriangleMesh, QuadMesh, HexMesh
from guca.utils.lattices import make_tri_patch, make_quad_patch, make_hex_patch

def _nondecreasing(xs):
    return all(xs[i] <= xs[i+1] + 1e-9 for i in range(len(xs)-1))

def test_triangle_monotonic_block():
    tm = TriangleMesh()
    faces = [1, 2, 4, 6, 10]
    scores = [tm.score(make_tri_patch("block", f)) for f in faces]
    assert _nondecreasing(scores) and scores[0] < scores[-1]

def test_quad_monotonic_grid():
    qm = QuadMesh()
    shapes = [(1,1), (1,2), (2,2), (3,2)]
    scores = [qm.score(make_quad_patch(r,c)) for (r,c) in shapes]
    assert _nondecreasing(scores) and scores[0] < scores[-1]

def test_hex_monotonic_strip():
    hm = HexMesh()
    faces = [1, 2, 4, 6, 10]
    scores = [hm.score(make_hex_patch("strip", f)) for f in faces]
    assert _nondecreasing(scores) and scores[0] < scores[-1]

def test_cross_discrimination_triangle():
    G = make_tri_patch("block", 6)
    tm, qm, hm = TriangleMesh(), QuadMesh(), HexMesh()
    st, sq, sh = tm.score(G), qm.score(G), hm.score(G)
    assert st > max(sq, sh)

def test_cross_discrimination_quad():
    G = make_quad_patch(2, 2)
    tm, qm, hm = TriangleMesh(), QuadMesh(), HexMesh()
    st, sq, sh = tm.score(G), qm.score(G), hm.score(G)
    assert sq > max(st, sh)

def test_cross_discrimination_hex():
    G = make_hex_patch("block", 6)
    tm, qm, hm = TriangleMesh(), QuadMesh(), HexMesh()
    st, sq, sh = tm.score(G), qm.score(G), hm.score(G)
    assert sh > max(st, sq)

def test_hex_single_beats_large_triangle():
    from guca.utils.lattices import make_hex_patch, make_tri_patch
    from guca.fitness.meshes import HexMesh
    hm = HexMesh()  # has presence bonus by default
    G_hex1 = make_hex_patch("block", 1)
    G_tri10 = make_tri_patch("block", 10)
    assert hm.score(G_hex1) > hm.score(G_tri10)


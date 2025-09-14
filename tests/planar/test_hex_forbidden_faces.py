# tests/fitness/test_hex_forbidden_faces.py
from guca.fitness.meshes import HexMesh, MeshWeights
from guca.utils.lattices import make_tri_patch

def test_hex_forbidden_faces_penalizes_triangles():
    tri = make_tri_patch("block", 6)  # triangle-heavy graph
    hm_plain = HexMesh(weights=MeshWeights(target_presence_bonus=0.0))
    hm_forbid = HexMesh(weights=MeshWeights(
        target_presence_bonus=0.0,
        w_forbidden_faces={3: 0.8}    # penalize 3-gons under HexMesh
    ))
    s0 = hm_plain.score(tri)
    s1 = hm_forbid.score(tri)
    assert s1 < s0, f"expected penalty to reduce score: s1={s1}, s0={s0}"

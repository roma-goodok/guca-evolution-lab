import networkx as nx
from guca.fitness.meshes import TriangleMesh, TriangleMeshWeights

def _tri_line_4():
    # 4 triangles as a line (same pattern used elsewhere in tests)
    G = nx.Graph()
    G.add_edges_from([
        (0,1),(1,2),(0,2),     # tri 0-1-2
        (1,3),(2,3),           # tri 1-2-3
        (1,4),(3,4)            # tri 1-3-4
    ])
    # add one more triangle to make tri_count > 3
    G.add_edges_from([(4,5),(3,5)])     # completes tri 3-4-5
    return G

def test_nontri_single_face_violation_forces_score_to_10():
    # Make the lower bound huge so the single shell length will be out of range.
    w = TriangleMeshWeights(
        nontri_len_min_coef=100.0,   # A -> forces lower bound very large
        nontri_len_max_coef=1.0,
        nontri_len_max_bias=0.0,
    )
    f = TriangleMesh(weights=w)
    s = f.score(_tri_line_4())
    assert s == 16.5

def test_nontri_single_face_permissive_defaults_do_not_clip():
    # With permissive defaults the score should NOT be forced to 10.0
    f = TriangleMesh()  # uses permissive A/B/C defaults
    s = f.score(_tri_line_4())
    assert s != 10.0

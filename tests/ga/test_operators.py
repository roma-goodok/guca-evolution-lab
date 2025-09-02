# tests/ga/test_operators.py
import random
from guca.ga.encoding import random_gene, decode_gene
from guca.ga.operators import splice_cx, make_mutate_fn

def test_splice_crossover_changes_length():
    rng = random.Random(1)
    a = [random_gene(state_count=3, rng=rng) for _ in range(6)]
    b = [random_gene(state_count=3, rng=rng) for _ in range(4)]
    c1, c2 = splice_cx(a, b, rng=rng, unaligned=True)
    assert len(c1) != len(a) or len(c2) != len(b)  # often changes length

def test_mutate_fn_keeps_domain_and_bounds():
    rng = random.Random(2)
    mutate = make_mutate_fn(
        state_count=3, min_len=2, max_len=10,
        structural_cfg={"insert_pb": 0.8, "delete_pb": 0.2, "duplicate_pb": 0.2},
        field_cfg={"enum_delta_pb": 0.9, "bitflip_pb": 0.9, "byte_pb": 0.9, "rotate_pb": 0.9},
        rng=rng,
    )
    ind = [random_gene(state_count=3, rng=rng) for _ in range(3)]
    (ind,) = mutate(ind)
    assert 2 <= len(ind) <= 10
    # decode should succeed
    for g in ind:
        _ = decode_gene(g, state_count=3)

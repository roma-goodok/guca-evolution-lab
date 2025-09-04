# tests/ga/test_operators.py
import random
import math

from guca.ga.operators import make_mutate_fn, splice_cx
from guca.ga.encoding import random_gene

U64_MAX = (1 << 64) - 1


def _valid_gene(g: int) -> bool:
    return isinstance(g, int) and 0 <= g <= U64_MAX


def test_mutate_fn_keeps_domain_and_bounds():
    rng = random.Random(2)

    mutate = make_mutate_fn(
        state_count=3,
        min_len=2,
        max_len=10,
        # generic structural knobs (legacy-compatible)
        structural_cfg={"insert_pb": 0.8, "delete_pb": 0.2, "duplicate_pb": 0.2},
        # low-level field mutation (aggressive to force changes)
        field_cfg={"enum_delta_pb": 0.9, "bitflip_pb": 0.9, "byte_pb": 0.9, "rotate_pb": 0.9, "allbytes_pb": 0.9},
        # C#-inspired regimes
        active_cfg={"factor": 0.10, "kind": "byte", "shift_pb": 0.02},
        passive_cfg={"factor": 0.50, "kind": "all_bytes", "shift_pb": 0.10},
        structuralx_cfg={"insert_active_pb": 0.20, "delete_inactive_pb": 0.10, "duplicate_head_pb": 0.20},
        rng=rng,
    )

    # start from a valid individual of length 6
    ind = [random_gene(state_count=3, rng=rng) for _ in range(6)]
    assert 2 <= len(ind) <= 10
    for g in ind:
        assert _valid_gene(g)

    # apply mutation several times; ensure bounds + at least one change
    orig = ind[:]
    changed = False
    for _ in range(30):
        ind = mutate(ind)
        assert 2 <= len(ind) <= 10
        for g in ind:
            assert _valid_gene(g)
        if ind != orig:
            changed = True
    assert changed, "mutation should eventually modify the individual"


def test_splice_cx_preserves_bounds_and_varlen():
    rng = random.Random(3)
    # make two parents with different lengths in the allowed window
    p1 = [random_gene(state_count=3, rng=rng) for _ in range(5)]
    p2 = [random_gene(state_count=3, rng=rng) for _ in range(9)]

    # splice with unaligned crossover (as used in toolbox)
    c1, c2 = splice_cx(p1, p2, rng=rng, unaligned=True)

    # variable-length is expected; but both must remain non-empty and genes valid
    assert isinstance(c1, list) and isinstance(c2, list)
    assert len(c1) >= 1 and len(c2) >= 1
    for g in c1 + c2:
        assert _valid_gene(g)

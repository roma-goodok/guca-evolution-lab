import random
from deap import base, creator

from guca.ga.encoding import (
    random_gene,
    sanitize_gene,
    decode_gene,
    OpKind,
)
from guca.ga.operators import make_mutate_fn


def _ensure_creator():
    """Create DEAP Individual class once for these tests."""
    try:
        getattr(creator, "Individual")
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)


def _semantically_valid_gene(g: int, state_count: int = 3) -> bool:
    """
    A gene is 'valid' if its *decoded* rule is in-domain and the reserved-low byte is 0x00.

    We do NOT enforce a specific raw-byte encoding for 'unset' fields (0 vs 0xFF/0xF),
    because runtime ignores fields that are irrelevant to the op. The structural
    invariant we keep here is only the low reserved byte == 0x00.
    """
    # Structural invariant: low reserved byte must be zero.
    if (g & 0xFF) != 0:
        return False

    # Decode into semantic rule and check domains.
    r = decode_gene(g, state_count=state_count)

    # cond must be a valid state id
    if not (0 <= r.cond_current < state_count):
        return False

    # Only TurnToState uses operand meaningfully; others may ignore it
    if r.op_kind == OpKind.TurnToState:
        if r.operand is None or not (0 <= r.operand < state_count):
            return False

    return True


def test_activity_mask_guides_mutation():
    _ensure_creator()
    rng = random.Random(0)

    mutate = make_mutate_fn(
        state_count=3,
        min_len=2,
        max_len=16,
        # no structural ops; we only test field-level mutation guided by mask
        structural_cfg={"insert_pb": 0.0, "delete_pb": 0.0, "duplicate_pb": 0.0},
        # zero generic field probs; we want only the active-regime to act
        field_cfg={
            "bitflip_pb": 0.0,
            "byte_pb": 0.0,
            "allbytes_pb": 0.0,
            "rotate_pb": 0.0,
            "enum_delta_pb": 0.0,
        },
        # C#-inspired regimes: mutate actives, not passives
        active_cfg={"factor": 1.0, "kind": "byte", "shift_pb": 0.0},
        passive_cfg={"factor": 0.0, "kind": "byte", "shift_pb": 0.0},
        # disable the C#-extra structural side-effects here
        structuralx_cfg={
            "insert_active_pb": 0.0,
            "delete_inactive_pb": 0.0,
            "duplicate_head_pb": 0.0,
        },
        rng=rng,
    )

    # build a 6-gene DEAP individual (not a plain list)
    genes = [random_gene(state_count=3, rng=rng) for _ in range(6)]
    # Make them canonical enough for this test's validator
    genes = [
        sanitize_gene(
            g,
            state_count=3,
            enforce_semantics=True,
            canonicalize_flags=True,
            enforce_bounds_order=True,
        )
        for g in genes
    ]
    ind = creator.Individual(genes)

    for g in ind:
        assert _semantically_valid_gene(g)

    # pretend: first 3 were active, last 3 were not
    ind.active_mask = [True, True, True, False, False, False]

    orig = list(ind)
    ind2 = mutate(ind)

    # Safety: still valid and same length
    assert len(ind2) == 6
    for g in ind2:
        assert _semantically_valid_gene(g)

    # Expect: at least one of the first 3 positions changed…
    changed_active = any(ind2[i] != orig[i] for i in range(3))
    # …and none of the last 3 changed (because passive factor = 0.0)
    unchanged_passive = all(ind2[i] == orig[i] for i in range(3, 6))

    assert changed_active, "Active genes did not receive preferential mutation"
    assert unchanged_passive, "Passive genes should not have changed with passive factor=0.0"

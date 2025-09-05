# tests/ga/test_activity_mask_influence.py
import random
from deap import creator
from guca.ga.encoding import random_gene
from guca.ga.operators import make_mutate_fn, _valid_gene

def _ensure_creator():
    try:
        creator.Individual  # type: ignore[attr-defined]
    except Exception:
        from deap import base
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

def test_activity_mask_guides_mutation():
    _ensure_creator()
    rng = random.Random(0)

    mutate = make_mutate_fn(
        state_count=3, min_len=2, max_len=16,
        # no structural ops; we only test field-level mutation guided by mask
        structural_cfg={"insert_pb": 0.0, "delete_pb": 0.0, "duplicate_pb": 0.0},
        # zero generic field probs; we want only the active-regime to act
        field_cfg={"bitflip_pb": 0.0, "byte_pb": 0.0, "allbytes_pb": 0.0, "rotate_pb": 0.0, "enum_delta_pb": 0.0},
        # C#-inspired regimes: mutate actives, not passives
        active_cfg={"factor": 1.0, "kind": "byte", "shift_pb": 0.0},
        passive_cfg={"factor": 0.0, "kind": "byte", "shift_pb": 0.0},
        # disable the C#-extra structural side-effects here
        structuralx_cfg={"insert_active_pb": 0.0, "delete_inactive_pb": 0.0, "duplicate_head_pb": 0.0},
        rng=rng,
    )

    # build a 6-gene DEAP individual (not a plain list)
    genes = [random_gene(state_count=3, rng=rng) for _ in range(6)]
    ind = creator.Individual(genes)
    for g in ind:
        assert _valid_gene(g)

    # pretend: first 3 were active, last 3 were not
    ind.active_mask = [True, True, True, False, False, False]

    orig = list(ind)
    ind2 = mutate(ind)

    # Safety: still valid and same length
    assert len(ind2) == 6
    for g in ind2:
        assert _valid_gene(g)

    # Expect: at least one of the first 3 positions changed…
    changed_active = any(ind2[i] != orig[i] for i in range(3))
    # …and none of the last 3 changed (because passive factor = 0.0)
    unchanged_passive = all(ind2[i] == orig[i] for i in range(3, 6))

    assert changed_active, "Active genes did not receive preferential mutation"
    assert unchanged_passive, "Passive genes should not have changed with passive factor=0.0"

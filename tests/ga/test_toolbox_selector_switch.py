# tests/ga/test_toolbox_selector_switch.py
import random
from deap import base, creator
from guca.ga.toolbox import _make_selector

def _mk_pop():
    try:
        creator.FitnessMax
    except Exception:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.Individual
    except Exception:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    # 5 inds with different fitness
    pop = []
    for f in [1,2,3,4,5]:
        ind = creator.Individual([0x0])
        ind.fitness.values = (float(f),)
        pop.append(ind)
    return pop

def test_make_selector_rank_and_tournament_return_k():
    rng = random.Random(42)
    pop = _mk_pop()

    rank_sel = _make_selector("rank", tournament_k=3, random_ratio=0.0, rng=rng)
    out = rank_sel(pop, k=7)
    assert len(out) == 7

    tourn_sel = _make_selector("tournament", tournament_k=3, random_ratio=0.0, rng=rng)
    out2 = tourn_sel(pop, k=7)
    assert len(out2) == 7

# tests/ga/test_selection_rank.py
import random
from deap import base, creator
from guca.ga.toolbox import _sel_rank

def _mk_pop(fits):
    try:
        creator.FitnessMax
    except Exception:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.Individual
    except Exception:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    pop = []
    for f in fits:
        ind = creator.Individual([0x0])
        ind.fitness.values = (float(f),)
        pop.append(ind)
    return pop

def test_rank_selection_biases_top():
    rng = random.Random(123)
    # fitness 10..1 (descending)
    pop = _mk_pop([10,9,8,7,6,5,4,3,2,1])
    sel = _sel_rank(pop, k=1000, rng=rng, random_ratio=0.0)
    # count indices
    counts = {i: 0 for i in range(len(pop))}
    for ind in sel:
        i = pop.index(ind)
        counts[i] += 1
    # top-3 should be collectively > bottom-3
    top3 = counts[0] + counts[1] + counts[2]
    bot3 = counts[7] + counts[8] + counts[9]
    assert top3 > bot3

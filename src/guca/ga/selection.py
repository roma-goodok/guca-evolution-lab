# src/guca/ga/selection.py
from __future__ import annotations
import random
from deap import tools

def _sel_rank(pop, k: int, rng: random.Random, random_ratio: float = 0.0):
    """Linear rank selection with optional random tail (with replacement)."""
    if k <= 0:
        return []
    sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
    n = len(sorted_pop)
    if n == 0:
        return []
    ranks = list(range(n, 0, -1))
    total = sum(ranks)
    probs = [r / total for r in ranks]
    k_rank = int(round(k * max(0.0, 1.0 - random_ratio)))
    k_rand = max(0, k - k_rank)
    selected = []
    if k_rank > 0:
        selected.extend(rng.choices(sorted_pop, weights=probs, k=k_rank))
    if k_rand > 0:
        selected.extend(rng.choices(pop, k=k_rand))
    return selected

def _make_selector(method: str, tournament_k: int, random_ratio: float, rng: random.Random):
    """Return sel(pop,k) based on GA config."""
    m = (method or "rank").lower()
    if m == "rank":
        return lambda pop, k: _sel_rank(pop, k, rng, random_ratio=random_ratio)
    if m == "tournament":
        return lambda pop, k: tools.selTournament(pop, k=k, tournsize=max(2, int(tournament_k)))
    if m == "roulette":
        def _roulette(pop, k):
            k_rank = int(round(k * max(0.0, 1.0 - random_ratio)))
            k_rand = max(0, k - k_rank)
            part = tools.selRoulette(pop, k=k_rank) if k_rank > 0 else []
            if k_rand > 0:
                part.extend(rng.choices(pop, k=k_rand))
            return part
        return _roulette
    # fallback
    return lambda pop, k: _sel_rank(pop, k, rng, random_ratio=random_ratio)

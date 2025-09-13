# src/guca/ga/toolbox.py
from __future__ import annotations

# Facade re-exports to keep API stable for tests and callers
from .loop import evolve
from .simulate import simulate_genome
from .selection import _sel_rank, _make_selector
from .checkpoint import _activity_scheme, _graph_summary

__all__ = [
    "evolve",
    "simulate_genome",
    "_sel_rank",
    "_make_selector",
    "_activity_scheme",
    "_graph_summary",
]

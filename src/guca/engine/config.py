# src/guca/engine/config.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class NearestSearchCfg:
    max_depth: int = 2
    tie_breaker: str = "stable"   # stable|random|by_id|by_creation
    connect_all: bool = False

@dataclass
class MachineConfig:
    max_steps: int = 120
    max_vertices: int = 2000
    start_state: str = "A"
    rng_seed: int = 42
    transcription: str = "continuable"
    nearest_search: NearestSearchCfg = field(default_factory=NearestSearchCfg)

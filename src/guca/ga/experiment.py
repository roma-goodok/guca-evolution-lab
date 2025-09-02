# src/guca/ga/experiment.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field as dc_field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from guca.engine.config import MachineConfig

# OmegaConf is optional at import time (keeps module import clean in any env)
try:
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:
    DictConfig = None  # type: ignore
    OmegaConf = None   # type: ignore


# ---- Sub-configs as dataclasses (safe defaults) -------------------------------

@dataclass
class StructuralCfg:
    insert_pb: float = 0.20
    delete_pb: float = 0.15
    duplicate_pb: float = 0.10

@dataclass
class FieldCfg:
    bitflip_pb: float = 0.10
    byte_pb: float = 0.05
    allbytes_pb: float = 0.02
    rotate_pb: float = 0.05
    enum_delta_pb: float = 0.15

@dataclass
class CheckpointCfg:
    save_best: bool = True
    save_last: bool = True
    save_every: int = 5
    save_population: str = "best"  # none | best | all
    fmt: str = "json"              # json | yaml (best artifact only)
    out_dir: str = "checkpoints"
    export_full_condition_shape: bool = False
    save_best_png: bool = False


# ---- Helper: normalize nested value -> dataclass instance ---------------------

def _normalize_dc(cls, value):
    """
    Convert 'value' to a dataclass instance of type 'cls'. Handles:
    - already a dataclass instance
    - OmegaConf DictConfig (via OmegaConf.to_container)
    - plain dict
    - old-style object with attributes
    """
    if is_dataclass(value):
        return value

    # OmegaConf DictConfig -> dict
    if DictConfig is not None and isinstance(value, DictConfig):  # type: ignore
        try:
            container = OmegaConf.to_container(value, resolve=True)  # type: ignore
            if isinstance(container, dict):
                return cls(**container)
        except Exception:
            pass

    # plain dict
    if isinstance(value, dict):
        return cls(**value)

    # old-style object with attributes
    if hasattr(value, "__dict__"):
        data = {}
        # only take fields known to the target dataclass
        for k in getattr(cls, "__dataclass_fields__", {}):
            if hasattr(value, k):
                data[k] = getattr(value, k)
        return cls(**data)

    # fallback to defaults
    return cls()


# ---- Main Hydra-instantiated GA config ---------------------------------------

@dataclass
class GAExperiment:
    # GA parameters
    pop_size: int = 40
    generations: int = 20
    cx_pb: float = 0.70
    mut_pb: float = 0.30
    tournament_k: int = 3
    elitism: int = 2

    # Genome bounds
    init_len: int = 6
    min_len: int = 1
    max_len: int = 64

    # Nested groups (dataclasses with default_factory)
    structural: StructuralCfg = dc_field(default_factory=StructuralCfg)
    field: FieldCfg = dc_field(default_factory=FieldCfg)
    checkpoint: CheckpointCfg = dc_field(default_factory=CheckpointCfg)

    # UI / ergonomics
    progress: bool = True

    def __post_init__(self):
        # Normalize any form (dataclass, DictConfig, dict, legacy object) -> dataclass instance
        self.structural = _normalize_dc(StructuralCfg, self.structural)
        self.field      = _normalize_dc(FieldCfg,      self.field)
        self.checkpoint = _normalize_dc(CheckpointCfg, self.checkpoint)

    def run(
        self,
        *,
        fitness: Any,
        machine_cfg: MachineConfig,
        states: List[str],
        seed: int,
        n_workers: int,
        run_dir: Path | None = None,
    ) -> Dict[str, Any]:
        try:
            from guca.ga.toolbox import evolve
        except Exception:
            return {
                "status": "toolbox_missing",
                "note": "Add guca.ga.toolbox.evolve to run the GA loop."
            }

        run_dir = run_dir or Path.cwd()

        summary = evolve(
            fitness=fitness,
            machine_cfg={
                "max_steps": machine_cfg.max_steps,
                "max_vertices": machine_cfg.max_vertices,
                "start_state": machine_cfg.start_state,
                "rng_seed": machine_cfg.rng_seed,
                "nearest_search": {
                    "max_depth": machine_cfg.nearest_search.max_depth,
                    "tie_breaker": machine_cfg.nearest_search.tie_breaker,
                    "connect_all": machine_cfg.nearest_search.connect_all,
                },
            },
            ga_cfg={
                "pop_size": self.pop_size,
                "generations": self.generations,
                "cx_pb": self.cx_pb,
                "mut_pb": self.mut_pb,
                "tournament_k": self.tournament_k,
                "elitism": self.elitism,
                "init_len": self.init_len,
                "min_len": self.min_len,
                "max_len": self.max_len,
                "structural": asdict(self.structural),
                "field": asdict(self.field),
            },
            states=states,
            seed=seed,
            n_workers=n_workers,
            checkpoint_cfg=asdict(self.checkpoint),  # <- fmt flows through (yaml/json)
            run_dir=run_dir,
            progress=self.progress,
        )
        return summary

# src/guca/ga/experiment.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict


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
class EncodingCfg:
    """Decode/sanitize policy for genes before simulation."""
    sanitize_on_decode: bool = False
    enforce_semantics: bool = False
    canonicalize_flags: bool = False
    enforce_bounds_order: bool = False

@dataclass
class SelectionCfg:
    method: str = "rank"        # rank | tournament | roulette | tournament | elite (selector is chosen in toolbox)
    random_ratio: float = 0.0   # portion of picks done uniformly at random

@dataclass
class ActiveCfg:
    factor: float = 0.10          # P(mutate) per active gene
    kind: str = "byte"            # bit|byte|allbytes|rotate
    rotate_extra_pb: float = 0.20 # extra rotate chance *after* main mutate

@dataclass
class PassiveCfg:
    factor: float = 0.50
    kind: str = "allbytes"
    rotate_extra_pb: float = 0.20

@dataclass
class StructuralXCfg:
    insert_active_pb: float = 0.10         # insert(copy_of_random_active_gene)
    remove_inactive_pb: float = 0.10       # remove(random_inactive_gene) if len>=threshold
    remove_inactive_min_len: int = 100
    duplicate_head_pb: float = 0.20        # duplicate gene[0] to front


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
    hist_bins: int = 100



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
    # Selection method & exploration
    selection: SelectionCfg = field(default_factory=SelectionCfg)

    # Active/passive mutation regimes (C#-style)
    active: ActiveCfg = field(default_factory=ActiveCfg)
    passive: PassiveCfg = field(default_factory=PassiveCfg)
    structuralx: StructuralXCfg = field(default_factory=StructuralXCfg)

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
    encoding: EncodingCfg = dc_field(default_factory=EncodingCfg)

    # UI / ergonomics
    progress: bool = True

    def __post_init__(self):
        # Normalize any form (dataclass, DictConfig, dict, legacy object) -> dataclass instance
        self.structural = _normalize_dc(StructuralCfg, self.structural)
        self.field      = _normalize_dc(FieldCfg,      self.field)
        self.checkpoint = _normalize_dc(CheckpointCfg, self.checkpoint)
        self.active     = _normalize_dc(ActiveCfg,     self.active)
        self.passive    = _normalize_dc(PassiveCfg,    self.passive)
        self.structuralx= _normalize_dc(StructuralXCfg,self.structuralx)
        # Accept selection as dict/DictConfig and coerce into SelectionCfg
        if isinstance(self.selection, dict):
            self.selection = SelectionCfg(**self.selection)
        if DictConfig is not None and isinstance(self.selection, DictConfig):  # type: ignore
            self.selection = SelectionCfg(**OmegaConf.to_container(self.selection, resolve=True))  # type: ignore
        # Normalize encoding from dict/DictConfig to dataclass
        self.encoding = _normalize_dc(EncodingCfg, self.encoding)

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
                "active": asdict(self.active),
                "passive": asdict(self.passive),
                "structuralx": asdict(self.structuralx),
                "selection": {
                    "method": self.selection.method,
                    "random_ratio": self.selection.random_ratio,
                    },
                "encoding": asdict(self.encoding),
            },
            states=states,
            seed=seed,
            n_workers=n_workers,
            checkpoint_cfg=asdict(self.checkpoint),  # <- fmt flows through (yaml/json)
            run_dir=run_dir,
            progress=self.progress,
        )
        return summary

# src/guca/ga/experiment.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from guca.engine.config import MachineConfig

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

    # Structural mutation
    class Structural:
        insert_pb: float = 0.20
        delete_pb: float = 0.15
        duplicate_pb: float = 0.10
    structural: Structural = Structural()

    # Field mutation
    class Field:
        bitflip_pb: float = 0.10
        byte_pb: float = 0.05
        allbytes_pb: float = 0.02
        rotate_pb: float = 0.05
        enum_delta_pb: float = 0.15
    field: Field = Field()

    # Checkpointing
    class Checkpoint:
        save_best: bool = True
        save_last: bool = True
        save_every: int = 5
        save_population: str = "best"  # none|best|all
        fmt: str = "json"              # json|yaml
        out_dir: str = "checkpoints"
    checkpoint: Checkpoint = Checkpoint()

    # Hydra will pass states / seed / n_workers via run() args

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
        """
        Launch GA using guca.ga.toolbox.evolve (added in subsequent step).
        For now, return a minimal summary if toolbox isn't available.
        """
        try:
            from guca.ga.toolbox import evolve
        except Exception:
            return {
                "status": "toolbox_missing",
                "pop_size": self.pop_size,
                "generations": self.generations,
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
                "structural": vars(self.structural),
                "field": vars(self.field),
            },
            states=states,
            seed=seed,
            n_workers=n_workers,
            checkpoint_cfg=vars(self.checkpoint),
            run_dir=run_dir,
        )
        return summary

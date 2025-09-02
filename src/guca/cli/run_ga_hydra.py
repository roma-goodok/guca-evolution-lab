# src/guca/cli/run_ga_hydra.py
from __future__ import annotations
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig  # <<< NEW
from omegaconf import DictConfig, OmegaConf

GREEN = "\033[1;32m"
RESET = "\033[0m"

@hydra.main(config_path='../../../configs', config_name='ga', version_base=None)
def main(cfg: DictConfig) -> None:
    fitness = instantiate(cfg.fitness)
    machine_cfg = instantiate(cfg.machine)
    ga_runner = instantiate(cfg.ga)

    # Resolve Hydra-managed output dir (runs/.../_hydra etc.)
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    # --- Friendly header ---
    exp_name = str(cfg.experiment.name)
    print(f"{GREEN}Experiment: {exp_name}{RESET}")
    print(f"Logbook root: {cfg.logbook_dir}")
    print(f"Run dir     : {run_dir}\n")

    summary = ga_runner.run(
        fitness=fitness,
        machine_cfg=machine_cfg,
        states=list(cfg.states),
        seed=int(cfg.seed),
        n_workers=int(cfg.n_workers),
        run_dir=run_dir,  # <<< ensure checkpoints go under the Hydra dir
    )

    # (Optional) echo resolved config at the end
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("GA summary:", summary)

if __name__ == "__main__":
    main()

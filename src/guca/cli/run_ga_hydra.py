# src/guca/cli/run_ga_hydra.py
from __future__ import annotations
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='../../../configs', config_name='ga', version_base=None)
def main(cfg: DictConfig) -> None:
    # Instantiate directly from Hydra config
    fitness = instantiate(cfg.fitness)
    machine_cfg = instantiate(cfg.machine)
    ga_runner = instantiate(cfg.ga)

    # States / seed / workers from cfg; run dir = Hydra's cwd
    summary = ga_runner.run(
        fitness=fitness,
        machine_cfg=machine_cfg,
        states=list(cfg.states),
        seed=int(cfg.seed),
        n_workers=int(cfg.n_workers),
        run_dir=Path.cwd(),
    )
    print(OmegaConf.to_yaml(cfg, resolve=True))  # echo resolved config
    print("GA summary:", summary)

if __name__ == "__main__":
    main()

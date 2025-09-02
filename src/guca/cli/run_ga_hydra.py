# src/guca/cli/run_ga_hydra.py
from __future__ import annotations
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path='../../../configs', config_name='ga', version_base=None)
def main(cfg: DictConfig) -> None:
    fitness = instantiate(cfg.fitness)
    machine_cfg = instantiate(cfg.machine)
    ga_runner = instantiate(cfg.ga)

    # >>> get Hydra-managed output dir (robust to any cwd issues)
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    summary = ga_runner.run(
        fitness=fitness,
        machine_cfg=machine_cfg,
        states=list(cfg.states),
        seed=int(cfg.seed),
        n_workers=int(cfg.n_workers),
        run_dir=run_dir,               # <<< use hydra output dir
    )
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    print("GA summary:", summary)


if __name__ == "__main__":
    main()

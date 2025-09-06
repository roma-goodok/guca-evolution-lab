#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src/ layout is importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guca.ga.toolbox import evolve
from guca.fitness.meshes import TriangleMesh, MeshWeights


def main():
    # --- fitness and machine config (keep close to test defaults) ---
    states = ["A", "B", "C"]

    fitness = TriangleMesh(weights=MeshWeights())
    machine_cfg = {
        "max_steps": 60,
        "max_vertices": 2000,
        "start_state": "A",
        "nearest_search": {
            "max_depth": 2,
            "tie_breaker": "stable",
            "connect_all": False,
        },
    }

    # --- GA config (short smoke run) ---
    ga_cfg = {
        "pop_size": 24,
        "generations": 8,
        "cx_pb": 0.7,
        "mut_pb": 0.3,
        "tournament_k": 2,
        "elitism": 1,
        "init_len": 6,
        "min_len": 1,
        "max_len": 16,
        "structural": {"insert_pb": 0.2, "delete_pb": 0.1, "duplicate_pb": 0.1},
        "field": {"enum_delta_pb": 0.2},
        # optional: selection policy + random immigrants (exercise both code paths)
        "selection": {"method": "rank", "random_ratio": 0.05},
    }

    # --- checkpointing ---
    # export_full_condition_shape=True => places full condition placeholders in YAML
    ckpt_cfg = {
        "save_best": True,
        "save_last": True,
        "save_every": 0,                 # only best/last for this smoke
        "save_population": "best",
        "fmt": "yaml",
        "out_dir": "checkpoints/w5_smoke",   # will be created under run_dir
        "export_full_condition_shape": True,
        "save_best_png": True,               # requires matplotlib
    }

    # --- where to write the run artifacts ---
    run_dir = ROOT / "runs" / "w5_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = evolve(
        fitness=fitness,
        machine_cfg=machine_cfg,
        ga_cfg=ga_cfg,
        states=states,
        seed=123,
        n_workers=0,
        checkpoint_cfg=ckpt_cfg,
        run_dir=run_dir,        # <- REQUIRED keyword-only argument
        progress=True,          # tqdm if installed; falls back to prints
    )

    ckpts = summary.get("checkpoints", {})
    print("best.yaml:", ckpts.get("best"))
    print("best.png :", ckpts.get("best_png"))
    print("done. Summary:", {k: v for k, v in summary.items() if k != "checkpoints"})


if __name__ == "__main__":
    main()

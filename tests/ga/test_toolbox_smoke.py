# tests/ga/test_toolbox_smoke.py
from pathlib import Path
from guca.ga.toolbox import evolve
from guca.fitness.meshes import QuadMesh
from guca.fitness.meshes import MeshWeights

def test_evolve_runs_and_writes_checkpoints(tmp_path: Path):
    fitness = QuadMesh(weights=MeshWeights())
    machine_cfg = {"max_steps": 60, "max_vertices": 2000, "start_state": "A",
                   "nearest_search": {"max_depth": 2, "tie_breaker": "stable", "connect_all": False}}
    ga_cfg = {"pop_size": 10, "generations": 3, "cx_pb": 0.7, "mut_pb": 0.3,
              "tournament_k": 2, "elitism": 1, "init_len": 4, "min_len": 1, "max_len": 16,
              "structural": {"insert_pb": 0.2, "delete_pb": 0.1, "duplicate_pb": 0.1},
              "field": {"enum_delta_pb": 0.2}}
    ckpt_cfg = {"save_best": True, "save_last": True, "save_every": 0, "save_population": "best",
                "fmt": "json", "out_dir": "checkpoints"}
    res = evolve(fitness=fitness, machine_cfg=machine_cfg, ga_cfg=ga_cfg,
                 states=["A", "B", "C"], seed=123, n_workers=0,
                 checkpoint_cfg=ckpt_cfg, run_dir=tmp_path)
    assert res["status"] == "ok"
    assert "checkpoints" in res and "best" in res["checkpoints"]

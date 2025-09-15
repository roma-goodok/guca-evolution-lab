import random
from pathlib import Path
from guca.ga.toolbox import evolve
from guca.fitness.meshes import TriangleMesh, TriangleMeshWeights

def test_cx_handles_len1_parents(tmp_path: Path):
    # Fitness and machine are trivial; we just want to exercise GA crossover.
    fitness = TriangleMesh(weights=TriangleMeshWeights())
    machine_cfg = {
        "max_steps": 3,
        "max_vertices": 50,
        "start_state": "A",
        "nearest_search": {"max_depth": 2, "tie_breaker": "stable", "connect_all": False},
    }
    # Force many length-1 individuals and always crossover.
    ga_cfg = {
        "pop_size": 6,
        "generations": 2,
        "cx_pb": 1.0,           # always mate
        "mut_pb": 0.0,          # keep lengths stable for this test
        "tournament_k": 2,
        "elitism": 1,
        "init_len": 1,          # <-- length-1 parents are expected
        "min_len": 1,
        "max_len": 4,
        "structural": {"insert_pb": 0.0, "delete_pb": 0.0, "duplicate_pb": 0.0},
        "field": {"enum_delta_pb": 0.0},
        "selection": {"method": "rank", "random_ratio": 0.0},
    }
    ckpt_cfg = {"save_best": False, "save_last": False, "save_every": 0, "save_population": "none", "fmt": "json"}
    res = evolve(
        fitness=fitness, machine_cfg=machine_cfg, ga_cfg=ga_cfg,
        states=["A", "B"], seed=123, n_workers=0,
        checkpoint_cfg=ckpt_cfg, run_dir=tmp_path, progress=False
    )
    assert res["status"] == "ok"

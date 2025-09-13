from pathlib import Path
import csv
from guca.ga.toolbox import evolve
from guca.fitness.meshes import TriangleMesh, MeshWeights

def test_progress_csv_written(tmp_path: Path):
    fitness = TriangleMesh(weights=MeshWeights())
    machine_cfg = {
        "max_steps": 3, "max_vertices": 200, "start_state": "A",
        "nearest_search": {"max_depth": 2, "tie_breaker": "stable", "connect_all": False},
    }
    ga_cfg = {
        "pop_size": 8, "generations": 2, "cx_pb": 0.7, "mut_pb": 0.3,
        "tournament_k": 2, "elitism": 1, "init_len": 4, "min_len": 1, "max_len": 16,
        "structural": {"insert_pb": 0.2, "delete_pb": 0.1, "duplicate_pb": 0.1},
        "field": {"enum_delta_pb": 0.2},
        "selection": {"method": "rank", "random_ratio": 0.0},
    }
    ckpt_cfg = {
        "save_best": True, "save_last": True, "save_every": 0,
        "save_population": "best", "fmt": "yaml", "out_dir": "checkpoints",
        "export_full_condition_shape": True,
    }
    res = evolve(
        fitness=fitness, machine_cfg=machine_cfg, ga_cfg=ga_cfg,
        states=["A","B","C"], seed=7, n_workers=0,
        checkpoint_cfg=ckpt_cfg, run_dir=tmp_path, progress=False
    )
    ckroot = tmp_path / "checkpoints"
    progress = ckroot / "progress.csv"
    assert progress.exists(), "progress.csv not created"
    rows = list(csv.DictReader(progress.open("r", encoding="utf-8")))
    assert len(rows) >= 2, "should have at least 2 rows (gen 0 and gen 1/2)"
    hdr = rows[0].keys()
    assert "datetime" in hdr and "best_activity_scheme" in hdr

# tests/ga/test_checkpoint_bundle.py
from pathlib import Path
from guca.ga.toolbox import evolve
from guca.fitness.meshes import TriangleMesh, MeshWeights

def test_checkpoint_bundle_contains_artifacts(tmp_path: Path):
    fitness = TriangleMesh(weights=MeshWeights())
    machine_cfg = {
        "max_steps": 5,
        "max_vertices": 200,
        "start_state": "A",
        "nearest_search": {"max_depth": 2, "tie_breaker": "stable", "connect_all": False},
    }
    ga_cfg = {
        "pop_size": 8,
        "generations": 3,
        "cx_pb": 0.7,
        "mut_pb": 0.3,
        "tournament_k": 2,
        "elitism": 1,
        "init_len": 4,
        "min_len": 1,
        "max_len": 16,
        "structural": {"insert_pb": 0.2, "delete_pb": 0.1, "duplicate_pb": 0.1},
        "field": {"enum_delta_pb": 0.2},
        "selection": {"method": "rank", "random_ratio": 0.0},
    }
    ckpt_cfg = {
        "save_best": True,
        "save_last": True,
        "save_every": 1,              # force periodic
        "save_population": "best",
        "fmt": "yaml",
        "out_dir": "checkpoints",
        "export_full_condition_shape": True,
        "save_best_png": True,
        "hist_bins": 16,
    }
    res = evolve(
        fitness=fitness, machine_cfg=machine_cfg, ga_cfg=ga_cfg,
        states=["A", "B", "C"], seed=321, n_workers=0,
        checkpoint_cfg=ckpt_cfg, run_dir=tmp_path, progress=False
    )
    assert res["status"] == "ok"
    ckroot = tmp_path / "checkpoints"
    assert ckroot.exists()
    # find at least one epoch_* or last_* folder
    found = list(ckroot.glob("epoch_*")) + list(ckroot.glob("last_*"))
    assert found, "No checkpoint folders created"
    # check artifacts in one folder
    d = found[0]
    assert (d / "population.json").exists()
    assert (d / "genome.yaml").exists()
    assert (d / "metrics.csv").exists()
    # hist + png are best-effort; still assert at least one histogram exists
    hists = [(d / "hist_fitness.png"), (d / "hist_length.png"), (d / "hist_active.png")]
    assert any(p.exists() for p in hists)

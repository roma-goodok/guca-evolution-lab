# tests/test_cli_run_gum_enriched_yaml.py
import sys, json
from pathlib import Path
import yaml
from guca.cli import run_gum as rg

def test_run_gum_enriched_yaml(tmp_path: Path):
    genome = Path("examples/single_run_nearest.yaml")
    run_dir = tmp_path / "runs"
    # emulate CLI argv (run_gum.parse_args() reads sys.argv)
    argv = [
        "prog",
        "--genome", str(genome),
        "--run-dir", str(run_dir),
        "--save-png",
        "--vis-node-render", "ids",
    ]
    old = sys.argv[:]
    sys.argv = argv
    try:
        rc = rg.main()
    finally:
        sys.argv = old
    # enriched file should exist
    enrich = run_dir / "single_run_nearest" / "genome_enriched.yaml"
    assert enrich.exists()
    data = yaml.safe_load(enrich.read_text(encoding="utf-8"))
    assert "meta" in data and "graph_summary" in data["meta"]
    assert isinstance(data["meta"]["graph_summary"].get("edge_list"), list)
    assert isinstance(data["meta"].get("activity_scheme"), str)

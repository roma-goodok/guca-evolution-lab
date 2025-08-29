import json
import sys
from pathlib import Path
import subprocess
import yaml
import pytest

EXAMPLES = [
    Path("examples/genomes/dumb_belly_genom.yaml"),
    Path("examples/genomes/fractal7_genom.yaml"),
]

def _run_cli(genome: Path):
    out = subprocess.check_output([sys.executable, "-m", "guca.cli.run_gum", "--genome", str(genome)])
    return json.loads(out)

@pytest.mark.fast
@pytest.mark.parametrize("genome", EXAMPLES)
def test_m1_golden_template(genome: Path):
    cfg = yaml.safe_load(genome.read_text(encoding="utf-8"))
    res = _run_cli(genome)
    assert isinstance(res, dict) and "nodes" in res and "edges" in res

    exp = cfg.get("expected") or {}
    nodes = exp.get("nodes")
    edges = exp.get("edges")
    state_counts = exp.get("states_count")

    if isinstance(nodes, int):
        assert res["nodes"] == nodes
    else:
        assert res["nodes"] >= 1

    if isinstance(edges, int):
        assert res["edges"] == edges
    else:
        assert res["edges"] >= 0

    if isinstance(state_counts, dict):
        for k, v in state_counts.items():
            assert res["states_count"].get(k, 0) == int(v)

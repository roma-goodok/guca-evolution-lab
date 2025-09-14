# tests/test_cli_draw_edge_list.py
from pathlib import Path
import yaml
from guca.cli import draw_edge_list as cli

def test_draw_edge_list_cli(tmp_path: Path):
    y = tmp_path / "g.yaml"
    y.write_text(yaml.safe_dump({"edge_list": [[0,1],[1,2],[2,0]]}, sort_keys=False), encoding="utf-8")
    outdir = tmp_path / "runs"
    rc = cli.main(["--yaml", str(y), "--run-dir", str(outdir), "--vis-node-render", "ids"])
    assert rc == 0
    png = outdir / "g" / "vis" / "edge_list.png"
    assert png.exists() and png.stat().st_size > 0

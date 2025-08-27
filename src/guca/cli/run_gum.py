from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict
import yaml

from guca.core.graph import GUMGraph, stats_summary
from guca.core.machine import GraphUnfoldingMachine
from guca.core.rules import change_table_from_yaml, TranscriptionWay, CountCompare

def _load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main() -> None:
    ap = argparse.ArgumentParser(description="Run a GUM genome and print stats (JSON).")
    ap.add_argument("--genome", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(args.genome)
    machine_cfg = cfg.get("machine", {})
    rules_yaml = cfg.get("rules", [])

    graph = GUMGraph()
    m = GraphUnfoldingMachine(
        graph,
        start_state=str(machine_cfg.get("start_state", "A")),
        transcription=TranscriptionWay(machine_cfg.get("transcription", "resettable")),
        count_compare=CountCompare(machine_cfg.get("count_compare", "range")),
        max_vertices=int(machine_cfg.get("max_vertices", 0)),
        max_steps=(args.steps if args.steps is not None else int(machine_cfg.get("max_steps", 100))),
    )
    m.change_table = change_table_from_yaml(rules_yaml)
    m.run()

    summary = stats_summary(graph)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.do_assert and "expected" in cfg:
        exp = cfg["expected"] or {}
        if "nodes" in exp:
            assert summary["nodes"] == int(exp["nodes"])
        if "edges" in exp:
            assert summary["edges"] == int(exp["edges"])
        if "states_count" in exp and isinstance(exp["states_count"], dict):
            for k, v in exp["states_count"].items():
                assert summary["states_count"].get(k, 0) == int(v)

if __name__ == "__main__":
    main()

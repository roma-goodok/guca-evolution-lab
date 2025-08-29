#!/usr/bin/env python
"""
Freeze (or preview) expected stats for genomes:
- Default: DRY RUN (no file changes) shows what would be written
- Use --write to actually update the YAML 'expected' section

Examples:
  python scripts/freeze_expected.py --steps 120
  python scripts/freeze_expected.py --glob "fractal*.yaml" --steps 200
  python scripts/freeze_expected.py --write --steps 120
"""

from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from typing import Dict, Any
import yaml

def run_cli(genome: Path, steps_override: int | None) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "guca.cli.run_gum", "--genome", str(genome)]
    if steps_override is not None:
        cmd += ["--steps", str(steps_override)]
    out = subprocess.check_output(cmd).decode("utf-8")
    return json.loads(out)

def main():
    ap = argparse.ArgumentParser(description="Freeze (or preview) expected stats in genome YAMLs.")
    ap.add_argument("--dir", default="examples/genomes", help="Directory with genome YAMLs")
    ap.add_argument("--glob", default="*.yaml", help="Glob pattern (default: *.yaml)")
    ap.add_argument("--steps", type=int, default=None, help="Override steps when running CLI")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="Write changes to files")
    g.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    args = ap.parse_args()

    # default behavior = dry-run
    dry_run = (not args.write) or args.dry_run

    root = Path(args.dir)
    files = sorted(root.glob(args.glob))
    if not files:
        print(f"[WARN] No genomes found in {root} matching {args.glob}")
        sys.exit(0)

    changed = 0
    for f in files:
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        stats = run_cli(f, steps_override=args.steps)
        proposed = {
            "steps": args.steps if args.steps is not None else cfg.get("expected", {}).get("steps", None),
            "nodes": int(stats["nodes"]),
            "edges": int(stats["edges"]),
            "states_count": {str(k): int(v) for k, v in (stats.get("states_count") or {}).items()},
        }

        # Build the new cfg (without mutating file yet)
        new_cfg = dict(cfg)
        exp = dict(new_cfg.get("expected") or {})
        if proposed["steps"] is not None:
            exp["steps"] = int(proposed["steps"])
        exp["nodes"] = proposed["nodes"]
        exp["edges"] = proposed["edges"]
        exp["states_count"] = proposed["states_count"]
        new_cfg["expected"] = exp

        if dry_run:
            print(f"--- DRY RUN: {f.name} ---")
            print("would set expected to:")
            print(yaml.safe_dump({"expected": new_cfg["expected"]}, sort_keys=False, allow_unicode=True).rstrip())
            print()
        else:
            if new_cfg != cfg:
                f.write_text(yaml.safe_dump(new_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
                print(f"[OK] updated expected for {f.name}: nodes={proposed['nodes']} edges={proposed['edges']}")
                changed += 1
            else:
                print(f"[SKIP] {f.name} already up to date")

    if dry_run:
        print("[INFO] DRY RUN complete. No files were modified.")
    else:
        print(f"[INFO] Done. Files changed: {changed}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any, Dict, List
import yaml

import re

VALID_KINDS = {
    "TurnToState",
    "TryToConnectWith",
    "TryToConnectWithNearest",
    "GiveBirth",
    "GiveBirthConnected",
    "Die",
    "DisconnectFrom",
}

ALIASES = {
    "DisconectFrom": "DisconnectFrom",  # legacy typo
}

def camel_to_snake(name: str) -> str:
    # Split CamelCase into words, join with underscores, lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()

def as_int(x, default=-1):
    try:
        return int(x)
    except Exception:
        return default

def map_prior(value: Any) -> str:
    # Legacy uses "Min" and "Ignored" to mean "don't check prior"; we map to "any".
    if value is None:
        return "any"
    v = str(value)
    if v in ("Min", "Ignored"):
        return "any"
    return v

def normalize_condition(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "current": str(c.get("currentState", "A")),
        "prior": map_prior(c.get("priorState", "any")),
        "conn_ge": as_int(c.get("allConnectionsCount_GE"), -1),
        "conn_le": as_int(c.get("allConnectionsCount_LE"), -1),
        "parents_ge": as_int(c.get("parentsCount_GE"), -1),
        "parents_le": as_int(c.get("parentsCount_LE"), -1),
    }

def normalize_operation(o: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(o.get("kind", ""))
    kind = ALIASES.get(kind, kind)
    operand = o.get("operandNodeState")
    if kind not in VALID_KINDS:
        # leave as-is but warn to console
        print(f"[WARN] Unknown operation kind: {kind}")
    return {"kind": kind, "operand": (str(operand) if operand is not None else None)}

def convert_gene_list_to_yaml_rules(gene_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules = []
    for g in gene_list:
        cond = g.get("condition", {})
        op = g.get("operation", {})
        rules.append({
            "condition": normalize_condition(cond),
            "op": normalize_operation(op),
        })
    return rules

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_m1_json_to_yaml.py data_demo_2010_dict_genes.json [--outdir examples/genomes]")
        sys.exit(1)

    src = Path(sys.argv[1])
    outdir = Path("examples/genomes")
    if "--outdir" in sys.argv:
        outdir = Path(sys.argv[sys.argv.index("--outdir")+1])

    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "genes" not in data or not isinstance(data["genes"], dict):
        print("[ERROR] Expected a dict at top-level with key 'genes'")
        sys.exit(2)

    genes_dict = data["genes"]
    print(f"[INFO] Found {len(genes_dict)} genomes: {list(genes_dict.keys())[:6]}{' ...' if len(genes_dict)>6 else ''}")

    outdir.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for name, gene_list in genes_dict.items():
        if not isinstance(gene_list, list):
            print(f"[SKIP] Key '{name}' is not a list of rules")
            continue
        rules = convert_gene_list_to_yaml_rules(gene_list)
        out = {
            "machine": {
                "start_state": "A",
                "max_steps": 120,
                "transcription": "resettable",
                "count_compare": "range",
                "max_vertices": 2000,
            },
            "rules": rules,
            "expected": {
                # filled by freeze script
            },
        }        
        fname = camel_to_snake(name)
        if not fname.endswith(".yaml"):
            fname += ".yaml"
        (outdir / fname).write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"[OK] Wrote {outdir / fname}")
        wrote += 1

    if wrote == 0:
        print("[WARN] No YAMLs written. Inspect the input structure.")
        sys.exit(3)

if __name__ == "__main__":
    main()

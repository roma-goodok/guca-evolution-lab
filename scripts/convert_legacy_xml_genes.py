#!/usr/bin/env python3
"""
convert_legacy_xml_genes.py
---------------------------------

From-scratch converter for legacy XAML/XML "ChangeTable" genomes into:

  1) Intermediate JSON (with "rules" you can inspect)
  2) YAML that matches the expected schema:
       machine: { ... }
       init_graph: { nodes: [...] }
       rules: [ {condition: {...}, op: {...}}, ... ]

It parses locally (no shell-outs) and tries to be tolerant about tag
namespaces and minor structural variations.

CLI:
  python convert_legacy_xml_genes.py INPUT.xml -o OUTPUT.yaml
  # also writes INPUT.json next to INPUT.xml unless -j/--json-out is provided
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re
import xml.etree.ElementTree as ET

# ---- yaml import (friendly error) -------------------------------------------
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# ---- utilities ---------------------------------------------------------------
_num_re = re.compile(r'^[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?$')

def to_scalar(s: Optional[str]) -> Any:
    """Convert string to int/float/bool/None when obvious; else return original string."""
    if s is None:
        return None
    st = s.strip()
    low = st.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    # int?
    if st.isdigit() or (st.startswith(('+', '-')) and st[1:].isdigit()):
        try:
            return int(st, 10)
        except Exception:
            pass
    # float?
    if _num_re.match(st):
        try:
            return float(st)
        except Exception:
            pass
    return st

def strip_ns(tag: str) -> str:
    """Remove {namespace} from an element/attribute tag."""
    return tag.split('}', 1)[1] if '}' in tag else tag

def text_of(elem: Optional[ET.Element]) -> Optional[str]:
    """Return trimmed text; None if elem missing or empty."""
    if elem is None or elem.text is None:
        return None
    s = elem.text.strip()
    return s if s else None

def lower_attrs(elem: ET.Element) -> Dict[str, Any]:
    """All attributes lower-cased, values normalized to scalars."""
    return {strip_ns(k).lower(): to_scalar(v) for k, v in elem.attrib.items()}

def find_child_by_suffix(elem: ET.Element, suffixes: List[str]) -> Optional[ET.Element]:
    """First direct child whose tag (without ns) ends with any suffix (case-insensitive)."""
    for ch in list(elem):
        t = strip_ns(ch.tag).lower()
        for suf in suffixes:
            if t.endswith(suf.lower()):
                return ch
    return None

def find_descendant(elem: ET.Element, name_variants: List[str]) -> Optional[ET.Element]:
    """First descendant whose tag matches (case-insensitive) one of name_variants."""
    names = {n.lower() for n in name_variants}
    for e in elem.iter():
        if strip_ns(e.tag).lower() in names:
            return e
    return None

# ---- data model --------------------------------------------------------------
@dataclass
class Rule:
    condition: Dict[str, Any] = field(default_factory=dict)
    op: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Genome:
    name: str
    rules: List[Rule] = field(default_factory=list)
    # Optional meta we can infer from legacy XML:
    capacity: Optional[int] = None
    start_state_hint: Optional[str] = None  # e.g., from most-common 'current'

# ---- parsing: ChangeTable -> rules ------------------------------------------
def parse_operation_condition(cond_box: Optional[ET.Element]) -> Dict[str, Any]:
    """
    Extract condition from a ChangeTableItem scope.
    Looks for <OperationCondition .../> anywhere under cond_box.
    """
    if cond_box is None:
        return {}

    oc = find_descendant(cond_box, ["OperationCondition"])
    if oc is None:  # IMPORTANT: don't use "or cond_box"
        oc = cond_box

    # Lowercase keys, scalarize values
    attrs = {strip_ns(k).lower(): to_scalar(v) for k, v in oc.attrib.items()}

    def pick(*keys: str, default=None):
        for k in keys:
            v = attrs.get(k)
            if v is not None:
                return v
        return default

    cond: Dict[str, Any] = {}

    cur = pick("currentstate", "current")
    prv = pick("priorstate", "prior")
    if cur is not None:
        cond["current"] = cur
    if prv is not None:
        cond["prior"] = prv

    cond_ge = pick("allconnectionscount_ge", "connections_ge", "allconn_ge", "conn_ge")
    cond_le = pick("allconnectionscount_le", "connections_le", "allconn_le", "conn_le")
    par_ge  = pick("parentscount_ge", "parents_ge")
    par_le  = pick("parentscount_le", "parents_le")

    if cond_ge is not None:
        cond["conn_ge"] = int(cond_ge)
    if cond_le is not None:
        cond["conn_le"] = int(cond_le)
    if par_ge is not None:
        cond["parents_ge"] = int(par_ge)
    if par_le is not None:
        cond["parents_le"] = int(par_le)

    return cond


def parse_operation(op_box: Optional[ET.Element]) -> Dict[str, Any]:
    """
    Extract op from a ChangeTableItem scope.
    Looks for <Operation .../> anywhere under op_box.
    """
    if op_box is None:
        return {}

    op_el = find_descendant(op_box, ["Operation"])
    if op_el is None:  # IMPORTANT: don't use "or op_box"
        op_el = op_box

    attrs = {strip_ns(k).lower(): to_scalar(v) for k, v in op_el.attrib.items()}

    op: Dict[str, Any] = {}
    if "kind" in attrs:
        op["kind"] = attrs["kind"]

    operand = (
        attrs.get("operandnodestate")
        or attrs.get("operand")
        or attrs.get("state")
    )
    if operand is not None:
        op["operand"] = operand

    # Preserve extra operation attributes (non-breaking)
    extras = {k: v for k, v in attrs.items() if k not in {"kind", "operandnodestate", "operand", "state"}}
    if extras:
        op["args"] = extras

    return op


def parse_change_table(root: ET.Element, xml_path: Path) -> Genome:
    """Parse a <ChangeTable> at any depth into Genome(rules=[...])."""
    # infer name and capacity
    name = (root.attrib.get("Name") or root.attrib.get("name") or xml_path.stem)
    capacity = None
    cap_str = root.attrib.get("Capacity") or root.attrib.get("capacity")
    if cap_str is not None:
        try:
            capacity = int(cap_str)
        except Exception:
            capacity = to_scalar(cap_str)

    # Collect all ChangeTableItem elements at ANY depth (handles x:Array or property wrappers)
    items: List[ET.Element] = []
    for e in root.iter():
        local = strip_ns(e.tag)
        base = local.split(".")[-1]  # supports "ChangeTableItem.Condition" property elements elsewhere
        if base.lower() == "changetableitem":
            items.append(e)

    rules: List[Rule] = []
    current_state_counter: Dict[str, int] = {}

    for item in items:
        # Prefer property containers if present; else just search inside the item
        cond_box = find_child_by_suffix(item, ["ChangeTableItem.Condition", "Condition"]) or item
        op_box   = find_child_by_suffix(item, ["ChangeTableItem.Operation", "Operation"]) or item

        cond = parse_operation_condition(cond_box)
        op   = parse_operation(op_box)

        if cond or op:  # now cond/op won’t be dropped by falsey Element fallback
            rules.append(Rule(condition=cond, op=op))
            cs = cond.get("current")
            if isinstance(cs, str):
                current_state_counter[cs] = current_state_counter.get(cs, 0) + 1

    # pick a start state hint if any conditions had 'current'
    start_hint = None
    if current_state_counter:
        start_hint = max(current_state_counter.items(), key=lambda kv: kv[1])[0]

    return Genome(name=str(name), rules=rules, capacity=capacity, start_state_hint=start_hint)


# ---- top-level XML dispatcher -----------------------------------------------
def parse_legacy_xml(xml_path: Path) -> Genome:
    """
    Detect legacy format and parse accordingly.
    Currently supports ChangeTable -> rules.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    root_tag = strip_ns(root.tag)

    # Explicit ChangeTable format
    if root_tag.lower().endswith("changetable"):
        return parse_change_table(root, xml_path)

    # If the file actually wraps ChangeTable as a child, pick it.
    for ch in list(root):
        if strip_ns(ch.tag).lower().endswith("changetable"):
            return parse_change_table(ch, xml_path)

    # Fallback: no rules recognized
    return Genome(name=xml_path.stem, rules=[])

# ---- JSON model --------------------------------------------------------------
def build_json_model(genome: Genome, *, include_start_hint: bool = False) -> Dict[str, Any]:
    """
    Build the intermediate JSON structure centered on 'rules'.
    Optionally include a start_state_hint when inference is explicitly requested.
    """
    j: Dict[str, Any] = {
        "format": "guca.legacy_rules.v1",
        "genome": {
            "name": genome.name,
            "rules": [],
        }
    }
    for r in genome.rules:
        entry = {}
        if r.condition:
            entry["condition"] = r.condition
        if r.op:
            entry["op"] = r.op
        j["genome"]["rules"].append(entry)

    meta: Dict[str, Any] = {}
    if genome.capacity is not None:
        meta["capacity"] = genome.capacity
    if include_start_hint and genome.start_state_hint:
        meta["start_state_hint"] = genome.start_state_hint

    if meta:
        j["meta"] = meta

    return j


# ---- YAML model (matches your example schema) --------------------------------
def default_machine(
    meta: Dict[str, Any],
    rules: List[Dict[str, Any]],
    *,
    start_state: Optional[str] = "A",
    infer_from_rules: bool = False
) -> Dict[str, Any]:
    """
    Machine block:
    - start_state: default 'A', unless --start-state overrides it.
    - If --infer-start-state is set, use meta.start_state_hint when available.
    - max_vertices: use meta.capacity if it's an int; else 2000 default.
    """
    # baseline: explicit CLI or default 'A'
    start = start_state or "A"

    # optional inference (opt-in only)
    if infer_from_rules:
        hint = meta.get("start_state_hint")
        if isinstance(hint, str) and hint:
            start = hint

    mv = meta.get("capacity")
    if not isinstance(mv, int):
        mv = 2000

    return {
        "start_state": start,
        "transcription": "resettable",
        "count_compare": "range",
        "max_vertices": mv,
        "max_steps": 120,
        "nearest_search": {
            "max_depth": 2,
            "tie_breaker": "stable",
            "connect_all": False,
        },
        # "rng_seed": 42,  # enable if you want determinism
    }


def default_init_graph(machine_block: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal init_graph: single node with the machine.start_state."""
    start = machine_block.get("start_state", "A")
    return {"nodes": [{"state": start}]}

def build_yaml_model(
    j: Dict[str, Any],
    *,
    start_state: Optional[str] = "A",
    infer_start_state: bool = False
) -> Dict[str, Any]:
    """
    Map the JSON model to the YAML schema, applying legacy->new fixes:
      - op.kind 'DisconectFrom'  -> 'DisconnectFrom' (typo fix)
      - condition.prior 'Min'    -> 'any'           (semantic change)
    """
    meta = j.get("meta", {})
    rules_in = j.get("genome", {}).get("rules", [])

    # Legacy -> new name fixes for op kinds (case-insensitive keys)
    OP_KIND_FIXES = {
        "disconectfrom": "DisconnectFrom",  # legacy typo -> new canonical
    }

    rules_out: List[Dict[str, Any]] = []
    for r in rules_in:
        entry: Dict[str, Any] = {}
        cond_in = r.get("condition") or {}
        op_in   = r.get("op") or {}

        # --- condition normalization with legacy mapping for 'prior' ---
        cond_norm: Dict[str, Any] = {}

        # current (unchanged)
        if "current" in cond_in:
            cond_norm["current"] = cond_in["current"]

        # prior: legacy 'Min' -> new 'any'
        if "prior" in cond_in:
            pv = cond_in["prior"]
            if isinstance(pv, str) and pv.strip().lower() == "min":
                cond_norm["prior"] = "any"
            else:
                cond_norm["prior"] = pv

        # numeric ranges (pass through if present)
        for k in ["conn_ge", "conn_le", "parents_ge", "parents_le"]:
            if k in cond_in:
                cond_norm[k] = cond_in[k]

        # --- op normalization with typo fix for kind ---
        op_norm: Dict[str, Any] = {}
        if "kind" in op_in and isinstance(op_in["kind"], str):
            raw_kind = op_in["kind"]
            key = raw_kind.strip().lower()
            op_norm["kind"] = OP_KIND_FIXES.get(key, raw_kind)  # apply fix if known
        if "operand" in op_in:
            op_norm["operand"] = op_in["operand"]
        # keep any extra args non-destructively
        if "args" in op_in and isinstance(op_in["args"], dict) and op_in["args"]:
            op_norm["args"] = op_in["args"]

        entry["condition"] = cond_norm
        entry["op"] = op_norm
        rules_out.append(entry)

    # machine/init_graph the same as before, honoring your start-state knobs
    machine_block = default_machine(
        meta,
        rules_out,
        start_state=start_state,
        infer_from_rules=infer_start_state
    )
    init_graph_block = default_init_graph(machine_block)

    return {
        "machine": machine_block,
        "init_graph": init_graph_block,
        "rules": rules_out,
    }



# ---- IO ----------------------------------------------------------------------
def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_yaml(obj: Dict[str, Any], path: Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    path.parent.mkdir(parents=True, exist_ok=True
    )
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

# ---- CLI ---------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert legacy ChangeTable XML -> intermediate JSON + YAML (machine/init_graph/rules).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="Legacy XML/XAML file (ChangeTable).")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output YAML path; defaults to input with .yaml")
    ap.add_argument("-j", "--json-out", type=Path, default=None, help="Output JSON path; defaults to input with .json")
    ap.add_argument(
        "--start-state",
        type=str,
        default="A",
        help="Start state for YAML machine block (default: A)."
    )
    ap.add_argument(
        "--infer-start-state",
        action="store_true",
        help="Infer start_state from the most frequent condition.current in rules (overrides --start-state)."
    )

    args = ap.parse_args(argv)

    xml_path: Path = args.input
    if not xml_path.exists():
        ap.error(f"Input not found: {xml_path}")

    json_out = args.json_out or xml_path.with_suffix(".json")
    yaml_out = args.output or xml_path.with_suffix(".yaml")

    genome = parse_legacy_xml(xml_path)

    # Build & write JSON (include start hint only if inference is requested)
    j = build_json_model(genome, include_start_hint=args.infer_start_state)
    write_json(j, json_out)

    # Build & write YAML with explicit start-state semantics
    y = build_yaml_model(
        j,
        start_state=args.start_state,
        infer_start_state=args.infer_start_state
    )
    write_yaml(y, yaml_out)


    print(f"[ok] JSON -> {json_out}")
    print(f"[ok] YAML -> {yaml_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

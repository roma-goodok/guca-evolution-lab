from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class TranscriptionWay(str, Enum):
    resettable = "resettable"
    continuable = "continuable"

class CountCompare(str, Enum):
    range = "range"
    exact = "exact"

class OperationKind(str, Enum):
    TurnToState = "TurnToState"
    TryToConnectWith = "TryToConnectWith"
    TryToConnectWithNearest = "TryToConnectWithNearest"
    GiveBirth = "GiveBirth"
    GiveBirthConnected = "GiveBirthConnected"
    Die = "Die"
    DisconnectFrom = "DisconnectFrom"

@dataclass
class Condition:
    current: str
    prior: Optional[str] = "any"
    conn_ge: int = -1
    conn_le: int = -1
    parents_ge: int = -1
    parents_le: int = -1

@dataclass
class Operation:
    kind: OperationKind
    operand: Optional[str] = None

@dataclass
class Rule:
    condition: Condition
    operation: Operation
    is_enabled: bool = True
    is_active: bool = False
    was_active: bool = False
    last_activation_index: int = -1

class ChangeTable(List[Rule]):
    pass

def _match_int(val: int, ge: int, le: int, cmp_mode: CountCompare) -> bool:
    if ge < 0 and le < 0:
        return True
    if cmp_mode == CountCompare.exact and ge >= 0:
        return val == ge
    if ge >= 0 and val < ge:
        return False
    if le >= 0 and val > le:
        return False
    return True

def rule_matches(saved_state: Optional[str], prior_state: Optional[str],
                 degree: int, parents_count: int, rule: Rule, cmp_mode: CountCompare) -> bool:
    if not rule.is_enabled:
        return False
    if rule.condition.current != saved_state:
        return False
    if rule.condition.prior not in ("any", None) and rule.condition.prior != prior_state:
        return False
    if not _match_int(degree, rule.condition.conn_ge, rule.condition.conn_le, cmp_mode):
        return False
    if not _match_int(parents_count, rule.condition.parents_ge, rule.condition.parents_le, cmp_mode):
        return False
    return True

def change_table_from_yaml(rules_yaml: List[dict]) -> ChangeTable:
    tbl = ChangeTable()
    for item in rules_yaml:
        c = item.get("condition", {})
        o = item.get("op", {})
        cond = Condition(
            current=str(c["current"]),
            prior=c.get("prior", "any"),
            conn_ge=int(c.get("conn_ge", -1)),
            conn_le=int(c.get("conn_le", -1)),
            parents_ge=int(c.get("parents_ge", -1)),
            parents_le=int(c.get("parents_le", -1)),
        )
        op = Operation(kind=OperationKind(o["kind"]), operand=o.get("operand"))
        tbl.append(Rule(condition=cond, operation=op))
    return tbl

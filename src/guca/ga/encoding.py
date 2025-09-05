# src/guca/ga/encoding.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any
import random

# -----------------------------
# 64-bit gene layout (MSB → LSB)
# [ 8 ] OP_KIND
# [ 8 ] COND_CURRENT_STATE_ID
# [ 8 ] OPERAND_STATE_ID (255 = NONE)
# [ 8 ] FLAGS (bitfield, reserved)
# [ 8 ] PRIOR_STATE_ID (255 = NONE)          <-- inside RESERVED (high byte)
# [ 4 ] CONN_GE  (0..14; 15 = NONE)
# [ 4 ] CONN_LE  (0..14; 15 = NONE)
# [ 4 ] PARENTS_GE (0..14; 15 = NONE)
# [ 4 ] PARENTS_LE (0..14; 15 = NONE)
# [ 8 ] RESERVED_LOW (unused; 0)
#
# The last 32 bits are the "RESERVED" area from v1; we now give them structure.
# Old v1 genes will decode with PRIOR=NONE and all *_GE/LE=None automatically.
# -----------------------------

_OPER_NONE = 0xFF
_NIBBLE_NONE = 0xF

class OpKind(IntEnum):
    TurnToState = 0
    GiveBirth = 1
    GiveBirthConnected = 2
    TryToConnectWith = 3
    TryToConnectWithNearest = 4
    DisconnectFrom = 5
    Die = 6

@dataclass
class Rule:
    """One GUCA rule row encoded by a single gene."""
    cond_current: int                 # state id (0..state_count-1)
    op_kind: OpKind
    operand: Optional[int] = None     # state id or None
    flags: int = 0                    # 8-bit flags, reserved
    # NEW: extended condition (Week 5)
    prior: Optional[int] = None
    conn_ge: Optional[int] = None
    conn_le: Optional[int] = None
    parents_ge: Optional[int] = None
    parents_le: Optional[int] = None

    def __int__(self) -> int:
        return encode_rule(self)

# -----------------------------
# State label mapping helpers
# -----------------------------

def labels_to_state_maps(states: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """
    Build (label->id, id->label) from an ordered list of state labels.
    """
    inv = list(states)
    fwd = {s: i for i, s in enumerate(inv)}
    return fwd, inv

# -----------------------------
# Packing / unpacking primitives
# -----------------------------

def _pack_fields(op_id: int, cond_id: int, oper_id_or_ff: int, flags: int, reserved: int) -> int:
    gene = (
        ((op_id & 0xFF) << 56)
        | ((cond_id & 0xFF) << 48)
        | ((oper_id_or_ff & 0xFF) << 40)
        | ((flags & 0xFF) << 32)
        | (reserved & 0xFFFFFFFF)
    )
    return gene & 0xFFFFFFFFFFFFFFFF

def _unpack_fields(gene: int) -> Tuple[int, int, int, int, int]:
    return (
        (gene >> 56) & 0xFF,  # op
        (gene >> 48) & 0xFF,  # cond
        (gene >> 40) & 0xFF,  # oper
        (gene >> 32) & 0xFF,  # flags
        gene & 0xFFFFFFFF,    # reserved
    )

def _build_reserved(prior: Optional[int], conn_ge: Optional[int], conn_le: Optional[int],
                    parents_ge: Optional[int], parents_le: Optional[int]) -> int:
    p = _OPER_NONE if prior is None else int(prior) & 0xFF
    cg = _NIBBLE_NONE if conn_ge is None else int(conn_ge) & 0xF
    cl = _NIBBLE_NONE if conn_le is None else int(conn_le) & 0xF
    pg = _NIBBLE_NONE if parents_ge is None else int(parents_ge) & 0xF
    pl = _NIBBLE_NONE if parents_le is None else int(parents_le) & 0xF
    return ((p & 0xFF) << 24) | ((cg & 0xF) << 20) | ((cl & 0xF) << 16) | ((pg & 0xF) << 12) | ((pl & 0xF) << 8)

def _parse_reserved(res: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    prior = (res >> 24) & 0xFF
    conn_ge = (res >> 20) & 0xF
    conn_le = (res >> 16) & 0xF
    parents_ge = (res >> 12) & 0xF
    parents_le = (res >> 8) & 0xF
    prior = None if prior == _OPER_NONE else int(prior)
    def de(n: int) -> Optional[int]:
        return None if n == _NIBBLE_NONE else int(n)
    return prior, de(conn_ge), de(conn_le), de(parents_ge), de(parents_le)

# -----------------------------
# Public API
# -----------------------------

def encode_rule(rule: Rule) -> int:
    op = int(rule.op_kind) & 0xFF
    cond = int(rule.cond_current) & 0xFF
    oper = _OPER_NONE if rule.operand is None else (int(rule.operand) & 0xFF)
    flags = int(rule.flags) & 0xFF
    reserved = _build_reserved(rule.prior, rule.conn_ge, rule.conn_le, rule.parents_ge, rule.parents_le)
    return _pack_fields(op, cond, oper, flags, reserved)

def decode_gene(gene: int, *, state_count: int) -> Rule:
    op_id, cond_id, oper_id, flags, res = _unpack_fields(int(gene))
    try:
        opk = OpKind(op_id)
    except ValueError:
        opk = OpKind.TurnToState
    operand = None if oper_id == _OPER_NONE else int(oper_id)
    prior, conn_ge, conn_le, parents_ge, parents_le = _parse_reserved(res)
    # Clamp ids into domain
    cond_id = int(cond_id) % max(1, state_count)
    operand = None if operand is None else (int(operand) % max(1, state_count))
    prior = None if prior is None else (int(prior) % max(1, state_count))
    return Rule(
        cond_current=cond_id,
        op_kind=opk,
        operand=operand,
        flags=flags,
        prior=prior,
        conn_ge=conn_ge,
        conn_le=conn_le,
        parents_ge=parents_ge,
        parents_le=parents_le,
    )


def random_gene(state_count: int, rng: Optional[random.Random] = None) -> int:
    """
    Create a random, domain-safe 64-bit gene.
    """
    r = rng or random
    op = r.randrange(len(OpKind))
    cond = r.randrange(max(1, state_count))
    # 1/5 chance of None operand for diversity
    oper = _OPER_NONE if r.random() < 0.2 else r.randrange(max(1, state_count))
    flags = r.getrandbits(8)

    # Extended condition: default to 'None' (unset)
    prior = None if r.random() < 0.8 else r.randrange(max(1, state_count))
    # Use modest chance to set bounds, otherwise leave None (unset)
    def maybe_bound(p=0.3, hi=6):
        return None if r.random() > p else r.randrange(0, max(1, hi))
    conn_ge = maybe_bound()
    conn_le = maybe_bound()
    parents_ge = maybe_bound(p=0.2, hi=3)
    parents_le = maybe_bound(p=0.2, hi=3)

    reserved = _build_reserved(prior, conn_ge, conn_le, parents_ge, parents_le)
    return _pack_fields(op, cond, oper, flags, reserved)

def sanitize_gene(gene: int, state_count: int) -> int:
    """
    Normalize a raw 64-bit gene into a domain-safe encoding:
    - decode (which already clamps enums/ids/nibbles),
    - re-encode into our 64-bit layout,
    - and hard-mask to 64 bits.
    """
    try:
        rule = decode_gene(int(gene) & 0xFFFFFFFFFFFFFFFF, state_count=state_count)
        return int(encode_rule(rule)) & 0xFFFFFFFFFFFFFFFF
    except Exception:
        # Extremely defensive: if something goes wrong, fall back to a random safe gene.
        # (Tests should never hit this, but keeps behavior consistent with plan.)
        return random_gene(state_count)

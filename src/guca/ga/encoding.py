# src/guca/ga/encoding.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import random


# -----------------------------
# 64-bit gene layout (MSB → LSB)
# [ 8 ] OP_KIND
# [ 8 ] COND_STATE_ID
# [ 8 ] OPERAND_STATE_ID (255 = NONE)
# [ 8 ] FLAGS (bitfield, reserved for future)
# [32 ] RESERVED (random payload / future use)
# -----------------------------

# bit shifts
_OP_SHIFT = 56
_COND_SHIFT = 48
_OPER_SHIFT = 40
_FLAGS_SHIFT = 32
# masks
_BYTE = 0xFF
_U32 = 0xFFFFFFFF
_MASK64 = (1 << 64) - 1

# "no operand" sentinel
_OPER_NONE = 0xFF


class OpKind(IntEnum):
    """Operation kinds aligned with YAML names."""
    TurnToState = 0
    GiveBirth = 1
    GiveBirthConnected = 2
    TryToConnectWith = 3
    TryToConnectWithNearest = 4
    DisconnectFrom = 5
    Die = 6

    @classmethod
    def from_id(cls, value: int) -> "OpKind":
        """Map any int into a valid OpKind (modulo domain)."""
        keys = list(cls)
        return keys[value % len(keys)]


@dataclass
class Rule:
    """One GUCA rule row encoded by a single gene."""
    cond_current: int            # state id (0..state_count-1)
    op_kind: OpKind
    operand: Optional[int] = None  # state id or None
    flags: int = 0                # 8-bit flags, reserved for future


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
        ((op_id & _BYTE) << _OP_SHIFT)
        | ((cond_id & _BYTE) << _COND_SHIFT)
        | ((oper_id_or_ff & _BYTE) << _OPER_SHIFT)
        | ((flags & _BYTE) << _FLAGS_SHIFT)
        | (reserved & _U32)
    )
    return gene & _MASK64


def _unpack_fields(gene: int) -> Tuple[int, int, int, int, int]:
    gene &= _MASK64
    op_id = (gene >> _OP_SHIFT) & _BYTE
    cond_id = (gene >> _COND_SHIFT) & _BYTE
    oper_id = (gene >> _OPER_SHIFT) & _BYTE
    flags = (gene >> _FLAGS_SHIFT) & _BYTE
    reserved = gene & _U32
    return op_id, cond_id, oper_id, flags, reserved


# -----------------------------
# Public API
# -----------------------------

def encode_rule(rule: Rule, *, reserved: Optional[int] = None) -> int:
    """
    Encode a Rule into a 64-bit gene. `reserved` can be provided for determinism; else random.
    """
    op_id = int(rule.op_kind)
    cond = int(rule.cond_current)
    oper = _OPER_NONE if rule.operand is None else int(rule.operand)
    flags = int(rule.flags) & _BYTE
    if reserved is None:
        reserved = random.getrandbits(32)
    return _pack_fields(op_id, cond, oper, flags, reserved)


def decode_gene(gene: int, *, state_count: Optional[int] = None) -> Rule:
    """
    Decode a 64-bit gene into a Rule. If `state_count` is provided, fields are clamped into range.
    """
    op_id, cond, oper, flags, _ = _unpack_fields(gene)
    op = OpKind.from_id(op_id)
    if state_count is not None and state_count > 0:
        cond %= state_count
        if oper != _OPER_NONE:
            oper %= state_count
    operand = None if oper == _OPER_NONE else int(oper)
    return Rule(cond_current=int(cond), op_kind=op, operand=operand, flags=int(flags))


def sanitize_gene(gene: int, *, state_count: int) -> int:
    """
    Clamp fields to valid domains:
      - op_kind -> modulo valid enum size
      - cond/operand -> modulo [0, state_count)
    """
    op_id, cond, oper, flags, reserved = _unpack_fields(gene)
    op = OpKind.from_id(op_id)
    cond = cond % max(1, state_count)
    if oper != _OPER_NONE:
        oper = oper % max(1, state_count)
    return _pack_fields(int(op), int(cond), int(oper), int(flags), int(reserved))


def random_gene(*, state_count: int, rng: Optional[random.Random] = None) -> int:
    """
    Create a random, domain-safe 64-bit gene.
    """
    r = rng or random
    op = r.randrange(len(OpKind))
    cond = r.randrange(max(1, state_count))
    # 1/5 chance of None operand for diversity
    if r.random() < 0.2:
        oper = _OPER_NONE
    else:
        oper = r.randrange(max(1, state_count))
    flags = r.getrandbits(8)
    reserved = r.getrandbits(32)
    return _pack_fields(op, cond, oper, flags, reserved)

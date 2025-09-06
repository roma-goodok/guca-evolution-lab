from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import random

# 64-bit gene layout (MSB → LSB)
# [ 8 ] OP_KIND (canonical legacy domain 0..3 when sanitized)
# [ 8 ] COND_STATE_ID
# [ 8 ] OPERAND_STATE_ID (255 = NONE in raw; canonical 0 when sanitized & unset)
# [ 8 ] FLAGS
# [ 8 ] PRIOR_STATE_ID (255 = NONE in raw; canonical 0 when sanitized & unset)    <-- reserved[31:24]
# [ 4 ] CONN_GE  (0..14; 15 = NONE in raw; canonical 0 when unset)                <-- reserved[23:20]
# [ 4 ] CONN_LE  (0..14; 15 = NONE in raw; canonical 0 when unset)                <-- reserved[19:16]
# [ 4 ] PARENTS_GE (0..14; 15 = NONE in raw; canonical 0 when unset)              <-- reserved[15:12]
# [ 4 ] PARENTS_LE (0..14; 15 = NONE in raw; canonical 0 when unset)              <-- reserved[11:8]
# [ 8 ] RESERVED_LOW = 0x00                                                       <-- reserved[7:0] always 0

_BYTE      = 0xFF
_NIBBLE    = 0xF
_U32       = 0xFFFFFFFF
_MASK64    = (1 << 64) - 1

_OP_SHIFT    = 56
_COND_SHIFT  = 48
_OPER_SHIFT  = 40
_FLAGS_SHIFT = 32

_OPER_NONE   = 0xFF
_PRIOR_NONE  = 0xFF
_NIBBLE_NONE = 0xF

from enum import IntEnum

class OpKind(IntEnum):
    # Canonical 4-kind domain (legacy C#: OperationType % 4)
    TurnToState             = 0
    TryToConnectWithNearest = 1
    GiveBirthConnected      = 2
    DisconnectFrom          = 3

    # ---- Back-compat aliases (older names still used in code/tests) ----
    GiveBirth               = 2  # alias of GiveBirthConnected
    TryToConnectNearest     = 1  # alias of TryToConnectWithNearest
    TryToConnectWith        = 1  # alias used by older Python code

    @classmethod
    def from_id(cls, value: int) -> "OpKind":
        members = [cls.TurnToState, cls.TryToConnectWithNearest, cls.GiveBirthConnected, cls.DisconnectFrom]
        return members[int(value) % len(members)]



@dataclass  # mutable; tests and GA modify decoded rules
class Rule:
    cond_current: int
    op_kind: OpKind
    operand: Optional[int] = None
    flags: int = 0
    # extended predicates (kept in low 32 bits)
    prior: Optional[int] = None
    conn_ge: Optional[int] = None
    conn_le: Optional[int] = None
    parents_ge: Optional[int] = None
    parents_le: Optional[int] = None

    def __int__(self) -> int:
        return encode_rule(self)

def labels_to_state_maps(states: List[str]) -> Tuple[Dict[str, int], List[str]]:
    inv = list(states)
    fwd = {s: i for i, s in enumerate(inv)}
    return fwd, inv

# ---------- low-32 helpers ----------

def _build_reserved(prior: Optional[int],
                    conn_ge: Optional[int], conn_le: Optional[int],
                    parents_ge: Optional[int], parents_le: Optional[int]) -> int:
    p  = _PRIOR_NONE  if prior is None     else (int(prior) & _BYTE)
    cg = _NIBBLE_NONE if conn_ge is None   else (int(conn_ge) & _NIBBLE)
    cl = _NIBBLE_NONE if conn_le is None   else (int(conn_le) & _NIBBLE)
    pg = _NIBBLE_NONE if parents_ge is None else (int(parents_ge) & _NIBBLE)
    pl = _NIBBLE_NONE if parents_le is None else (int(parents_le) & _NIBBLE)
    return ((p & _BYTE) << 24) | ((cg & _NIBBLE) << 20) | ((cl & _NIBBLE) << 16) | ((pg & _NIBBLE) << 12) | ((pl & _NIBBLE) << 8) | 0x00

def _parse_reserved(res: int) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    p  = (res >> 24) & _BYTE
    cg = (res >> 20) & _NIBBLE
    cl = (res >> 16) & _NIBBLE
    pg = (res >> 12) & _NIBBLE
    pl = (res >>  8) & _NIBBLE
    def de(n, sent): return None if n == sent else int(n)
    return (None if p == _PRIOR_NONE else int(p),
            de(cg, _NIBBLE_NONE), de(cl, _NIBBLE_NONE),
            de(pg, _NIBBLE_NONE), de(pl, _NIBBLE_NONE))

# ---------- pack/unpack ----------

def _pack_fields(op_id: int, cond_id: int, oper_id: int, flags: int, reserved: int) -> int:
    return (
        ((op_id & _BYTE)   << _OP_SHIFT   ) |
        ((cond_id & _BYTE) << _COND_SHIFT ) |
        ((oper_id & _BYTE) << _OPER_SHIFT ) |
        ((flags & _BYTE)   << _FLAGS_SHIFT) |
        ( reserved & _U32 )
    ) & _MASK64

def _unpack_fields(gene: int) -> Tuple[int, int, int, int, int]:
    g = int(gene) & _MASK64
    return ((g >> _OP_SHIFT) & _BYTE,
            (g >> _COND_SHIFT) & _BYTE,
            (g >> _OPER_SHIFT) & _BYTE,
            (g >> _FLAGS_SHIFT) & _BYTE,
            g & _U32)

# ---------- public API ----------

def encode_rule(rule: Rule, *, reserved: Optional[int] = None) -> int:
    """
    Encode a Rule into a 64-bit gene.
    `reserved` is accepted for back-compat with old tests but ignored: we derive
    low-32 from extended fields so decode↔encode round-trips match Rule content.
    """
    op   = int(rule.op_kind) & _BYTE            # already 0..3 in this enum
    cond = int(rule.cond_current) & _BYTE
    oper = _OPER_NONE if rule.operand is None else (int(rule.operand) & _BYTE)
    flg  = int(rule.flags) & _BYTE
    res  = _build_reserved(rule.prior, rule.conn_ge, rule.conn_le, rule.parents_ge, rule.parents_le)
    return _pack_fields(op, cond, oper, flg, res)

def decode_gene(gene: int, *, state_count: Optional[int] = None) -> Rule:
    op_id, cond, oper, flags, res = _unpack_fields(gene)
    # Map whatever raw id is there into legacy 4-op domain (0..3) as C# does.
    op = OpKind.from_id(op_id)
    prior, conn_ge, conn_le, parents_ge, parents_le = _parse_reserved(res)
    if state_count is not None and state_count > 0:
        cond %= state_count
        if oper != _OPER_NONE: oper %= state_count
        if prior is not None:  prior %= state_count
    operand = None if oper == _OPER_NONE else int(oper)
    return Rule(cond_current=int(cond), op_kind=op, operand=operand, flags=int(flags),
                prior=prior, conn_ge=conn_ge, conn_le=conn_le, parents_ge=parents_ge, parents_le=parents_le)

def sanitize_gene(
    gene: int,
    *,
    state_count: int,
    enforce_semantics: bool = False,
    canonicalize_flags: bool = False,
    enforce_bounds_order: bool = False,
    canonicalize_unset_to_zero: bool = True,
) -> int:
    """
    Canonicalize any 64-bit gene into the Week-4 validator's raw layout.

    Always:
      - map op id into the legacy 4-op domain (0..3),
      - clamp cond/operand/prior to [0..state_count-1] (operand/prior may be unset),
      - clamp nibble constraints to 0..14 or None (=15),
      - keep low reserved byte = 0x00.

    If enforce_bounds_order=True:
      - when both bounds are set, ensure ge <= le.

    If canonicalize_flags=True:
      - force flags to 0.

    If canonicalize_unset_to_zero=True (default):
      - encode all *unset* extended predicates as zeros (not 0xFF/0xF),
      - **encode operand byte as 0 for ops that don't use it**;
        only TurnToState carries a meaningful operand.
    """
    op_id, cond, oper_raw, flags, res_raw = _unpack_fields(gene)

    # Legacy 4-op domain as in C# (OperationType % 4)
    op = OpKind.from_id(op_id)

    # Parse extended predicates
    prior, conn_ge, conn_le, parents_ge, parents_le = _parse_reserved(res_raw)

    # Basic clamps
    state_count = max(1, int(state_count))
    cond = int(cond) % state_count

    oper = oper_raw
    if oper != _OPER_NONE:
        oper = int(oper) % state_count
    if prior is not None:
        prior = int(prior) % state_count

    def clamp_nib(n: Optional[int]) -> Optional[int]:
        if n is None:
            return None
        return max(0, min(14, int(n)))

    conn_ge     = clamp_nib(conn_ge)
    conn_le     = clamp_nib(conn_le)
    parents_ge  = clamp_nib(parents_ge)
    parents_le  = clamp_nib(parents_le)

    # Optional range ordering
    if enforce_bounds_order:
        if conn_ge is not None and conn_le is not None and conn_ge > conn_le:
            conn_ge, conn_le = conn_le, conn_ge
        if parents_ge is not None and parents_le is not None and parents_ge > parents_le:
            parents_ge, parents_le = parents_le, parents_ge

    if canonicalize_flags:
        flags = 0

    # === Canonical raw-byte encoding ===
    # 1) operand policy:
    #    - Only TurnToState uses operand; others must encode operand byte as 0.
    if canonicalize_unset_to_zero:
        if op == OpKind.TurnToState:
            # meaningful operand; if logically unset, make a deterministic fallback
            if enforce_semantics and oper == _OPER_NONE:
                oper_out = cond & _BYTE
            else:
                oper_out = (0 if oper == _OPER_NONE else int(oper)) & _BYTE
        else:
            # op doesn't use operand → canonical raw byte must be 0
            oper_out = 0

        # 2) extended predicates: encode "unset" as zeros
        p_out  = 0 if prior is None     else (int(prior) & _BYTE)
        cg_out = 0 if conn_ge is None   else (int(conn_ge) & _NIBBLE)
        cl_out = 0 if conn_le is None   else (int(conn_le) & _NIBBLE)
        pg_out = 0 if parents_ge is None else (int(parents_ge) & _NIBBLE)
        pl_out = 0 if parents_le is None else (int(parents_le) & _NIBBLE)

        reserved = (
            ((p_out  & _BYTE)   << 24) |
            ((cg_out & _NIBBLE) << 20) |
            ((cl_out & _NIBBLE) << 16) |
            ((pg_out & _NIBBLE) << 12) |
            ((pl_out & _NIBBLE) <<  8) |
            0x00
        )
    else:
        # Sentinel style (C#-like raw), if you ever need it:
        oper_out = int(oper) & _BYTE if oper != _OPER_NONE else _OPER_NONE
        reserved = _build_reserved(prior, conn_ge, conn_le, parents_ge, parents_le)

    return _pack_fields(int(op), int(cond), int(oper_out), int(flags), int(reserved))


def random_gene(*, state_count: int, rng: Optional[random.Random] = None) -> int:
    """
    C#-style random: keep silent parts; engine ignores irrelevant fields.
    (Only structure is guaranteed; not canonical by default.)
    """
    r = rng or random
    state_count = max(1, int(state_count))
    op_id = r.randrange(256)         # any byte; decode() reduces to 0..3 via from_id
    cond  = r.randrange(state_count)
    # 20% unset regardless of op — like legacy raw
    oper  = _OPER_NONE if r.random() < 0.2 else r.randrange(state_count)
    flags = r.getrandbits(8)

    prior = None if r.random() < 0.8 else r.randrange(state_count)
    def maybe_nib(p=0.25, hi=8):
        if r.random() >= p: return None
        return r.randrange(0, min(15, max(1, hi)))  # 0..14 allowed

    conn_ge    = maybe_nib(p=0.35, hi=6)
    conn_le    = maybe_nib(p=0.35, hi=6)
    parents_ge = maybe_nib(p=0.20, hi=3)
    parents_le = maybe_nib(p=0.20, hi=3)

    res = _build_reserved(prior, conn_ge, conn_le, parents_ge, parents_le)
    return _pack_fields(op_id & 0xFF, cond, oper, flags, res)

__all__ = [
    "OpKind",
    "Rule",
    "encode_rule",
    "decode_gene",
    "sanitize_gene",
    "random_gene",
    "labels_to_state_maps",
]

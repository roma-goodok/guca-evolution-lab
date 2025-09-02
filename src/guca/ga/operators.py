# src/guca/ga/operators.py
from __future__ import annotations

from typing import List, Tuple, Optional
import random

from guca.ga.encoding import OpKind, sanitize_gene, random_gene


# -----------------------------
# Crossover
# -----------------------------
def splice_cx(a: List[int], b: List[int], *, rng: Optional[random.Random] = None, unaligned: bool = True) -> Tuple[List[int], List[int]]:
    """
    Two-point splice crossover. If unaligned=True, cut points are per-gene indices
    but splice lengths may differ (length-changing). If unaligned=False, segments must
    have equal length, preserving chromosome lengths.
    """
    r = rng or random
    if not a or not b:
        return a[:], b[:]

    i1 = r.randrange(0, len(a) + 1)
    i2 = r.randrange(i1, len(a) + 1)
    j1 = r.randrange(0, len(b) + 1)
    j2 = r.randrange(j1, len(b) + 1)

    seg_a = a[i1:i2]
    seg_b = b[j1:j2]

    if not unaligned and (len(seg_a) != len(seg_b)):
        # force equal-length by shrinking the longer one
        m = min(len(seg_a), len(seg_b))
        seg_a = seg_a[:m]
        seg_b = seg_b[:m]

    c1 = a[:i1] + seg_b + a[i2:]
    c2 = b[:j1] + seg_a + b[j2:]
    return c1, c2


# -----------------------------
# Structural mutations
# -----------------------------
def mut_insert(ch: List[int], *, n: int, state_count: int, rng: Optional[random.Random] = None) -> List[int]:
    r = rng or random
    pos = r.randrange(0, len(ch) + 1)
    genes = [random_gene(state_count=state_count, rng=r) for _ in range(max(1, n))]
    return ch[:pos] + genes + ch[pos:]


def mut_delete(ch: List[int], *, n: int, rng: Optional[random.Random] = None) -> List[int]:
    r = rng or random
    if not ch:
        return ch
    pos = r.randrange(0, len(ch))
    span = min(max(1, n), len(ch) - pos)
    return ch[:pos] + ch[pos + span:]


def mut_duplicate(ch: List[int], *, span: int, rng: Optional[random.Random] = None) -> List[int]:
    r = rng or random
    if not ch:
        return ch
    pos = r.randrange(0, len(ch))
    span = max(1, min(span, len(ch) - pos))
    block = ch[pos:pos + span]
    ins = r.randrange(0, len(ch) + 1)
    return ch[:ins] + block + ch[ins:]


# -----------------------------
# Field-level mutations (64-bit)
# -----------------------------
def mut_bitflip_gene(g: int, *, p: float, rng: Optional[random.Random] = None) -> int:
    r = rng or random
    if r.random() < p:
        bit = 1 << r.randrange(64)
        g ^= bit
    return g & ((1 << 64) - 1)


def mut_byte_gene(g: int, *, rng: Optional[random.Random] = None) -> int:
    r = rng or random
    byte_idx = r.randrange(8)  # 0..7 (LSB..MSB)
    mask = 0xFF << (8 * byte_idx)
    val = r.randrange(256)
    g = (g & ~mask) | (val << (8 * byte_idx))
    return g & ((1 << 64) - 1)


def mut_allbytes_gene(g: int, *, p_byte: float, rng: Optional[random.Random] = None) -> int:
    r = rng or random
    for i in range(8):
        if r.random() < p_byte:
            mask = 0xFF << (8 * i)
            val = r.randrange(256)
            g = (g & ~mask) | (val << (8 * i))
    return g & ((1 << 64) - 1)


def mut_rotate_gene(g: int, *, by: str = "byte") -> int:
    if by == "byte":
        # rotate left by one byte
        hi = (g >> 56) & 0xFF
        rest = (g << 8) & ((1 << 64) - 1)
        return rest | hi
    elif by == "nibble":
        hi = (g >> 60) & 0xF
        rest = (g << 4) & ((1 << 64) - 1)
        return rest | hi
    return g


def mut_enum_delta(g: int, *, state_count: int, rng: Optional[random.Random] = None) -> int:
    """
    Small delta on enum-like fields (op kind, cond, operand).
    """
    from guca.ga.encoding import _unpack_fields, _pack_fields, _OPER_NONE
    r = rng or random
    op_id, cond, oper, flags, reserved = _unpack_fields(g)

    # tweak op kind by +/-1
    if r.random() < 0.5:
        op_id = (op_id + 1) % len(OpKind)
    else:
        op_id = (op_id - 1) % len(OpKind)

    # tweak cond by +/-1
    if state_count > 0:
        if r.random() < 0.5:
            cond = (cond + 1) % state_count
        else:
            cond = (cond - 1) % state_count

        # tweak operand if present
        if oper != _OPER_NONE:
            if r.random() < 0.5:
                oper = (oper + 1) % state_count
            else:
                oper = (oper - 1) % state_count

    return _pack_fields(op_id, cond, oper, flags, reserved)


# -----------------------------
# Repair & mutate wrapper
# -----------------------------
def repair_chromosome(ch: List[int], *, state_count: int, min_len: int, max_len: int) -> List[int]:
    ch = [sanitize_gene(g, state_count=state_count) for g in ch]
    if len(ch) < min_len:
        # pad with random valid genes
        pad = [random_gene(state_count=state_count) for _ in range(min_len - len(ch))]
        ch = ch + pad
    if len(ch) > max_len:
        ch = ch[:max_len]
    return ch


def make_mutate_fn(
    *,
    state_count: int,
    min_len: int,
    max_len: int,
    structural_cfg: dict,
    field_cfg: dict,
    rng: Optional[random.Random] = None,
):
    """
    Returns a DEAP-compatible mutate(individual) -> (individual,) function that mutates in-place.
    """
    r = rng or random
    insert_pb = float(structural_cfg.get("insert_pb", 0.2))
    delete_pb = float(structural_cfg.get("delete_pb", 0.15))
    duplicate_pb = float(structural_cfg.get("duplicate_pb", 0.10))

    bitflip_pb = float(field_cfg.get("bitflip_pb", 0.10))
    byte_pb = float(field_cfg.get("byte_pb", 0.05))
    allbytes_pb = float(field_cfg.get("allbytes_pb", 0.02))
    rotate_pb = float(field_cfg.get("rotate_pb", 0.05))
    enum_delta_pb = float(field_cfg.get("enum_delta_pb", 0.15))

    def mutate(individual: List[int]):
        # Structural edits (each with its own probability)
        ch = individual
        if r.random() < insert_pb:
            ch = mut_insert(ch, n=1, state_count=state_count, rng=r)
        if r.random() < delete_pb and len(ch) > min_len:
            ch = mut_delete(ch, n=1, rng=r)
        if r.random() < duplicate_pb and len(ch) < max_len:
            span = 1 if len(ch) < 3 else r.randrange(1, min(3, len(ch)))
            ch = mut_duplicate(ch, span=span, rng=r)

        # Field-level on each gene (low probabilities)
        out = []
        for g in ch:
            if r.random() < bitflip_pb:
                g = mut_bitflip_gene(g, p=1.0 / 64.0, rng=r)
            if r.random() < byte_pb:
                g = mut_byte_gene(g, rng=r)
            if r.random() < allbytes_pb:
                g = mut_allbytes_gene(g, p_byte=1.0 / 8.0, rng=r)
            if r.random() < rotate_pb:
                g = mut_rotate_gene(g, by="byte")
            if r.random() < enum_delta_pb:
                g = mut_enum_delta(g, state_count=state_count, rng=r)
            out.append(g)

        out = repair_chromosome(out, state_count=state_count, min_len=min_len, max_len=max_len)
        individual[:] = out
        return (individual,)

    return mutate

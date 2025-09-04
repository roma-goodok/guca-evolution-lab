# src/guca/ga/operators.py
from __future__ import annotations

from typing import List, Tuple, Callable, Optional
import random

# Gene (rule) encoding/decoding and enums
from guca.ga.encoding import (
    Rule,
    OpKind,
    decode_gene,
    encode_rule,
    random_gene,
)

_MASK64 = (1 << 64) - 1


# ---------- Low-level, 64-bit-safe mutations on raw integer genes ----------

def mutate_bitflip(g: int, rng: random.Random) -> int:
    """
    Flip a few random bits (1..4) in a 64-bit gene.
    """
    n_flips = 1 + rng.randrange(4)
    for _ in range(n_flips):
        b = rng.randrange(64)
        g ^= (1 << b)
    return g & _MASK64


def mutate_byte(g: int, rng: random.Random) -> int:
    """
    Replace exactly one of 8 bytes with a fresh random byte.
    """
    k = rng.randrange(8)                 # which byte (0..7), 0 is least significant
    byte_val = rng.randrange(256)
    shift = 8 * k
    mask = ~(0xFF << shift) & _MASK64
    g = (g & mask) | ((byte_val & 0xFF) << shift)
    return g & _MASK64


def mutate_allbytes(g: int, rng: random.Random) -> int:
    """
    Replace the whole 64-bit value with a fresh random 64-bit.
    """
    # Build 8 random bytes to avoid Python's unlimited int range surprises
    val = 0
    for k in range(8):
        val |= (rng.randrange(256) & 0xFF) << (8 * k)
    return val & _MASK64


def mutate_rotate(g: int, rng: random.Random) -> int:
    """
    Rotate left by one byte (8 bits) to mimic the legacy 'Shift'.
    """
    g &= _MASK64
    return (((g << 8) & _MASK64) | (g >> 56)) & _MASK64


def mutate_enum_delta(g: int, *, state_count: int, rng: random.Random) -> int:
    """
    Decode the gene into a structured Rule, nudge enumerated fields slightly,
    then re-encode. Keeps enums in-range and op_kind within OpKind bounds.
    """
    try:
        r = decode_gene(g, state_count=state_count)
    except Exception:
        # If decoding fails (shouldn't happen for valid genes), randomize as fallback
        return mutate_allbytes(g, rng)

    # Carefully perturb enumerated fields
    # 1) cond_current in [0..state_count-1]
    if hasattr(r, "cond_current"):
        delta = rng.choice([-1, +1])
        r.cond_current = int((int(r.cond_current) + delta) % max(1, state_count))

    # 2) operand may be None or an enum in state domain
    if hasattr(r, "operand"):
        if r.operand is None:
            # sometimes introduce an operand
            if rng.random() < 0.25 and state_count > 0:
                r.operand = int(rng.randrange(state_count))
        else:
            if state_count > 0:
                delta = rng.choice([-1, +1])
                r.operand = int((int(r.operand) + delta) % state_count)
            # occasionally drop operand
            if rng.random() < 0.05:
                r.operand = None

    # 3) op_kind: small local change inside the enum range
    if hasattr(r, "op_kind") and isinstance(r.op_kind, OpKind):
        all_kinds = list(OpKind)
        idx = all_kinds.index(r.op_kind)
        if rng.random() < 0.15:
            idx = (idx + rng.choice([-1, +1])) % len(all_kinds)
            r.op_kind = all_kinds[idx]

    try:
        g2 = encode_rule(r, state_count=state_count)
    except Exception:
        # If re-encoding fails (e.g., unexpected shape), keep original
        g2 = g
    return int(g2) & _MASK64


# ----------------------------- Crossover (splice) -----------------------------

def splice_cx(ind1: List[int], ind2: List[int], *, rng: random.Random, unaligned: bool = True) -> Tuple[List[int], List[int]]:
    """
    One-point splice crossover with variable lengths.
    If `unaligned` is True, cut points are chosen independently in each parent.
    """
    L1, L2 = len(ind1), len(ind2)
    if L1 == 0 or L2 == 0:
        return ind1[:], ind2[:]

    # Choose cut points in [1..L-1] when possible, otherwise 0
    c1 = rng.randrange(1, L1) if L1 > 1 else 0
    if unaligned:
        c2 = rng.randrange(1, L2) if L2 > 1 else 0
    else:
        # clamp to range of second parent
        c2 = min(max(c1, 0), max(L2 - 1, 0))

    child1 = ind1[:c1] + ind2[c2:]
    child2 = ind2[:c2] + ind1[c1:]
    return child1, child2


# ----------------------------- Gene validity check ----------------------------

def _valid_gene(g: int) -> bool:
    """
    Quick sanity: decodes & re-encodes with a small state domain; ensures no exception.
    We do not require exact round-trips bit-for-bit (encoders may normalize).
    """
    try:
        r = decode_gene(g, state_count=8)  # small domain (enough for tests A..H)
        _ = encode_rule(r, state_count=8)
        return True
    except Exception:
        return False


# ------------------------ Mutation factory (GA operators) ---------------------

def make_mutate_fn(
    *,
    state_count: int,
    min_len: int,
    max_len: int,
    structural_cfg: dict,
    field_cfg: dict,
    active_cfg: dict,
    passive_cfg: dict,
    structuralx_cfg: dict,
    rng: random.Random,
) -> Callable[[List[int]], List[int]]:
    """
    Builds a mutation function that combines:
      - legacy structural ops (insert/delete/duplicate)
      - C#-inspired extras (insert_active, delete_inactive, duplicate_head)
      - field-level bit/byte rotations and enum nudges (enum_delta)
      - coarse 'active/passive' regimes (applied uniformly as we don't track activity yet)
    All changes respect [min_len, max_len].
    """

    # --- Unpack structural (legacy) ---
    ins_pb = float(structural_cfg.get("insert_pb", 0.0))
    del_pb = float(structural_cfg.get("delete_pb", 0.0))
    dup_pb = float(structural_cfg.get("duplicate_pb", 0.0))

    # --- Unpack field-level ---
    bitflip_pb   = float(field_cfg.get("bitflip_pb", 0.0))
    byte_pb      = float(field_cfg.get("byte_pb", 0.0))
    allbytes_pb  = float(field_cfg.get("allbytes_pb", 0.0))
    rotate_pb    = float(field_cfg.get("rotate_pb", 0.0))
    enum_delta_pb= float(field_cfg.get("enum_delta_pb", 0.0))

    # --- C#-inspired regimes (applied uniformly due to no activity mask yet) ---
    # active/passive 'kind' can be: "bit", "byte", "all_bytes"
    active_factor = float(active_cfg.get("factor", 0.0))
    active_kind   = str(active_cfg.get("kind", "byte")).lower()
    active_shift  = float(active_cfg.get("shift_pb", 0.0))   # rotate chance

    passive_factor = float(passive_cfg.get("factor", 0.0))
    passive_kind   = str(passive_cfg.get("kind", "all_bytes")).lower()
    passive_shift  = float(passive_cfg.get("shift_pb", 0.0))  # rotate chance

    # structural extras
    ins_active_pb   = float(structuralx_cfg.get("insert_active_pb", 0.0))
    del_inactive_pb = float(structuralx_cfg.get("delete_inactive_pb", 0.0))
    dup_head_pb     = float(structuralx_cfg.get("duplicate_head_pb", 0.0))

    # Map "kind" → helper
    def _apply_kind(g: int, *, kind: str) -> int:
        if kind == "bit":
            return mutate_bitflip(g, rng)
        if kind == "byte":
            return mutate_byte(g, rng)
        if kind in ("allbytes", "all_bytes", "all"):
            return mutate_allbytes(g, rng)
        # default: no-op
        return g

    def mutate(ind: List[int]) -> List[int]:
        # -------- Structural (legacy) --------
        L = len(ind)

        # INSERT
        if ins_pb > 0.0 and L < max_len and rng.random() < ins_pb:
            pos = rng.randrange(L + 1) if L > 0 else 0
            ind.insert(pos, random_gene(state_count=state_count, rng=rng))
            L += 1

        # DELETE (guard with min_len)
        if del_pb > 0.0 and L > min_len and rng.random() < del_pb:
            pos = rng.randrange(L)
            del ind[pos]
            L -= 1

        # DUPLICATE (respect max_len)
        if dup_pb > 0.0 and L > 0 and L < max_len and rng.random() < dup_pb:
            pos = rng.randrange(L)
            ind.insert(pos + 1, ind[pos])
            L += 1

        # -------- Structural (C#-inspired extras) --------
        # INSERT "active": emulate by inserting near head
        if ins_active_pb > 0.0 and L < max_len and rng.random() < ins_active_pb:
            pos = 1 if L >= 1 else 0  # insert after head when possible
            ind.insert(pos, random_gene(state_count=state_count, rng=rng))
            L += 1

        # DELETE "inactive": emulate by random delete, still keep min_len
        if del_inactive_pb > 0.0 and L > min_len and rng.random() < del_inactive_pb:
            pos = rng.randrange(L)
            del ind[pos]
            L -= 1

        # DUP HEAD (legacy C# sometimes duplicates head gene)
        if dup_head_pb > 0.0 and L > 0 and L < max_len and rng.random() < dup_head_pb:
            ind.insert(1 if L >= 1 else 0, ind[0])
            L += 1

        # Ensure length bounds after structural stage
        if L < min_len:
            # pad with fresh random genes
            need = min_len - L
            for _ in range(need):
                ind.append(random_gene(state_count=state_count, rng=rng))
            L = len(ind)
        elif L > max_len:
            # trim excess
            del ind[max_len:]
            L = max_len

        # -------- Field-level mutations --------
        # Per-gene regime depends on activity (if available)
        active_mask = getattr(ind, "active_mask", None)

        # unpack regimes with defaults
        akind = str(active_cfg.get("kind", "byte")).lower()
        afactor = float(active_cfg.get("factor", 0.10))
        ashift = float(active_cfg.get("shift_pb", 0.02))

        pkind = str(passive_cfg.get("kind", "all_bytes")).lower()
        pfactor = float(passive_cfg.get("factor", 0.50))
        pshift = float(passive_cfg.get("shift_pb", 0.10))

        def apply_kind(x: int, kind: str) -> int:
            if kind in ("bit", "bits", "bitflip"):
                return mutate_bitflip(x, rng)
            if kind in ("byte", "one_byte"):
                return mutate_byte(x, rng)
            if kind in ("all_bytes", "allbytes", "bytes"):
                return mutate_allbytes(x, rng)
            # default fallback
            return mutate_byte(x, rng)

        for i in range(len(ind)):
            g = ind[i]

            # low-level generic knobs (independent of activity)
            if bitflip_pb > 0.0 and rng.random() < bitflip_pb:
                g = mutate_bitflip(g, rng)
            if byte_pb > 0.0 and rng.random() < byte_pb:
                g = mutate_byte(g, rng)
            if allbytes_pb > 0.0 and rng.random() < allbytes_pb:
                g = mutate_allbytes(g, rng)
            if rotate_pb > 0.0 and rng.random() < rotate_pb:
                g = mutate_rotate(g, rng)
            if enum_delta_pb > 0.0 and rng.random() < enum_delta_pb:
                g = mutate_enum_delta(g, state_count=state_count, rng=rng)

            # C#-inspired regimes guided by "was active"
            was_active = bool(active_mask[i]) if (isinstance(active_mask, list) and i < len(active_mask)) else False
            if was_active:
                if afactor > 0.0 and rng.random() < afactor:
                    g = apply_kind(g, akind)
                if ashift > 0.0 and rng.random() < ashift:
                    g = mutate_rotate(g, rng)
            else:
                if pfactor > 0.0 and rng.random() < pfactor:
                    g = apply_kind(g, pkind)
                if pshift > 0.0 and rng.random() < pshift:
                    g = mutate_rotate(g, rng)

            #ind[i] = g

            # Final sanity: keep as 64-bit
            ind[i] = int(g) & _MASK64

            # Optional: keep gene decodable; if it ever becomes invalid, randomize
            if not _valid_gene(ind[i]):
                ind[i] = random_gene(state_count=state_count, rng=rng)

        return ind

    return mutate

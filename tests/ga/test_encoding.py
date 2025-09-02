# tests/ga/test_encoding.py
import random
import pytest

from guca.ga.encoding import (
    Rule, OpKind,
    encode_rule, decode_gene, sanitize_gene, random_gene
)


def test_roundtrip_encode_decode():
    rng = random.Random(123)
    state_count = 3
    for _ in range(200):
        cond = rng.randrange(state_count)
        op = OpKind.from_id(rng.randrange(len(OpKind)))
        operand = None if rng.random() < 0.3 else rng.randrange(state_count)
        flags = rng.getrandbits(8)
        r = Rule(cond_current=cond, op_kind=op, operand=operand, flags=flags)
        g = encode_rule(r, reserved=rng.getrandbits(32))  # keep deterministic reserved
        rr = decode_gene(g, state_count=state_count)
        assert rr.op_kind == r.op_kind
        assert rr.cond_current == r.cond_current
        assert rr.operand == r.operand
        assert rr.flags == r.flags


def test_sanitize_gene_maps_into_domain():
    state_count = 3
    # construct absurd fields (beyond domain); then sanitize
    # op_id=255, cond=254, oper=253 (not NONE), flags, reserved arbitrary
    gene = ((255 << 56) | (254 << 48) | (253 << 40) | (0xAB << 32) | 0x12345678)
    s = sanitize_gene(gene, state_count=state_count)
    rr = decode_gene(s, state_count=state_count)
    assert rr.op_kind in list(OpKind)
    assert 0 <= rr.cond_current < state_count
    assert rr.operand is None or (0 <= rr.operand < state_count)
    assert rr.flags == 0xAB


def test_random_gene_is_decodable_and_in_domain():
    rng = random.Random(42)
    state_count = 4
    for _ in range(100):
        g = random_gene(state_count=state_count, rng=rng)
        r = decode_gene(g, state_count=state_count)
        assert r.op_kind in list(OpKind)
        assert 0 <= r.cond_current < state_count
        assert r.operand is None or (0 <= r.operand < state_count)
        assert 0 <= r.flags <= 255

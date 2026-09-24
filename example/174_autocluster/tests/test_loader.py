"""Unit tests for loader.py (CPU-only, no GPU needed)."""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader import decode_blob, load_embeddings

DB = "/workspace/src/rs-summarizer/summaries.db"
COMPACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "summaries_compact_20260924.db")


def pack(vec):
    return struct.pack("<%df" % len(vec), *vec)


def test_decode_roundtrip():
    v = [0.5, -1.25, 3.0, 0.0]
    assert np.allclose(decode_blob(pack(v)), np.array(v, dtype=np.float32))


def test_decode_bad_blob():
    assert decode_blob(b"").shape[0] == 0
    assert decode_blob(b"\x00\x01\x02").shape[0] == 0  # not multiple of 4


def test_truncation_is_prefix():
    v = list(range(16))
    assert decode_blob(pack(v))[:8].tolist() == list(range(8))


def test_live_db_counts():
    r = load_embeddings(DB, 128)
    assert r["total_rows"] == 14355
    assert r["X"].shape == (14146, 128), r["X"].shape
    assert r["skipped"] == {"null": 209, "short": 0, "zero_norm": 0}


def test_live_db_width_3072_keeps_only_full_rows():
    r = load_embeddings(DB, 3072)
    assert r["X"].shape == (12569, 3072), r["X"].shape


def test_x_norm_is_unit():
    r = load_embeddings(DB, 128)
    norms = np.linalg.norm(r["X_norm"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_compact_db_counts():
    r = load_embeddings(COMPACT, 128)
    assert r["total_rows"] == 19293
    assert r["X"].shape == (18269, 128), r["X"].shape
    assert r["skipped"] == {"null": 1024, "short": 0, "zero_norm": 0}


def test_compact_db_width_3072_keeps_only_full_rows():
    r = load_embeddings(COMPACT, 3072)
    assert r["X"].shape == (16692, 3072), r["X"].shape


def test_compact_db_x_norm_is_unit():
    r = load_embeddings(COMPACT, 128)
    norms = np.linalg.norm(r["X_norm"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

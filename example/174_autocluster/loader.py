"""Loader: embeddings from rs-summarizer SQLite (read-only) to float32 numpy.

BLOB format: little-endian float32 (see
`viz-tool/src/embedding.rs::bytes_to_embedding_truncated` in rs-summarizer,
read-only reference). Matryoshka semantics: shorter vectors are a prefix of
longer ones, so truncation to width k keeps the first k floats.
"""

import sqlite3
import struct
import sys

import numpy as np

DEFAULT_DB = "/workspace/src/rs-summarizer/summaries.db"


def decode_blob(blob: bytes) -> np.ndarray:
    """Decode a little-endian float32 BLOB to a 1-D float32 array."""
    if not blob or len(blob) % 4 != 0:
        return np.empty(0, dtype=np.float32)
    n = len(blob) // 4
    return np.array(struct.unpack("<%df" % n, blob), dtype=np.float32)


def load_embeddings(db_path=DEFAULT_DB, width=128):
    """Load embeddings truncated to `width` floats.

    Skips rows with NULL/short/zero-norm embeddings. Returns dict with
    X (N, width) float32, X_norm (L2-normalized, cosine-ready), ids, links,
    models, summaries, and skip stats.
    """
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    cols = [d[1] for d in con.execute("PRAGMA table_info(summaries)")]
    filt = " WHERE summary_done = 1" if "summary_done" in cols else ""
    rows = con.execute(
        "SELECT identifier, original_source_link, model, embedding_model,"
        " summary, embedding FROM summaries" + filt
    ).fetchall()
    con.close()

    xs, ids, links, models, summaries = [], [], [], [], []
    skipped = {"null": 0, "short": 0, "zero_norm": 0}
    for ident, link, model, emodel, summary, blob in rows:
        if blob is None:
            skipped["null"] += 1
            continue
        vec = decode_blob(blob)
        if vec.shape[0] < width:
            skipped["short"] += 1
            continue
        vec = vec[:width].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            skipped["zero_norm"] += 1
            continue
        xs.append(vec)
        ids.append(ident)
        links.append(link)
        models.append(model)
        summaries.append(summary)

    X = np.stack(xs).astype(np.float32) if xs else np.empty((0, width), np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = (X / np.maximum(norms, 1e-12)).astype(np.float32)
    return {
        "X": X, "X_norm": X_norm, "ids": ids, "links": links,
        "models": models, "summaries": summaries, "skipped": skipped,
        "total_rows": len(rows),
    }


def main():
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    r = load_embeddings(db, width)
    print("rows=%d kept=%d width=%d skipped=%s" % (
        r["total_rows"], r["X"].shape[0], width, r["skipped"]))


if __name__ == "__main__":
    main()

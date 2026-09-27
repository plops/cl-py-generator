"""Schnittmengen-Loader: unverzerrter Breitenvergleich (Review §1-Fix).

Problem im alten Stand: `load_embeddings(db, k)` verwirft alle Rows mit
weniger als k Floats. Bei k=3072 fallen 1.577 Rows weg (Compact-DB),
bei k=128/768 nicht — der Breitenvergleich lief auf verschieden großen,
verschieden „sauberen" Teilmengen (Survival Bias).

Fix: einmal auf voller Breite (3072) laden (N=16.692), danach pro
Breite k nur noch Prefix-trunkieren + neu L2-normieren. Alle Breiten
werden auf denselben Zeilen (denselben ids) evaluiert.
"""

import numpy as np

FULL_WIDTH = 3072


def load_full(db_path):
    """Lade volle 3072-D-Matrix + Metadaten (einmalig, dann trunkieren)."""
    from loader import load_embeddings

    return load_embeddings(db_path, FULL_WIDTH)


def truncate_norm(X_full, k):
    """Trunkiere auf Prefix k, liefere (X_k roh, X_k L2-normiert).

    Scoring-Raum ist der k-dim Originalraum (Kosinus ≙ L2-normiert),
    UMAP-Eingabe bleibt roh mit Kosinus-Metrik (wie bisher).
    """
    X = np.ascontiguousarray(X_full[:, :k], dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = (X / np.maximum(norms, 1e-12)).astype(np.float32)
    return X, Xn


def load_aligned(db_path, widths):
    """Lade Schnittmenge einmal, liefere pro Breite ausgerichtete Matrizen.

    Returns (base, per_width): base enthält ids/links/models/summaries der
    Schnittmenge, per_width[k] enthält X und X_norm auf denselben Rows.
    """
    base = load_full(db_path)
    per_width = {}
    for k in widths:
        X, Xn = truncate_norm(base["X"], k)
        per_width[k] = {"X": X, "X_norm": Xn}
    return base, per_width


def dedup_map(X):
    """Exakte Dubletten entfernen: liefert (uniq_idx, backmap).

    uniq_idx: sortierte Vertreter-Indizes (X_dedup = X[uniq_idx]).
    backmap: (N,)-Array; fitted_labels[backmap] projiziert Labels aus dem
    Dedup-Fit zurueck auf alle N Rows (fairer Blockvergleich auf identischen
    Rows: Dedup wirkt nur auf UMAP+HDBSCAN-Fitting, Scoring stets auf N).
    """
    _, idx, inv = np.unique(np.ascontiguousarray(X), axis=0,
                            return_index=True, return_inverse=True)
    order = np.argsort(idx)
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    return idx[order], remap[inv]

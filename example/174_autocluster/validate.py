"""Validation in ORIGINAL embedding space (prompt Strategy 1-3)."""

import numpy as np

from sweep_umap import adjusted_score, label_stats

MAX_NOISE = 0.40


def silhouette_orig(X_norm_gpu, labels):
    """Silhouette on L2-normalized ORIGINAL vectors, noise excluded."""
    import cupy as cp
    from cuml.metrics.cluster import silhouette_score

    labels = cp.asarray(np.asarray(labels))
    mask = labels != -1
    if int(cp.sum(mask)) == 0 or len(cp.unique(labels[mask])) < 2:
        return float("nan"), 0
    s = float(silhouette_score(X_norm_gpu[mask], labels[mask]))
    return s, int(cp.sum(mask))


def score_config(X_norm_gpu, labels):
    """Full prompt criterion; rejects noise>40% or <2 clusters (score None)."""
    st = label_stats(np.asarray(labels))
    if st["noise_ratio"] > MAX_NOISE or st["n_clusters"] < 2:
        return {"accepted": False, **st, "silhouette": float("nan"),
                "adjusted": float("nan")}
    s, n = silhouette_orig(X_norm_gpu, labels)
    return {"accepted": True, **st, "silhouette": s,
            "adjusted": adjusted_score(s, st["noise_ratio"]), "n_scored": n}

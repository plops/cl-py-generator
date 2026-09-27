"""DoE-Einzelexperiment: cuML UMAP + HDBSCAN + High-Dim-Scoring (GPU).

Kapselt pro (Design-Punkt × Seed) den vollen Pfad:
UMAP-Einbettung (mit Seed im Cache-Key, da nn_descent stochastisch ist)
→ HDBSCAN mit min_cluster_size UND min_samples
→ Scoring im k-dim Originalraum (Schnittmenge).
GPU-Imports bleiben lokal, damit design/data ohne GPU testbar sind.
"""

import os

import numpy as np


def cache_key(k, cfg, seed):
    md = str(cfg["min_dist"]).replace(".", "p")
    return "k%d_d%02d_nn%03d_md%s_mcs%02d_ms%02d_s%d" % (
        k, cfg["d"], cfg["n_neighbors"], md,
        cfg["min_cluster_size"], cfg["min_samples"], seed)


def run_umap_seed(X_gpu, cfg, seed):
    """cuML-UMAP mit explizitem Seed (Störgröße des Robust Designs)."""
    from cuml.manifold.umap import UMAP

    model = UMAP(
        n_components=cfg["d"], n_neighbors=cfg["n_neighbors"],
        min_dist=cfg["min_dist"], metric="cosine",
        build_algo="nn_descent", random_state=seed,
    )
    return model.fit_transform(X_gpu)


def run_hdbscan(X_red, cfg):
    """cuML-HDBSCAN mit beiden Dichte-Stellschrauben."""
    from cuml.cluster import HDBSCAN

    labels = HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        min_samples=cfg["min_samples"],
    ).fit_predict(X_red)
    return np.asarray(labels.get() if hasattr(labels, "get") else labels)


def run_one(X_gpu, Xn_gpu, k, cfg, seed, cache_dir):
    """Ein DoE-Run: (U)MAP → HDBSCAN → Score. Gibt (res, labels) zurück."""
    import cupy as cp

    from validate import score_config

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, cache_key(k, cfg, seed) + ".npy")
    if os.path.exists(path):
        Xr = cp.asarray(np.load(path))
    else:
        Xr = run_umap_seed(X_gpu, cfg, seed)
        cp.save(path, Xr)
    labels = run_hdbscan(Xr, cfg)
    res = score_config(Xn_gpu, labels)
    del Xr
    cp.get_default_memory_pool().free_all_blocks()
    return res, labels


def run_dbscan_eps(X_red_gpu, eps, min_samples=15):
    """cuML-DBSCAN bei gegebenem eps (für faire eps-Kalibrierung)."""
    from cuml.cluster import DBSCAN

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_red_gpu)
    return np.asarray(labels.get() if hasattr(labels, "get") else labels)

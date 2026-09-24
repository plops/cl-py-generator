"""DBSCAN / HDBSCAN clustering in UMAP-reduced space (cuML, GPU)."""

import argparse
import os

import numpy as np

from sweep_umap import eps_for_dim, label_stats


def cluster_dbscan(X_red, d, min_samples=15, base=0.3, step=0.02):
    from cuml.cluster import DBSCAN

    eps = eps_for_dim(d, base, step)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_red)
    return np.asarray(labels.get() if hasattr(labels, "get") else labels), eps


def cluster_hdbscan(X_red, min_cluster_size=15):
    from cuml.cluster import HDBSCAN

    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X_red)
    return np.asarray(labels.get() if hasattr(labels, "get") else labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="umap_cache")
    ap.add_argument("--min-samples", type=int, default=15)
    a = ap.parse_args()

    for f in sorted(os.listdir(a.cache)):
        if not f.endswith(".npy"):
            continue
        d = int(f.split("_")[0][1:])
        Xr = np.load(os.path.join(a.cache, f))
        import cupy as cp

        labels, eps = cluster_dbscan(cp.asarray(Xr), d, a.min_samples)
        st = label_stats(labels)
        print("%s eps=%.2f clusters=%d noise=%.1f%%" % (
            f, eps, st["n_clusters"], 100 * st["noise_ratio"]))


if __name__ == "__main__":
    main()

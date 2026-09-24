"""Pilot driver: UMAP sweep -> DBSCAN/HDBSCAN -> orig-space scoring -> CSV."""

import argparse
import csv
import os
import time

import numpy as np


def run_pilot(width=128, cache="umap_cache", out="results_pilot.csv",
              min_samples=15, min_cluster_size=15, seed=42,
              db=None):
    import cupy as cp

    from loader import load_embeddings, DEFAULT_DB
    from sweep_umap import DEFAULT_GRID, config_key, run_umap
    from cluster_search import cluster_dbscan, cluster_hdbscan
    from validate import score_config

    os.makedirs(cache, exist_ok=True)
    r = load_embeddings(db or DEFAULT_DB, width)
    X_gpu = cp.asarray(r["X"])
    Xn_gpu = cp.asarray(r["X_norm"])
    print("loaded %s" % (X_gpu.shape,), flush=True)

    rows = []
    for d in DEFAULT_GRID["n_components"]:
        for nn in DEFAULT_GRID["n_neighbors"]:
            for md in DEFAULT_GRID["min_dist"]:
                key = config_key(d, nn, md)
                path = os.path.join(cache, "k%d_" % width + key + ".npy")
                if os.path.exists(path):
                    Xr = cp.asarray(np.load(path))
                    secs = 0.0
                else:
                    t0 = time.time()
                    Xr, _ = run_umap(X_gpu, d, nn, md, seed)
                    secs = time.time() - t0
                    cp.save(path, Xr)
                lab_db, eps = cluster_dbscan(Xr, d, min_samples)
                lab_hd = cluster_hdbscan(Xr, min_cluster_size)
                for method, labels, param in (
                    ("dbscan", lab_db, "ms%d_eps%.2f" % (min_samples, eps)),
                    ("hdbscan", lab_hd, "mcs%d" % min_cluster_size),
                ):
                    res = score_config(Xn_gpu, np.asarray(
                        labels.get() if hasattr(labels, "get") else labels))
                    rows.append({"key": key, "d": d, "nn": nn, "md": md,
                                 "method": method, "param": param,
                                 "umap_secs": round(secs, 1), **{
                                     k: (round(v, 4) if isinstance(v, float)
                                         else v) for k, v in res.items()}})
                    print("%s %s clusters=%s noise=%s adj=%s" % (
                        key, method, res.get("n_clusters"),
                        res.get("noise_ratio"), res.get("adjusted")), flush=True)
                del Xr
                cp.get_default_memory_pool().free_all_blocks()
    with open(out, "w", newline="") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writerows(rows)
    print("wrote %s (%d rows)" % (out, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--cache", default="umap_cache")
    ap.add_argument("--out", default="results_pilot.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    run_pilot(a.width, a.cache, a.out, seed=a.seed, db=a.db)


if __name__ == "__main__":
    main()

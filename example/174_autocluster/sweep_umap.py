"""cuML UMAP sweep on GPU. All GPU imports are local so the pure helpers
stay importable/testable without cuml."""

import argparse
import os
import time

DEFAULT_GRID = {
    "n_components": [2, 4, 8, 12, 16],
    "n_neighbors": [15, 30],
    "min_dist": [0.0, 0.1],
}


def config_key(d, nn, md):
    return "d%02d_nn%03d_md%s" % (d, nn, str(md).replace(".", "p"))


def eps_for_dim(d, base=0.3, step=0.02):
    """DBSCAN eps scaled per UMAP dimension (prompt §Strategy 1)."""
    return base + d * step


def label_stats(labels):
    """Pure-numpy cluster stats. labels: 1-D int array, -1 = noise."""
    import numpy as np

    labels = np.asarray(labels)
    noise_ratio = float(np.mean(labels == -1))
    valid = labels[labels != -1]
    n_clusters = int(len(np.unique(valid))) if valid.size else 0
    return {"n_clusters": n_clusters, "noise_ratio": noise_ratio}


def adjusted_score(silhouette, noise_ratio):
    """Prompt criterion: Silhouette_orig x (1 - noise)."""
    return float(silhouette) * (1.0 - float(noise_ratio))


def run_umap(X_gpu, d, n_neighbors=30, min_dist=0.0, random_state=42):
    from cuml.manifold.umap import UMAP

    model = UMAP(
        n_components=d, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", build_algo="nn_descent", random_state=random_state,
    )
    t0 = time.time()
    out = model.fit_transform(X_gpu)
    return out, time.time() - t0


def sweep(X_gpu, grid=None, cache_dir="umap_cache", random_state=42):
    import cupy as cp

    grid = grid or DEFAULT_GRID
    os.makedirs(cache_dir, exist_ok=True)
    results = {}
    for d in grid["n_components"]:
        for nn in grid["n_neighbors"]:
            for md in grid["min_dist"]:
                key = config_key(d, nn, md)
                path = os.path.join(cache_dir, key + ".npy")
                if os.path.exists(path):
                    results[key] = {"path": path, "cached": True, "secs": 0.0}
                    continue
                Xr, secs = run_umap(X_gpu, d, nn, md, random_state)
                cp.save(path, Xr)
                del Xr
                results[key] = {"path": path, "cached": False, "secs": secs}
                print("%s done in %.1fs" % (key, secs), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--cache", default="umap_cache")
    a = ap.parse_args()

    import cupy as cp

    from loader import load_embeddings

    r = load_embeddings(width=a.width)
    X_gpu = cp.asarray(r["X"])
    print("loaded %s" % (X_gpu.shape,))
    sweep(X_gpu, cache_dir=a.cache)


if __name__ == "__main__":
    main()

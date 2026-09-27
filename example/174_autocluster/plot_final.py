"""Final 2D diagrams of the best clusters (prompt requirement).

Clusters with the winning native-d setup; projects to 2D with the same
UMAP hyper-parameters for the scatter. Noise points plotted grey.
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--nn", type=int, default=30)
    ap.add_argument("--md", type=float, default=0.0)
    ap.add_argument("--method", default="hdbscan", choices=["dbscan", "hdbscan"])
    ap.add_argument("--db", default=None)
    ap.add_argument("--outdir", default="plots")
    a = ap.parse_args()

    import cupy as cp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from loader import load_embeddings
    from sweep_umap import run_umap
    from cluster_search import cluster_dbscan, cluster_hdbscan

    os.makedirs(a.outdir, exist_ok=True)
    from loader import DEFAULT_DB
    r = load_embeddings(a.db or DEFAULT_DB, a.width)
    X_gpu = cp.asarray(r["X"])

    Xr, _ = run_umap(X_gpu, a.d, a.nn, a.md)  # native-d for clustering
    X2, _ = run_umap(X_gpu, 2, a.nn, a.md)    # 2D for plotting
    if a.method == "dbscan":
        labels, _ = cluster_dbscan(Xr, a.d)
    else:
        labels = cluster_hdbscan(Xr)
    labels = np.asarray(labels)
    pts = cp.asnumpy(X2)

    fig, ax = plt.subplots(figsize=(10, 8))
    noise = labels == -1
    ax.scatter(pts[noise, 0], pts[noise, 1], s=2, c="lightgrey", label="noise")
    # Review-Fix: bei >167 Clustern keine 168-zeilige Legende mehr —
    # nur die 12 größten Cluster bekommen Labels, der Rest bleibt lesbar.
    clusters = sorted(set(labels) - {-1},
                      key=lambda c: (labels == c).sum(), reverse=True)
    for c in clusters[:12]:
        m = labels == c
        ax.scatter(pts[m, 0], pts[m, 1], s=4, label="c%d (n=%d)" % (c, m.sum()))
    rest_n = sum((labels == c).sum() for c in clusters[12:])
    if clusters[12:]:
        m_rest = np.isin(labels, clusters[12:])
        ax.scatter(pts[m_rest, 0], pts[m_rest, 1], s=2, c="tab:gray",
                   label="weitere %d Cluster (n=%d)" % (
                       len(clusters[12:]), rest_n))
    ax.legend(markerscale=3, fontsize="small")
    ax.set_title("k=%d d=%d nn=%d md=%s %s" % (a.width, a.d, a.nn, a.md, a.method))
    path = os.path.join(a.outdir, "best_clusters.png")
    fig.savefig(path, dpi=150)
    print("wrote %s clusters=%d noise=%.1f%%" % (
        path, len(set(labels) - {-1}), 100 * noise.mean()))

    np.savetxt(os.path.join(a.outdir, "labels.csv"),
               np.column_stack([r["ids"], labels]), fmt="%d",
               header="identifier,cluster", comments="")


if __name__ == "__main__":
    main()

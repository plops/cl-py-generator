"""face_db_layout.py — Rendert das 2D-UMAP-Layout der EXPORTIERTEN DB mit
den empfohlenen Parametern und vergleicht "alle Punkte" vs. "nur Personen mit
>=min_ex Exemplaren". Zeigt, dass der zentrale Blob von den Singletons kommt,
nicht von falschen Embeddings.

Nutzung: python face_db_layout.py --export /pfad/export --min-ex 2
"""

import argparse
import os

import numpy as np


def load(export_dir):
    import json

    with open(os.path.join(export_dir, "meta.json")) as f:
        meta = json.load(f)
    n, dim = meta["n"], meta["dim"]
    emb = np.fromfile(os.path.join(export_dir, "emb.f32"), dtype="<f4").reshape(n, dim)
    labels = np.fromfile(os.path.join(export_dir, "labels.i32"), dtype="<i4")
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return emb.astype(np.float32), labels


def umap2d(X, nn, md):
    import cupy as cp
    from cuml.manifold.umap import UMAP

    Y = UMAP(n_components=2, n_neighbors=nn, min_dist=md, metric="cosine",
             build_algo="nn_descent", random_state=42).fit_transform(cp.asarray(X))
    return Y.get()


def hdb(Y, mcs):
    import cupy as cp
    from cuml.cluster.hdbscan import HDBSCAN

    lab = HDBSCAN(min_cluster_size=mcs, min_samples=mcs,
                  metric="euclidean").fit_predict(cp.asarray(Y))
    return lab.get() if hasattr(lab, "get") else np.asarray(lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="export")
    ap.add_argument("--min-ex", type=int, default=2)
    ap.add_argument("--nn", type=int, default=15)
    ap.add_argument("--md", type=float, default=0.0)
    ap.add_argument("--mcs", type=int, default=3)
    ap.add_argument("--plotdir", default="plots_face")
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, labels = load(a.export)
    os.makedirs(a.plotdir, exist_ok=True)

    uniq, counts = np.unique(labels, return_counts=True)
    keep_ids = set(uniq[counts >= a.min_ex].tolist())
    mask = np.array([lb in keep_ids for lb in labels])
    print("alle=%d  mit>=%d Exemplaren=%d (%d Personen)"
          % (len(X), a.min_ex, mask.sum(), len(keep_ids)))

    Yall = umap2d(X, a.nn, a.md)
    Xs, ls = X[mask], labels[mask]
    Ysub = umap2d(Xs, a.nn, a.md)
    lab_sub = hdb(Ysub, a.mcs)
    valid = lab_sub != -1
    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(ls[valid], lab_sub[valid]) if valid.sum() > 1 else 0.0

    fig, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=100)
    ax[0].scatter(Yall[:, 0], Yall[:, 1], s=3, c=labels, cmap="tab20", alpha=0.5)
    ax[0].set_title("ALLE %d Punkte (47%% Singletons) -> zentraler Blob" % len(X))
    ax[1].scatter(Ysub[:, 0], Ysub[:, 1], s=5, c=ls, cmap="tab20", alpha=0.6)
    ax[1].set_title("nur Personen mit >=%d Exemplaren (%d Pkt, ARI=%.3f)"
                    % (a.min_ex, mask.sum(), ari))
    p = os.path.join(a.plotdir, "db_singleton_effect.png")
    fig.savefig(p, bbox_inches="tight")
    print("Plot: %s" % p)
    print("Cluster(sub)=%d Rauschen=%.1f%% ARI=%.3f"
          % (len(set(lab_sub[valid])), 100 * (1 - valid.mean()), ari))


if __name__ == "__main__":
    main()

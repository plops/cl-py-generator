"""face_sweep.py — cuML-UMAP/HDBSCAN-Parametersuche fuer die Gesichts-Embeddings.

Ziel: gute Parameter finden, damit die 2D-Darstellung im Rust-`latent_viz`
tatsaechlich Struktur zeigt (gleiche Person = zusammen, verschiedene = getrennt).

Wir haben Ground-Truth-Personen-IDs, also bewerten wir direkt:
  * trust2d  — Trustworthiness der 2D-Einbettung ggü. dem 512D-Raum
               (bewahrt die Nachbarschaft? Kernmass fuer "sieht die 2D-Karte
               die echte Struktur?").
  * knn2d    — Anteil, mit dem der naechste 2D-Nachbar dieselbe Person ist
               (nur ueber Nicht-Singletons; misst ob gleiche Person zusammen
               landet).
  * ARI/AMI  — HDBSCAN-Cluster (im d-D-UMAP-Raum) gegen Personen-Labels.

GPU via cuML. Ergebnis-Tabelle + CSV + bestes 2D-Layout als PNG.

Nutzung:
    python face_sweep.py --export /pfad/export --plotdir plots_face
"""

import argparse
import itertools
import json
import os
import time

import numpy as np


def load_export(export_dir):
    with open(os.path.join(export_dir, "meta.json")) as f:
        meta = json.load(f)
    n, dim = meta["n"], meta["dim"]
    emb = np.fromfile(os.path.join(export_dir, "emb.f32"), dtype="<f4").reshape(n, dim)
    labels = np.fromfile(os.path.join(export_dir, "labels.i32"), dtype="<i4")
    # L2-normieren (Cosine-ready), auch wenn schon normiert.
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return emb.astype(np.float32), labels, meta


def umap_gpu(X, d, n_neighbors, min_dist, seed=42):
    from cuml.manifold.umap import UMAP

    m = UMAP(
        n_components=d,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        build_algo="nn_descent",
        random_state=seed,
    )
    out = m.fit_transform(X)
    return out.get() if hasattr(out, "get") else np.asarray(out)


def hdbscan_gpu(Xr, min_cluster_size, min_samples):
    from cuml.cluster.hdbscan import HDBSCAN

    m = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
    )
    out = m.fit_predict(Xr)
    return out.get() if hasattr(out, "get") else np.asarray(out)


def knn_same_person_2d(Y, labels, k=1):
    """Anteil Punkte, deren k naechste 2D-Nachbarn dieselbe Person haben.

    Nur Punkte werten, deren Person >=2 Exemplare hat (Singletons koennen
    per Definition keinen Same-Person-Nachbarn haben)."""
    from sklearn.neighbors import NearestNeighbors

    uniq, counts = np.unique(labels, return_counts=True)
    multi = set(uniq[counts >= 2].tolist())
    mask = np.array([lb in multi for lb in labels])
    if mask.sum() == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    _, idx = nn.kneighbors(Y)
    hits = 0
    tot = 0
    for i in np.where(mask)[0]:
        neigh = idx[i, 1 : k + 1]
        hits += int((labels[neigh] == labels[i]).any())
        tot += 1
    return hits / max(tot, 1)


def trustworthiness_2d(X, Y, n_neighbors=10, sample=3000, seed=0):
    """Trustworthiness (sklearn) auf einer Stichprobe (O(m^2))."""
    from sklearn.manifold import trustworthiness

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > sample:
        sel = rng.choice(n, size=sample, replace=False)
        Xs, Ys = X[sel], Y[sel]
    else:
        Xs, Ys = X, Y
    return float(trustworthiness(Xs, Ys, n_neighbors=n_neighbors, metric="cosine"))


def cluster_scores(labels_pred, labels_true):
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
    )

    noise = float(np.mean(labels_pred == -1))
    valid = labels_pred != -1
    n_clusters = int(len(np.unique(labels_pred[valid]))) if valid.any() else 0
    if valid.sum() >= 2:
        ari = float(adjusted_rand_score(labels_true[valid], labels_pred[valid]))
        ami = float(adjusted_mutual_info_score(labels_true[valid], labels_pred[valid]))
    else:
        ari = ami = 0.0
    return {"noise": noise, "n_clusters": n_clusters, "ari": ari, "ami": ami}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="export")
    ap.add_argument("--plotdir", default="plots_face")
    ap.add_argument("--csv", default="results_face.csv")
    ap.add_argument("--nn", default="15,30,50,80")
    ap.add_argument("--md", default="0.0,0.1,0.25")
    ap.add_argument("--dims", default="2,8,16")
    ap.add_argument("--mcs", default="3,5")
    args = ap.parse_args()

    import cupy as cp

    X, labels, meta = load_export(args.export)
    print("geladen: %s" % meta)
    Xg = cp.asarray(X)
    os.makedirs(args.plotdir, exist_ok=True)

    nns = [int(v) for v in args.nn.split(",")]
    mds = [float(v) for v in args.md.split(",")]
    dims = [int(v) for v in args.dims.split(",")]
    mcss = [int(v) for v in args.mcs.split(",")]

    rows = []
    best = None
    best_Y2 = None
    # 2D-Layout haengt nur an (nn, md) ab; d/mcs beeinflussen nur das Clustering.
    for nn, md in itertools.product(nns, mds):
        t0 = time.time()
        Y2 = umap_gpu(Xg, 2, nn, md)
        secs2 = time.time() - t0
        trust = trustworthiness_2d(X, Y2)
        knn = knn_same_person_2d(Y2, labels, k=1)
        print(
            "nn=%3d md=%.2f d=2 : trust=%.4f knn_same=%.4f (%.1fs)"
            % (nn, md, trust, knn, secs2),
            flush=True,
        )
        # Clustering pro d/mcs im d-D-UMAP-Raum.
        for d in dims:
            Yd = Y2 if d == 2 else umap_gpu(Xg, d, nn, md)
            for mcs in mcss:
                lab = hdbscan_gpu(Yd, mcs, mcs)
                sc = cluster_scores(lab, labels)
                row = {
                    "nn": nn, "md": md, "d": d, "mcs": mcs,
                    "trust2d": trust, "knn_same2d": knn,
                    **sc,
                }
                rows.append(row)
                # Guete fuer die 2D-Darstellung: Trust + Same-Person-kNN.
                viz_score = 0.5 * trust + 0.5 * (knn if knn == knn else 0.0)
                row["viz_score"] = viz_score
                if best is None or viz_score > best["viz_score"] or (
                    viz_score == best["viz_score"] and row["ari"] > best["ari"]
                ):
                    best = row
                    best_Y2 = Y2
                print(
                    "    d=%2d mcs=%d : clusters=%4d noise=%.1f%% ARI=%.3f AMI=%.3f"
                    % (d, mcs, sc["n_clusters"], sc["noise"] * 100,
                       sc["ari"], sc["ami"]),
                    flush=True,
                )

    # CSV.
    import csv

    keys = ["nn", "md", "d", "mcs", "trust2d", "knn_same2d",
            "n_clusters", "noise", "ari", "ami", "viz_score"]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print("\nCSV: %s (%d Zeilen)" % (args.csv, len(rows)))

    print("\n== BESTE VIZ-KONFIG (2D-Karte) ==")
    print(json.dumps({k: best[k] for k in keys}, indent=2))

    # Plot des besten 2D-Layouts, gefaerbt nach Cluster.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lab_best = hdbscan_gpu(best_Y2, best["mcs"], best["mcs"]) if best["d"] == 2 \
            else None
        fig, ax = plt.subplots(figsize=(11, 11), dpi=110)
        if lab_best is not None:
            c = lab_best.astype(float)
            c[c < 0] = np.nan
            ax.scatter(best_Y2[:, 0], best_Y2[:, 1], s=3, c=c, cmap="tab20",
                       alpha=0.6)
        else:
            ax.scatter(best_Y2[:, 0], best_Y2[:, 1], s=3, alpha=0.6)
        ax.set_title("best 2D UMAP nn=%d md=%.2f trust=%.3f knn_same=%.3f"
                     % (best["nn"], best["md"], best["trust2d"],
                        best["knn_same2d"]))
        p = os.path.join(args.plotdir, "best_2d.png")
        fig.savefig(p, bbox_inches="tight")
        print("Plot: %s" % p)
    except Exception as e:  # noqa: BLE001
        print("Plot uebersprungen: %s" % e)


if __name__ == "__main__":
    main()

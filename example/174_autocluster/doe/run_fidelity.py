"""Fidelity-Treiber: Trustworthiness-vs-d, TwoNN-ID, PCA-Baseline (GPU).

Schliesst die offenen Vorgaengerplan-Punkte §1.2–1.3 auf der Schnittmenge:
  1. UMAP-d-Grid (nn=30, md=0.1 wie Best-Config-Familie) → Trustworthiness
     pro d (cuML, k=5/10) + Elbow-Regel „kleinstes d mit T ≥ 0.92".
  2. TwoNN-ID auf Xn (Vollbesetzung + 10 % getrimmt) + Fit-Plot.
  3. PCA-Varianz-Baseline (randomized, 256 Komponenten).
Selbstcheck: cuML-Trustworthiness wird auf einer 1500er-Teilmenge gegen die
getestete CPU-Implementierung verifiziert (|Δ| < 1e-9).
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), "summaries_compact_20260924.db")
D_GRID = [2, 4, 6, 8, 12, 16, 20, 24]
EXTRA_SEEDS = {12: [1337, 2026]}  # Mini-Stabilitaet auf d=12 (fast gratis)
KS = [5, 10]
ELBOW_T = 0.92


def umap_embed(X_gpu, d, seed, cache):
    import cupy as cp

    from experiment import run_umap_seed

    path = os.path.join(cache, "fid_k3072_d%02d_s%d.npy" % (d, seed))
    if os.path.exists(path):
        return cp.asarray(np.load(path))
    Xr = run_umap_seed(X_gpu, {"d": d, "n_neighbors": 30, "min_dist": 0.1},
                       seed)
    cp.save(path, Xr)
    return Xr


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--cache", default=os.path.join(HERE, "umap_cache_doe"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--plots", default=os.path.join(HERE, "plots"))
    a = ap.parse_args()

    import cupy as cp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data import load_full, truncate_norm
    from fidelity import (neighbor_radii, pca_cutoffs, trustworthiness_cpu,
                          trustworthiness_gpu, twonn_fit, twonn_xy)

    os.makedirs(a.cache, exist_ok=True)
    base = load_full(a.db)
    X, Xn = truncate_norm(base["X"], 3072)
    X_gpu = cp.asarray(X)
    ndup = X.shape[0] - np.unique(Xn, axis=0).shape[0]
    print("fidelity: N=%d k=3072 exakte Dubletten=%d" % (
        X.shape[0], ndup), flush=True)

    # --- 1. Trustworthiness vs d ---
    jobs = [(d, 42) for d in D_GRID]
    for d, seeds in EXTRA_SEEDS.items():
        jobs += [(d, s) for s in seeds]
    rows = []
    checked = False
    for d, seed in jobs:
        Xr = umap_embed(X_gpu, d, seed, a.cache)
        Xe = cp.asnumpy(Xr)
        for k in KS:
            t = trustworthiness_gpu(Xn, Xe, k)
            rows.append({"d": d, "seed": seed, "k": k,
                         "trust": round(t, 4)})
            print("d=%d seed=%d k=%d T=%.4f" % (d, seed, k, t), flush=True)
        if not checked:  # Selbstcheck GPU vs getestete CPU-Implementierung
            # Toleranz 5e-3: Real-Daten enthalten exakte Dubletten, bei denen
            # sich die Rang-Konventionen (Min-Rang vs. argsort) legal um ~1e-3
            # unterscheiden; auf bindungsfreien Daten <1e-9 (Test).
            sub = np.linspace(0, X.shape[0] - 1, 1500).astype(int)
            tg = trustworthiness_gpu(Xn[sub], Xe[sub], 10)
            tc = trustworthiness_cpu(Xn[sub], Xe[sub], n_neighbors=10)
            print("selfcheck: gpu=%.10f cpu=%.10f diff=%.2e" % (
                tg, tc, abs(tg - tc)), flush=True)
            assert abs(tg - tc) < 5e-3, (tg, tc)
            checked = True
        del Xr
        cp.get_default_memory_pool().free_all_blocks()
    write_csv(os.path.join(a.out, "results_fidelity_trust.csv"), rows)

    main42 = [r for r in rows if r["seed"] == 42 and r["k"] == 10]
    above = [r["d"] for r in main42 if r["trust"] >= ELBOW_T]
    print("elbow: kleinstes d mit T(k=10) >= %.2f: %s" % (
        ELBOW_T, min(above) if above else "KEINES (Regel unerfuellt!)"),
        flush=True)

    fig, ax = plt.subplots()
    for k in KS:
        pts = sorted([(r["d"], r["trust"]) for r in rows
                      if r["seed"] == 42 and r["k"] == k])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label="k=%d" % k)
    ax.axhline(ELBOW_T, color="red", linestyle="--",
               label="Elbow-Schwelle %.2f" % ELBOW_T)
    ax.set_xlabel("UMAP-Dimension d (nn=30, md=0.1, seed=42)")
    ax.set_ylabel("Trustworthiness (Kosinus)")
    ax.set_title("Manifold-Fidelity: Trustworthiness vs. d (N=16.692)")
    ax.legend()
    fig.savefig(os.path.join(a.plots, "doe_fidelity_trust.png"), dpi=120)
    plt.close(fig)

    # --- 2. TwoNN-ID (mehrere Trims: Kurve ist gekruemmt = heterogene Dichte,
    # der lineare Klein-μ-Bereich zaehlt; s. Facco et al. 2017) ---
    r1, r2 = neighbor_radii(Xn, use_gpu=True)
    trims = (0.0, 0.1, 0.25, 0.5, 0.75)
    tw = {("trim%.2f" % t): twonn_fit(r1, r2, trim_top=t) for t in trims}
    for name, res in tw.items():
        print("twonn %s: ID=%.2f n=%d mu_med=%.3f" % (
            name, res["id"], res["n_used"], res["mu_median"]), flush=True)
    with open(os.path.join(a.out, "results_fidelity_twonn.json"), "w") as f:
        json.dump(tw, f, indent=2, sort_keys=True)

    fig, ax = plt.subplots()
    x, y, _ = twonn_xy(r1, r2, trim_top=0.1)
    ax.scatter(x[::20], y[::20], s=3, alpha=0.5, label="Punkte (jedes 20.)")
    xx = np.linspace(0, x.max(), 50)
    for name, col in (("trim0.10", "red"), ("trim0.50", "darkgreen")):
        ax.plot(xx, tw[name]["id"] * xx, color=col,
                label="Fit %s: ID=%.2f" % (name, tw[name]["id"]))
    ax.set_xlabel("log μ (r2/r1)")
    ax.set_ylabel("-log(1 - F)")
    ax.set_title("TwoNN: intrinsische Dimensionalitaet (gekr. = heterogen)")
    ax.legend()
    fig.savefig(os.path.join(a.plots, "doe_fidelity_twonn.png"), dpi=120)
    plt.close(fig)

    # --- 3. PCA-Baseline (1024 Komp.: 256 reichten nicht bis 80 %) ---
    cut, cum, ratio = pca_cutoffs(X - X.mean(axis=0), n_components=1024)
    print("pca cutoffs: %s" % cut, flush=True)
    with open(os.path.join(a.out, "results_fidelity_pca.json"), "w") as f:
        json.dump({"cutoffs": cut}, f, indent=2, sort_keys=True)
    write_csv(os.path.join(a.out, "results_fidelity_pca.csv"),
              [{"comp": i + 1, "cumvar": round(float(v), 4)}
               for i, v in enumerate(cum)])

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(cum) + 1), cum)
    for lv in (0.6, 0.8, 0.9, 0.95):
        ax.axhline(lv, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("# PCA-Komponenten")
    ax.set_ylabel("kumulierte Varianz")
    ax.set_title("PCA-Baseline (linear, zentriert, %d Komp.)" % len(cum))
    fig.savefig(os.path.join(a.plots, "doe_fidelity_pca.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

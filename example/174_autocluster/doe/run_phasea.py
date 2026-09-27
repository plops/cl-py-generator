"""Phase-A-Treiber: CCD-Bestaetigung × Dedup-Block (GPU, ~206 Fits).

Runde-2-Phase-A aus followup_vorschlag_de.md: Face-centered CCD ueber
mcs[5,19] × ms[5,15] × nn[24,60] (20 Punkte, d=11/md=0.09/k=3072 fixiert —
insignifikante Faktoren am Runden-1-Optimum), gekreuzt mit Dedup{an,aus},
je 5 Seeds. Scoring stets rueckprojiziert auf die vollen N=16.692 Rows
(Dedup wirkt nur auf UMAP+HDBSCAN-Fitting → fairer Blockvergleich).
Abfluss: gepooltes RSM + ANOVA (Dedup-Haupt- + Interaktionseffekte),
Argmax je Block per Grid, Konfirmierung mit 3 frischen Seeds, 3 Plots.
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), "summaries_compact_20260924.db")

CENTER = {"min_cluster_size": 12, "min_samples": 10, "n_neighbors": 42}
HALF = {"min_cluster_size": 7, "min_samples": 5, "n_neighbors": 18}
BOUNDS = {"min_cluster_size": (5, 19), "min_samples": (5, 15),
          "n_neighbors": (24, 60)}
FIXED = {"d": 11, "min_dist": 0.09}
SEEDS = [42, 1337, 2026, 7, 99]
CONFIRM_SEEDS = [555, 9021, 271828]

RSM_FORMULA = (
    "mean_adj ~ (min_cluster_size + min_samples + n_neighbors)**2"
    " + I(min_cluster_size**2) + I(min_samples**2) + I(n_neighbors**2)"
    " + dedup + dedup:min_cluster_size + dedup:min_samples + dedup:n_neighbors"
)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)), flush=True)


def r4(v):
    return round(float(v), 4) if isinstance(v, float) else v


def run_config(X_fit_gpu, Xn_full_gpu, backmap, cfg_full, seed, cache, key):
    """UMAP+HDBSCAN auf Fit-Set, Labels rueckprojiziert, Score auf VOLL-N."""
    import cupy as cp

    from experiment import run_hdbscan, run_umap_seed
    from validate import score_config

    path = os.path.join(cache, key + ".npy")
    if os.path.exists(path):
        Xr = cp.asarray(np.load(path))
    else:
        Xr = run_umap_seed(X_fit_gpu, cfg_full, seed)
        cp.save(path, Xr)
    lab_fit = run_hdbscan(Xr, cfg_full)
    lab_full = lab_fit[backmap] if backmap is not None else lab_fit
    res = score_config(Xn_full_gpu, np.asarray(lab_full))
    del Xr
    cp.get_default_memory_pool().free_all_blocks()
    return res, np.asarray(lab_fit)


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
    import pandas as pd

    from data import dedup_map, load_full, truncate_norm
    from design import (aggregate_point, fit_response_surface,
                        generate_ccd_design, rsm_argmax, save_design)
    from statsmodels.formula.api import ols

    os.makedirs(a.cache, exist_ok=True)
    base = load_full(a.db)
    X, Xn = truncate_norm(base["X"], 3072)
    uniq, backmap = dedup_map(Xn)
    Xn_full_gpu = cp.asarray(Xn)
    fit_sets = {0: (cp.asarray(X), None, X.shape[0]),
                1: (cp.asarray(X[uniq]), backmap, len(uniq))}
    print("phasea: N_voll=%d N_dedup=%d" % (X.shape[0], len(uniq)), flush=True)

    design = generate_ccd_design(CENTER, HALF, n_center=6)
    save_design([{**p, "d": FIXED["d"], "min_dist": FIXED["min_dist"]}
                 for p in design], os.path.join(a.out, "design_phasea.csv"))

    raw, agg = [], []
    for dedup in (0, 1):
        X_fit_gpu, back, n_fit = fit_sets[dedup]
        for i, pt in enumerate(design):
            cfg = {**pt, **FIXED}
            per_seed = []
            for s in SEEDS:
                key = "pa_dd%d_mcs%02d_ms%02d_nn%03d_s%d" % (
                    dedup, pt["min_cluster_size"], pt["min_samples"],
                    pt["n_neighbors"], s)
                t0 = time.time()
                res, lab = run_config(X_fit_gpu, Xn_full_gpu, back, cfg,
                                      s, a.cache, key)
                per_seed.append((pt, res, lab))
                raw.append({"pt": i, "dedup": dedup, "seed": s, **pt,
                            "secs": round(time.time() - t0, 1),
                            **{k: r4(v) for k, v in res.items()
                               if k != "n_scored"}})
            ag = aggregate_point(per_seed)
            if ag is not None:
                agg.append({"pt": i, "dedup": dedup,
                            **{k: r4(v) for k, v in ag.items()}})
            print("dd=%d pt=%d %s" % (dedup, i, "REJECTED" if ag is None else
                  "mean=%.4f std=%.4f sn=%.2f ari=%.3f ncl=%.0f" % (
                      ag["mean_adj"], ag["std_adj"], ag["sn_ratio"],
                      ag["ari_stability"], ag["mean_clusters"])), flush=True)
    write_csv(os.path.join(a.out, "results_phasea_raw.csv"), raw)
    write_csv(os.path.join(a.out, "results_phasea_agg.csv"), agg)

    # --- RSM + ANOVA (gepoolt, Dedup als Faktor) ---
    df = pd.DataFrame(agg)
    model, anova = fit_response_surface(df, formula=RSM_FORMULA)
    print("RSM: R²=%.3f adj-R²=%.3f n=%d" % (
        model.rsquared, model.rsquared_adj, len(df)), flush=True)
    print(anova.sort_values("F", ascending=False).to_string(), flush=True)
    for term in ("dedup", "dedup:min_cluster_size", "dedup:min_samples",
                 "dedup:n_neighbors"):
        if term in anova.index:
            print("EFFEKT %s: F=%.2f p=%.4f" % (
                term, anova.loc[term, "F"], anova.loc[term, "PR(>F)"]),
                flush=True)

    # --- Argmax je Block + Konfirmierung mit frischen Seeds ---
    confirm_rows, best = [], {}
    for dedup in (0, 1):
        pred_cfg, pred_val = rsm_argmax(model, BOUNDS, {"dedup": dedup},
                                        resolution=25)
        opt = {k: int(round(v)) for k, v in pred_cfg.items()
               if k != "dedup"}
        cfg = {**opt, **FIXED}
        X_fit_gpu, back, _ = fit_sets[dedup]
        obs = []
        for s in CONFIRM_SEEDS:
            key = "pa_confirm_dd%d_mcs%02d_ms%02d_nn%03d_s%d" % (
                dedup, opt["min_cluster_size"], opt["min_samples"],
                opt["n_neighbors"], s)
            res, _ = run_config(X_fit_gpu, Xn_full_gpu, back, cfg, s,
                                a.cache, key)
            if res.get("accepted"):
                obs.append(res["adjusted"])
            confirm_rows.append({"dedup": dedup, "seed": s, **opt,
                                 **{k: r4(v) for k, v in res.items()
                                    if k != "n_scored"}})
            print("confirm dd=%d seed=%d adj=%s" % (
                dedup, s, res.get("adjusted")), flush=True)
        best["dedup%d" % dedup] = {
            "opt": opt, "predicted": round(pred_val, 4),
            "observed_mean": round(float(np.mean(obs)), 4) if obs else None,
            "observed_std": round(float(np.std(obs)), 4) if obs else None}
        print("OPTIMUM dd=%d: %s pred=%.4f obs=%s" % (
            dedup, opt, pred_val, best["dedup%d" % dedup]["observed_mean"]),
            flush=True)
    write_csv(os.path.join(a.out, "results_phasea_confirm.csv"), confirm_rows)
    best["rsquared"] = float(model.rsquared)
    best["rsquared_adj"] = float(model.rsquared_adj)
    with open(os.path.join(a.out, "best_phasea.json"), "w") as f:
        json.dump(best, f, indent=2, sort_keys=True)

    # --- Plots ---
    os.makedirs(a.plots, exist_ok=True)
    mcs_grid = np.linspace(*BOUNDS["min_cluster_size"], 50)
    fig, ax = plt.subplots()
    for dedup, col in ((0, "tab:blue"), (1, "tab:orange")):
        sub = df[df["dedup"] == dedup]
        ax.scatter(sub["min_cluster_size"] + (dedup - 0.5) * 0.3,
                   sub["mean_adj"], color=col, alpha=0.7,
                   label="beob. dd=%d" % dedup)
        gd = pd.DataFrame({"min_cluster_size": mcs_grid,
                           "min_samples": CENTER["min_samples"],
                           "n_neighbors": CENTER["n_neighbors"],
                           "dedup": dedup})
        ax.plot(mcs_grid, model.predict(gd), color=col,
                label="RSM dd=%d" % dedup)
    ax.set_xlabel("min_cluster_size (ms=10, nn=42)")
    ax.set_ylabel("mean_adj (Voll-N)")
    ax.set_title("Phase A: mcs-Schnitt je Dedup-Block")
    ax.legend()
    fig.savefig(os.path.join(a.plots, "phasea_mcs_slice.png"), dpi=120)
    plt.close(fig)

    piv = df.pivot_table(index="pt", columns="dedup",
                         values="mean_adj").dropna()
    fig, ax = plt.subplots()
    ax.scatter(piv[0], piv[1], alpha=0.8)
    lo = min(piv[0].min(), piv[1].min()) - 0.001
    hi = max(piv[0].max(), piv[1].max()) + 0.001
    ax.plot([lo, hi], [lo, hi], "r--", label="Diagonale")
    ax.set_xlabel("mean_adj ohne Dedup")
    ax.set_ylabel("mean_adj mit Dedup")
    ax.set_title("Phase A: gepaarter Dedup-Effekt (Δ=%.4f)" % float(
        (piv[1] - piv[0]).mean()))
    ax.legend()
    fig.savefig(os.path.join(a.plots, "phasea_dedup_pair.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    for dedup, col in ((0, "tab:blue"), (1, "tab:orange")):
        sub = df[df["dedup"] == dedup]
        ax.scatter(sub["min_cluster_size"], sub["mean_clusters"], color=col,
                   alpha=0.7, label="dd=%d" % dedup)
    ax.set_xlabel("min_cluster_size")
    ax.set_ylabel("mittlere Clusterzahl")
    ax.set_title("Phase A: Fragmentierung vs. mcs")
    ax.legend()
    fig.savefig(os.path.join(a.plots, "phasea_count.png"), dpi=120)
    plt.close(fig)
    print("plots ok", flush=True)


if __name__ == "__main__":
    main()

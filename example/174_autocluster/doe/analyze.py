"""DoE-Auswertung: ANOVA/RSM, robustes Optimum, Plots (CPU-only).

Liest doe/results/*.csv, fittet das Response-Surface-Modell, druckt
ANOVA-Tabelle + Modellgüte und schreibt Plots nach doe/plots.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PLOTS = os.path.join(HERE, "plots")


def load(name):
    return pd.read_csv(os.path.join(RES, name))


def report_jitter():
    df = load("results_doe_jitter.csv")
    ok = df[df["accepted"] == True]  # noqa: E712
    print("=== Jitter (Best-Config × %d Seeds) ===" % len(df))
    print("akzeptiert: %d/%d" % (len(ok), len(df)))
    if len(ok) >= 2:
        print("mean_adj=%.4f std=%.4f min=%.4f max=%.4f" % (
            ok["adjusted"].mean(), ok["adjusted"].std(),
            ok["adjusted"].min(), ok["adjusted"].max()))
        print("clusters: %s  noise: %s" % (
            ok["n_clusters"].describe()[["min", "max"]].to_dict(),
            ok["noise_ratio"].describe()[["min", "max"]].to_dict()))
    fig, ax = plt.subplots()
    ax.hist(ok["adjusted"], bins=10)
    ax.set_xlabel("adjusted Silhouette")
    ax.set_ylabel("Häufigkeit")
    ax.set_title("Seed-Jitter der Best-Config (n=%d)" % len(ok))
    fig.savefig(os.path.join(PLOTS, "doe_jitter_hist.png"), dpi=120)
    plt.close(fig)


def report_screening():
    from design import PARAM_NAMES, fit_response_surface

    agg = load("results_doe_screening_agg.csv")
    print("=== Screening: %d aggregierte Punkte ===" % len(agg))
    model, anova = fit_response_surface(agg)
    print("R²=%.3f adj-R²=%.3f (n=%d)" % (
        model.rsquared, model.rsquared_adj, len(agg)))
    print("--- ANOVA (Typ II, sortiert nach F) ---")
    print(anova.sort_values("F", ascending=False).to_string())
    print("--- Top-5 nach S/N (robust) ---")
    print(agg.sort_values("sn_ratio", ascending=False).head(5).to_string())
    print("--- Top-5 nach Mittelwert (Peak) ---")
    print(agg.sort_values("mean_adj", ascending=False).head(5).to_string())
    best_sn = agg.loc[agg["sn_ratio"].idxmax()]
    best_mean = agg.loc[agg["mean_adj"].idxmax()]
    print("ROBUST-OPTIMUM (max S/N): %s" % best_sn.to_dict())
    print("PEAK-OPTIMUM (max mean): %s" % best_mean.to_dict())
    import json

    with open(os.path.join(RES, "best_doe.json"), "w") as f:
        json.dump({
            "robust_optimum": {k: (float(v) if isinstance(v, np.floating)
                                   else v)
                               for k, v in best_sn.to_dict().items()},
            "peak_optimum": {k: (float(v) if isinstance(v, np.floating)
                                 else v)
                             for k, v in best_mean.to_dict().items()},
            "rsquared": float(model.rsquared),
            "rsquared_adj": float(model.rsquared_adj),
        }, f, indent=2, sort_keys=True)
    print("wrote %s" % os.path.join(RES, "best_doe.json"))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, p in zip(axes.flat, PARAM_NAMES):
        ax.scatter(agg[p], agg["mean_adj"], alpha=0.7)
        ax.set_xlabel(p)
        ax.set_ylabel("mean_adj")
    axes.flat[-1].scatter(agg["mean_adj"], agg["sn_ratio"], alpha=0.7)
    axes.flat[-1].set_xlabel("mean_adj")
    axes.flat[-1].set_ylabel("S/N")
    fig.suptitle("Screening: Haupteffekte + S/N vs. Mittelwert")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "doe_screening_effects.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(agg["mean_adj"], agg["ari_stability"], alpha=0.7)
    ax.set_xlabel("mean_adj")
    ax.set_ylabel("ARI-Stabilität über Seeds")
    ax.set_title("Qualität vs. Cluster-Stabilität")
    fig.savefig(os.path.join(PLOTS, "doe_stability.png"), dpi=120)
    plt.close(fig)
    return best_sn, best_mean


def report_width():
    agg = load("results_doe_width_agg.csv")
    print("=== Breiten-Ablation auf Schnittmenge ===")
    piv = agg.pivot_table(index="run_id", columns="k", values="mean_adj")
    print(piv.to_string())
    print("--- Mittel über Design-Punkte ---")
    print(agg.groupby("k")["mean_adj"].agg(["mean", "std"]).to_string())
    fig, ax = plt.subplots()
    for k in sorted(agg["k"].unique()):
        sub = agg[agg["k"] == k].sort_values("run_id")
        ax.plot(sub["run_id"], sub["mean_adj"], "o-", label="k=%d" % k)
    ax.set_xlabel("Design-Punkt")
    ax.set_ylabel("mean_adj (gleiche Rows!)")
    ax.legend()
    ax.set_title("Faire Breiten-Ablation (Schnittmenge N=16.692)")
    fig.savefig(os.path.join(PLOTS, "doe_width.png"), dpi=120)
    plt.close(fig)


def report_dbscan():
    df = load("results_doe_dbscan.csv")
    print("=== DBSCAN-eps-Kalibrierung ===")
    ok = df[df["accepted"] == True]  # noqa: E712
    print("akzeptiert: %d/%d" % (len(ok), len(df)))
    if len(ok):
        print("bester DBSCAN: %s" % ok.loc[ok["adjusted"].idxmax()].to_dict())
    fig, ax = plt.subplots()
    for d in sorted(df["d"].unique()):
        sub = df[df["d"] == d].sort_values("eps")
        ax.plot(sub["eps"], sub["adjusted"], "o-", label="d=%d" % d)
    ax.set_xlabel("eps")
    ax.set_ylabel("adjusted Silhouette")
    ax.legend()
    ax.set_title("DBSCAN: faires eps-Grid pro Dimension")
    fig.savefig(os.path.join(PLOTS, "doe_dbscan.png"), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(PLOTS, exist_ok=True)
    report_jitter()
    report_screening()
    report_width()
    report_dbscan()


if __name__ == "__main__":
    main()

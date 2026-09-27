"""DoE-Design: Faktorraum, LHS-Plan, Robustheitsmetriken, RSM/ANOVA.

Faktorraum aus review.md Teil 3 (§1 Faktorenraum). Alle Funktionen hier
sind CPU-only und ohne GPU importierbar/testbar.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

# Steuergrößen: (low, high, dtype)
PARAM_BOUNDS = {
    "d": (4, 20, int),
    "n_neighbors": (15, 60, int),
    "min_dist": (0.0, 0.25, float),
    "min_cluster_size": (10, 40, int),
    "min_samples": (5, 25, int),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())

# Störgröße (Noise Factor): GPU-/nn_descent-Jitter über Seeds
SEEDS = [42, 1337, 2026, 7, 99]

# Breiten-Faktor (kategorial), stets auf der Schnittmenge evaluiert
WIDTHS = [128, 768, 3072]

# Erweiterte Bounds des Full-Sweeps (alle Raender offen, d ab 2, mcs bis 60)
FULL_BOUNDS = {
    "d": (2, 24, int),
    "n_neighbors": (10, 80, int),
    "min_dist": (0.0, 0.4, float),
    "min_cluster_size": (5, 60, int),
    "min_samples": (3, 30, int),
}
FULL_DESIGN_SEED = 20260927


def generate_lhs_design(n_samples=48, seed=42):
    """Latin-Hypercube-Plan über den 5-D-Steuergrößenraum (DeepWiki-Verfahren).

    Vektorisierte Skalierung via qmc.scale auf die Bounds, int-Faktoren
    gerundet. Gleicher seed → identischer Plan (reproduzierbar).
    """
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), rng=seed)
    unit = sampler.random(n=n_samples)
    lo = [PARAM_BOUNDS[p][0] for p in PARAM_NAMES]
    hi = [PARAM_BOUNDS[p][1] for p in PARAM_NAMES]
    scaled = qmc.scale(unit, lo, hi)
    design = []
    for row in scaled:
        cfg = {}
        for j, name in enumerate(PARAM_NAMES):
            typ = PARAM_BOUNDS[name][2]
            cfg[name] = typ(np.round(row[j])) if typ is int else float(row[j])
        design.append(cfg)
    return design


def generate_sobol_design(n_samples=128, seed=FULL_DESIGN_SEED,
                          bounds=None):
    """Sobol-Folge-Plan ueber den 5-D-Steuergrößenraum (erweiterbar).

    n_samples muss eine Zweierpotenz sein (random_base2); scramble=True
    randomisiert die deterministische Folge reproduzierbar per seed.
    bounds: dict Name -> (lo, hi, typ), default FULL_BOUNDS.
    """
    bounds = bounds or FULL_BOUNDS
    names = list(bounds.keys())
    m = int(np.log2(n_samples))
    if 2 ** m != n_samples:
        raise ValueError("n_samples muss Zweierpotenz sein, war %r"
                         % (n_samples,))
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    unit = sampler.random_base2(m=m)
    lo = [bounds[p][0] for p in names]
    hi = [bounds[p][1] for p in names]
    scaled = qmc.scale(unit, lo, hi)
    design = []
    for row in scaled:
        cfg = {}
        for j, name in enumerate(names):
            typ = bounds[name][2]
            cfg[name] = typ(np.round(row[j])) if typ is int else float(row[j])
        design.append(cfg)
    return design


def taguchi_sn(scores):
    """Taguchi S/N-Ratio, „larger-is-better": -10·log10(mean(1/s²)).

    Belohnt hohen Mittelwert UND kleine Streuung: 0,130±0,001 schlägt
    0,132±0,015 (Review §3, Metrik 2).
    """
    s = np.asarray(scores, dtype=float)
    return float(-10.0 * np.log10(np.mean(1.0 / (s ** 2 + 1e-9))))


def pairwise_ari(labels_list):
    """Mittlerer paarweiser Adjusted Rand Index über Seeds (Stabilität).

    Noise (-1) geht als eigene Kategorie ein: ein stabiles Thema muss
    über Seeds hinweg zusammenbleiben, stabiles Noise ebenfalls.
    """
    from sklearn.metrics import adjusted_rand_score

    n = len(labels_list)
    if n < 2:
        return float("nan")
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(adjusted_rand_score(labels_list[i], labels_list[j]))
    return float(np.mean(vals))


def aggregate_point(cfgs_seed_rows):
    """Aggregierung eines Design-Punkts über Seeds (akzeptierte Runs).

    cfgs_seed_rows: Liste von (cfg-dict, res-dict, labels). Liefert eine
    Ergebniszeile mit Mittelwert, Streuung, S/N, ARI — oder None, wenn
    weniger als 2 Runs akzeptiert wurden.
    """
    cfgs, scores, counts, noises, label_list = None, [], [], [], []
    for cfg, res, labels in cfgs_seed_rows:
        cfgs = cfg
        if res.get("accepted"):
            scores.append(res["adjusted"])
            counts.append(res["n_clusters"])
            noises.append(res["noise_ratio"])
            label_list.append(np.asarray(labels))
    if len(scores) < 2:
        return None
    scores = np.array(scores)
    return {
        **cfgs,
        "n_rep": len(scores),
        "mean_adj": float(np.mean(scores)),
        "std_adj": float(np.std(scores)),
        "sn_ratio": taguchi_sn(scores),
        "mean_clusters": float(np.mean(counts)),
        "mean_noise": float(np.mean(noises)),
        "ari_stability": pairwise_ari(label_list),
    }


RESPONSE_FORMULA = (
    "mean_adj ~ (d + n_neighbors + min_dist + min_cluster_size + min_samples)**2"
    " + I(d**2) + I(n_neighbors**2) + I(min_cluster_size**2)"
)

# Full-Sweep: alle 5 quadratischen Terme (md²/ms²-Luecke aus Runde 1 zu)
FULL_RESPONSE_FORMULA = (
    "mean_adj ~ (d + n_neighbors + min_dist + min_cluster_size + min_samples)**2"
    " + I(d**2) + I(n_neighbors**2) + I(min_dist**2)"
    " + I(min_cluster_size**2) + I(min_samples**2)"
)


def fit_response_surface(df, formula=RESPONSE_FORMULA):
    """Response-Surface-Modell (quadratisch + 2-Wege-Interaktionen) + ANOVA.

    Gibt (fitted_model, anova_typ2_tabelle) zurück. Die ANOVA zeigt, ob ein
    Faktor signifikant ist (p < 0.05) oder im nn_descent-Rauschen untergeht.
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    model = ols(formula, data=df).fit()
    anova = anova_lm(model, typ=2)
    return model, anova


def generate_ccd_design(center, half_range, n_center=6):
    """Face-centered Central-Composite-Design: Wuerfel + Achsen + Zentrum.

    center/half_range: dicts Faktor -> Wert. Liefert 2^k + 2k + n_center
    Punkte (k=3, n_center=6 → 20). Int-Faktoren bleiben int, wenn center
    und half_range int sind (symmetrische Bereiche vorausgesetzt).
    """
    import itertools

    factors = list(center.keys())
    pts = []
    for signs in itertools.product([-1, 1], repeat=len(factors)):
        pts.append({f: center[f] + s * half_range[f]
                    for f, s in zip(factors, signs)})
    for f in factors:
        for s in (-1, 1):
            p = dict(center)
            p[f] = center[f] + s * half_range[f]
            pts.append(p)
    pts += [dict(center) for _ in range(n_center)]
    return pts


def rsm_argmax(model, bounds, fixed=None, resolution=21):
    """Argmax eines gefitteten RSM-OLS-Modells per dichtem Grid.

    bounds: Faktor -> (lo, hi); fixed: Faktor -> Wert (z. B. Block).
    Gibt (best dict, predicted) zurueck; Patsy-Transfos (I(), :) wertet
    model.predict selbst aus.
    """
    import itertools

    fixed = fixed or {}
    names = [f for f in bounds if f not in fixed]
    grids = [np.linspace(bounds[f][0], bounds[f][1], resolution)
             for f in names]
    df = pd.DataFrame([dict(zip(names, v)) | fixed
                       for v in itertools.product(*grids)])
    pred = model.predict(df)
    return df.loc[pred.idxmax()].to_dict(), float(pred.max())


def save_design(design, path):
    pd.DataFrame(design).to_csv(path, index=False)


def load_design(path):
    df = pd.read_csv(path)
    for name in PARAM_NAMES:
        if PARAM_BOUNDS[name][2] is int:
            df[name] = df[name].astype(int)
    return df.to_dict("records")

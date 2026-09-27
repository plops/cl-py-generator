"""Manifold-Fidelity: Trustworthiness, TwoNN-ID, PCA-Baseline (DoE-Follow-up).

Offene Punkte aus Vorgaengerplan §1.2–1.3: Wie ehrlich bildet UMAP die
3072-D-Nachbarschaften auf d-D ab (Trustworthiness-Elbow)? Auf wie vielen
Dimensionen „lebt" die Mannigfaltigkeit wirklich (TwoNN + PCA)?

Alle Funktionen sind CPU-importierbar; GPU nur in den *_gpu-Pfaden
(lokale Imports). Referenz: Venna & Kaski 2001 (Trustworthiness),
Facco et al., Sci Rep 2017 (TwoNN).
"""

import numpy as np


def embedded_knn(X_emb, k):
    """Top-k-Nachbarindizes im Einbettungsraum (self maskiert), (N, k)."""
    from sklearn.neighbors import NearestNeighbors

    X_emb = np.asarray(X_emb)
    ind = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(
        X_emb).kneighbors(return_distance=False)
    out = np.empty((X_emb.shape[0], k), dtype=np.int64)
    for i in range(X_emb.shape[0]):
        out[i] = ind[i][ind[i] != i][:k]
    return out


def trustworthiness_cpu(X_norm, X_emb, n_neighbors=10, chunk=2000):
    """Exakte Trustworthiness (CPU, gechunkt): T(k) in [0, 1].

    X_norm: (N, D) L2-normiert — Kosinus-Ranking via Matmul, daher exakt
    die Kosinus-Trustworthiness. Formel: T = 1 - 2·Σ(r-k)/(N·k·(2N-3k-1)),
    summiert über eingebettete k-Nachbarn außerhalb der Original-k-Nachbarn.

    Konvention bei Bindungen (Duplikaten): Min-Rang-Zaelung — alle
    aequidistanten Punkte teilen den besten Rang. sklearn/cuML vergeben
    stattdessen argsort-fortlaufende Raenge; auf Real-Daten mit Duplikaten
    (Schnittmenge: 1.324 exakte Dubletten!) unterscheiden sich die
    Konventionen um ~1e-3 — echte Mehrdeutigkeit, kein Bug. Auf
    bindungsfreien Daten stimmen alle drei auf <1e-9 ueberein (Test).
    """
    X = np.asarray(X_norm, dtype=np.float64)
    N = X.shape[0]
    k = int(n_neighbors)
    if not 1 <= k < N:
        raise ValueError("brauche 1 <= k < N, habe k=%d N=%d" % (k, N))
    emb = embedded_knn(X_emb, k)
    total = 0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = X[s:e] @ X.T  # Kosinus-Ähnlichkeit (normiert)
        for ii in range(e - s):
            row = sim[ii]
            top = np.argpartition(-row, k + 1)[:k + 1]
            orig = set(int(j) for j in top if j != s + ii)
            nbrs = emb[s + ii]
            for j, sj in zip(nbrs, row[nbrs]):
                if j not in orig:
                    # Rang r(i,j): # strikt aehnlicherer Punkte (self zaehlt
                    # mit, da sim(self)=1.0 das Maximum ist).
                    total += int((row > sj).sum()) - k
    return float(1.0 - 2.0 * total / (N * k * (2 * N - 3 * k - 1)))


def trustworthiness_gpu(X_norm, X_emb, n_neighbors=10):
    """Trustworthiness auf der GPU (cuML). Exakt wie CPU: auf normierten
    Vektoren ist das euklidische Ranking identisch zum Kosinus-Ranking."""
    import cupy as cp
    from cuml.metrics import trustworthiness

    return float(trustworthiness(cp.asarray(np.asarray(X_norm)),
                                 cp.asarray(np.asarray(X_emb)),
                                 n_neighbors=int(n_neighbors)))


def neighbor_radii(X, use_gpu=True):
    """1.- und 2.-Nachbar-Distanzen (self ausgeschlossen) als (r1, r2)."""
    X = np.asarray(X)
    if use_gpu:
        try:
            import cupy as cp
            from cuml.neighbors import NearestNeighbors as CuNN

            d, _ = CuNN(n_neighbors=3, metric="euclidean").fit(
                cp.asarray(X)).kneighbors(cp.asarray(X))
            d = cp.asnumpy(d)
            if d.shape == (X.shape[0], 3) and np.all(d[:, 2] >= d[:, 1]):
                return d[:, 1].astype(float), d[:, 2].astype(float)
        except Exception:
            pass
    from sklearn.neighbors import NearestNeighbors

    d = NearestNeighbors(n_neighbors=3, algorithm="brute",
                         metric="euclidean", n_jobs=-1).fit(X).kneighbors(
        X, return_distance=True)[0]
    return d[:, 1], d[:, 2]


def twonn_xy(r1, r2, trim_top=0.0):
    """TwoNN-Plotdaten: x=log μ, y=-log(1-F), μ=r2/r1 sortiert (Facco 2017).

    trim_top behaelt nur den Bruchteil kleinster μ (linearer Bereich bei
    heterogener Dichte). WICHTIG: F wird stets auf der VOLLEN Stichprobe
    gezaehlt (i/N_voll) — wuerde man F auf der getrimmten Menge neu
    normieren, explodiert die Steigung (y-Skala fix, x → 0).
    """
    m = np.asarray(r1) > 0
    mu_full = np.sort(np.asarray(r2)[m] / np.asarray(r1)[m])
    n_full = len(mu_full)
    i = np.arange(1, n_full)  # letzter Punkt: F=1 → log(0), entfällt
    x = np.log(mu_full[:-1])
    y = -np.log(1.0 - i / n_full)
    k = max(2, int(len(x) * (1.0 - trim_top)))
    return x[:k], y[:k], mu_full[:k]


def twonn_fit(r1, r2, trim_top=0.0):
    """TwoNN-ID: Steigung durch den Ursprung (kleinste Quadrate)."""
    x, y, mu = twonn_xy(r1, r2, trim_top)
    return {"id": float(x @ y / (x @ x)), "n_used": len(mu),
            "n_dropped_zero": int((np.asarray(r1) <= 0).sum()),
            "mu_mean": float(mu.mean()), "mu_median": float(np.median(mu))}


def pca_cutoffs(X, n_components=256, levels=(0.6, 0.8, 0.9, 0.95)):
    """PCA-Varianz-Baseline: # Komponenten für je kumulierte Varianz.

    Gibt (cutoffs-dict, kumulierte Varianzkurve) zurück. Lineare Referenz
    für die intrinsische Dimensionalität (TwoNN ist nichtlinear/lokal).
    """
    from sklearn.decomposition import PCA

    n_comp = min(int(n_components), min(X.shape) - 1)
    p = PCA(n_components=n_comp, svd_solver="randomized",
            random_state=42).fit(np.asarray(X))
    cum = np.cumsum(p.explained_variance_ratio_)
    return ({float(lv): int(np.searchsorted(cum, lv) + 1) for lv in levels},
            cum, p.explained_variance_ratio_)

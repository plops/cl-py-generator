"""face_validate.py — Sanity-Checks fuer die exportierten 512D-Gesichts-Embeddings.

Liest emb.f32 (n*512 LE float32 row-major) + labels.i32 (Person-IDs) aus dem
Export des Rust-`export`-Binaries und prueft, ob die Embeddings ueberhaupt
plausibel sind, BEVOR wir UMAP-Parameter tunen. Wenn hier etwas kaputt ist
(nicht normiert, degeneriert, keine Intra-/Inter-Person-Trennung), dann liegt
das Problem in der ArcFace-Inferenz — nicht an UMAP.

Ausgabe: Kennzahlen + eine klare PASS/FAIL-Einschaetzung.

Nutzung:
    python face_validate.py --export /pfad/zu/export
"""

import argparse
import json
import os

import numpy as np


def load_export(export_dir):
    with open(os.path.join(export_dir, "meta.json")) as f:
        meta = json.load(f)
    n, dim = meta["n"], meta["dim"]
    emb = np.fromfile(os.path.join(export_dir, "emb.f32"), dtype="<f4")
    assert emb.size == n * dim, "emb.f32 Groesse passt nicht zu meta (%d != %d)" % (
        emb.size,
        n * dim,
    )
    emb = emb.reshape(n, dim)
    labels = np.fromfile(os.path.join(export_dir, "labels.i32"), dtype="<i4")
    assert labels.size == n, "labels.i32 Groesse passt nicht (%d != %d)" % (
        labels.size,
        n,
    )
    return emb, labels, meta


def basic_checks(emb):
    n, dim = emb.shape
    norms = np.linalg.norm(emb, axis=1)
    n_nan = int(np.isnan(emb).any(axis=1).sum())
    n_inf = int(np.isinf(emb).any(axis=1).sum())
    n_zero = int((norms < 1e-6).sum())
    # Norm-Verteilung (ArcFace-Embeddings sollten L2-normiert sein -> ~1.0).
    print("== Basis ==")
    print("n=%d dim=%d" % (n, dim))
    print(
        "L2-Norm: min=%.4f max=%.4f mean=%.4f std=%.6f"
        % (norms.min(), norms.max(), norms.mean(), norms.std())
    )
    print("NaN-Zeilen=%d  Inf-Zeilen=%d  Null-Vektoren=%d" % (n_nan, n_inf, n_zero))
    # Wie viele exakte Duplikate (byte-identisch gerundet)?
    uniq = np.unique(np.round(emb, 4), axis=0).shape[0]
    print("distinkte Zeilen (auf 1e-4 gerundet)=%d von %d" % (uniq, n))
    # Feature-weise Statistik: sind einzelne Dims tot (immer 0)?
    col_std = emb.std(axis=0)
    dead = int((col_std < 1e-6).sum())
    print("tote Dimensionen (std<1e-6)=%d von %d" % (dead, dim))
    return {"norms": norms, "n_nan": n_nan, "n_inf": n_inf, "n_zero": n_zero,
            "uniq": uniq, "dead_dims": dead}


def similarity_separation(emb, labels, max_pairs=200000, seed=0):
    """Vergleicht Cosine-Sim von Same-Person-Paaren gegen Random-Paare.

    Kernfrage: Wenn ArcFace korrekt laeuft, muessen Embeddings desselben
    Menschen deutlich aehnlicher sein als zwei zufaellige. Sonst ist die
    Inferenz kaputt und kein UMAP-Parameter rettet das.
    """
    rng = np.random.default_rng(seed)
    # L2-normieren fuer saubere Cosine (dot).
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    x = emb / np.maximum(norms, 1e-12)

    # Same-Person-Paare sammeln (pro Person alle Paare, gedeckelt).
    order = np.argsort(labels, kind="stable")
    labs = labels[order]
    same_sims = []
    i = 0
    n = len(labs)
    while i < n:
        j = i
        while j < n and labs[j] == labs[i]:
            j += 1
        idxs = order[i:j]
        if len(idxs) >= 2:
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    same_sims.append(float(x[idxs[a]] @ x[idxs[b]]))
        i = j
    same_sims = np.array(same_sims, dtype=np.float64)

    # Random-Paare (ueberwiegend verschiedene Personen bei 2863 IDs).
    m = min(max_pairs, len(same_sims) * 4 if len(same_sims) else max_pairs)
    m = max(m, 5000)
    ia = rng.integers(0, n, size=m)
    ib = rng.integers(0, n, size=m)
    keep = labels[ia] != labels[ib]
    ia, ib = ia[keep], ib[keep]
    diff_sims = np.einsum("ij,ij->i", x[ia], x[ib]).astype(np.float64)

    print("\n== Intra- vs Inter-Person-Aehnlichkeit (Cosine) ==")
    if len(same_sims) == 0:
        print("KEINE Same-Person-Paare (jede Person hat nur 1 Exemplar?)")
    else:
        print(
            "same-person : n=%d mean=%.4f std=%.4f  p05=%.4f p50=%.4f p95=%.4f"
            % (
                len(same_sims),
                same_sims.mean(),
                same_sims.std(),
                np.percentile(same_sims, 5),
                np.percentile(same_sims, 50),
                np.percentile(same_sims, 95),
            )
        )
    print(
        "diff-person : n=%d mean=%.4f std=%.4f  p05=%.4f p50=%.4f p95=%.4f"
        % (
            len(diff_sims),
            diff_sims.mean(),
            diff_sims.std(),
            np.percentile(diff_sims, 5),
            np.percentile(diff_sims, 50),
            np.percentile(diff_sims, 95),
        )
    )
    sep = None
    if len(same_sims):
        # Cohen's d als Trennschaerfe.
        pooled = np.sqrt((same_sims.var() + diff_sims.var()) / 2.0)
        sep = (same_sims.mean() - diff_sims.mean()) / max(pooled, 1e-9)
        print("Cohen's d (Trennschaerfe same vs diff) = %.3f" % sep)
        # Wieviele diff-Paare ueberschreiten die 0.45/0.65 Re-ID-Schwellen?
        for thr in (0.45, 0.65, 0.88):
            fp = float((diff_sims >= thr).mean())
            tp = float((same_sims >= thr).mean())
            print(
                "  @thr %.2f: same>=thr=%.1f%%  diff>=thr=%.1f%%"
                % (thr, tp * 100, fp * 100)
            )
    return same_sims, diff_sims, sep


def person_size_stats(labels):
    _, counts = np.unique(labels, return_counts=True)
    print("\n== Personen-Groessen ==")
    print(
        "Personen=%d  Exemplare/Person: min=%d max=%d mean=%.2f median=%d"
        % (len(counts), counts.min(), counts.max(), counts.mean(),
           int(np.median(counts)))
    )
    singletons = int((counts == 1).sum())
    print(
        "Singletons (1 Exemplar)=%d (%.1f%%)"
        % (singletons, 100.0 * singletons / len(counts))
    )
    for c in (1, 2, 3, 5):
        print("  Personen mit >=%d Exemplaren: %d" % (c, int((counts >= c).sum())))
    return counts


def verdict(basic, sep, emb):
    print("\n== EINSCHAETZUNG ==")
    ok = True
    norms = basic["norms"]
    if abs(norms.mean() - 1.0) > 0.05 or norms.std() > 0.05:
        print("WARN: Embeddings sind NICHT sauber L2-normiert (mean %.3f std %.3f)."
              % (norms.mean(), norms.std()))
        print("      -> ArcFace-Output evtl. nicht normalisiert / falsche Postproc.")
        ok = False
    else:
        print("OK: L2-Norm ~1.0 wie erwartet.")
    if basic["n_nan"] or basic["n_inf"]:
        print("FAIL: NaN/Inf im Embedding -> Inferenz kaputt.")
        ok = False
    if basic["dead_dims"] > 10:
        print("WARN: %d tote Dimensionen -> evtl. falscher Output-Tensor."
              % basic["dead_dims"])
        ok = False
    if sep is None:
        print("WARN: keine Same-Person-Paare, Trennschaerfe nicht messbar.")
    elif sep < 1.0:
        print("FAIL: Trennschaerfe zu gering (Cohen's d=%.2f). Same-Person-"
              "Embeddings kaum aehnlicher als zufaellige -> Embeddings falsch "
              "berechnet (Alignment/Preproc/falscher Layer?)." % sep)
        ok = False
    elif sep < 2.0:
        print("GRENZWERTIG: Cohen's d=%.2f. Trennung vorhanden aber schwach." % sep)
    else:
        print("OK: klare Intra-/Inter-Person-Trennung (Cohen's d=%.2f)." % sep)
    print("\nGESAMT: %s" % ("PLAUSIBEL" if ok else "PROBLEM ERKANNT"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="export")
    a = ap.parse_args()
    emb, labels, meta = load_export(a.export)
    print("geladen aus %s: %s" % (a.export, meta))
    basic = basic_checks(emb)
    person_size_stats(labels)
    _, _, sep = similarity_separation(emb, labels)
    verdict(basic, sep, emb)


if __name__ == "__main__":
    main()

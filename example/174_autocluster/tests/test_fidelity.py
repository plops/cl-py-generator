"""Fidelity-Tests: Trustworthiness vs sklearn-Orakel, TwoNN vs bekannte ID (CPU)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "doe"))

from fidelity import (embedded_knn, neighbor_radii, pca_cutoffs,  # noqa: E402
                      trustworthiness_cpu, twonn_fit)


def normed(X):
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float64)


def test_trustworthiness_matches_sklearn_oracle():
    from sklearn.manifold import trustworthiness as sk_trust

    rng = np.random.default_rng(3)
    X = normed(rng.normal(size=(400, 20)))
    X_emb = X[:, :3] + 0.05 * rng.normal(size=(400, 3))  # verrauschte Projektion
    for k in (5, 10):
        got = trustworthiness_cpu(X, X_emb, n_neighbors=k)
        want = sk_trust(X, X_emb, n_neighbors=k, metric="cosine")
        assert abs(got - want) < 1e-9, (k, got, want)


def test_trustworthiness_perfect_embedding_is_one():
    rng = np.random.default_rng(4)
    X = normed(rng.normal(size=(200, 10)))
    assert trustworthiness_cpu(X, X.copy(), n_neighbors=5) == 1.0


def test_trustworthiness_with_duplicates_stays_finite():
    # Real-Daten enthalten exakte Dubletten: Min-Rang-Konvention liefert
    # endliche Werte in [0, 1] (sklearn/cuML duerfen um ~1e-3 abweichen).
    rng = np.random.default_rng(6)
    X = normed(rng.normal(size=(200, 10)))
    X[::20] = X[0]  # jedes 20. eine exakte Dublette
    t = trustworthiness_cpu(X, X[:, :3] + 0.01, n_neighbors=5)
    assert 0.0 <= t <= 1.0 and np.isfinite(t)


def test_trustworthiness_rejects_bad_k():
    import pytest

    X = normed(np.random.default_rng(5).normal(size=(50, 8)))
    with pytest.raises(ValueError):
        trustworthiness_cpu(X, X[:, :2], n_neighbors=50)


def test_embedded_knn_excludes_self():
    X = np.array([[0.0], [1.0], [3.0], [6.0]])
    knn = embedded_knn(X, 2)
    assert knn.shape == (4, 2)
    for i in range(4):
        assert i not in knn[i]
    assert set(knn[0]) == {1, 2}  # naechste an 0.0: 1.0, 3.0


def test_twonn_recovers_uniform_cube_id_5():
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(3000, 5))  # bekannte ID = 5
    r1, r2 = neighbor_radii(X, use_gpu=False)
    est = twonn_fit(r1, r2)["id"]
    assert abs(est - 5.0) < 1.0, est


def test_twonn_recovers_plane_id_2():
    rng = np.random.default_rng(12)
    X = rng.uniform(size=(3000, 2))  # bekannte ID = 2
    r1, r2 = neighbor_radii(X, use_gpu=False)
    est = twonn_fit(r1, r2)["id"]
    assert abs(est - 2.0) < 0.5, est


def test_twonn_smoke_on_normalized_high_dim():
    rng = np.random.default_rng(13)
    X = normed(rng.normal(size=(500, 100)))
    r1, r2 = neighbor_radii(X, use_gpu=False)
    res = twonn_fit(r1, r2, trim_top=0.1)
    # F auf Voll-N=500, Fit auf int(499*0.9)=449 kleinsten μ
    assert np.isfinite(res["id"]) and res["n_used"] == int(499 * 0.9)


def test_twonn_trim_stable_on_uniform_data():
    # Gleichmaessige Dichte → Gerade → Trim nur der μ-Skala wegen stabil
    # (faengt die F-Neunormierungs-Explosion: trim50 wuchs auf ID=47).
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(3000, 5))
    r1, r2 = neighbor_radii(X, use_gpu=False)
    full = twonn_fit(r1, r2)["id"]
    for t in (0.1, 0.25, 0.5):
        got = twonn_fit(r1, r2, trim_top=t)["id"]
        assert abs(got - full) < 1.5, (t, full, got)


def test_pca_cutoffs_on_rank_10():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(200, 10)) @ rng.normal(size=(10, 50))  # Rang 10
    cut, cum, _ = pca_cutoffs(X, n_components=49)
    assert cut[0.9] <= 11, cut
    assert cut[0.95] <= 11, cut
    assert abs(cum[9] - 1.0) < 1e-6  # 10 Komponenten ≈ 100 %

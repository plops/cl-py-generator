"""DoE-Logiktests: LHS-Design, Taguchi-S/N, ARI, Schnittmengen-Trunkierung (CPU)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "doe"))

from data import truncate_norm  # noqa: E402
from design import (PARAM_BOUNDS, aggregate_point, generate_lhs_design,  # noqa: E402
                    pairwise_ari, taguchi_sn)


def test_lhs_respects_bounds_and_types():
    d = generate_lhs_design(48, seed=42)
    assert len(d) == 48
    for cfg in d:
        for name, (lo, hi, typ) in PARAM_BOUNDS.items():
            assert lo <= cfg[name] <= hi, (name, cfg[name])
            assert isinstance(cfg[name], typ), (name, cfg[name])


def test_lhs_reproducible_with_same_seed():
    a = generate_lhs_design(20, seed=42)
    b = generate_lhs_design(20, seed=42)
    assert a == b


def test_lhs_differs_with_other_seed():
    a = generate_lhs_design(20, seed=42)
    b = generate_lhs_design(20, seed=43)
    assert a != b


def test_taguchi_prefers_robust_over_peaky():
    # Gleicher Mittelwert (0,130), ungleiche Streuung: robust gewinnt
    # per Jensen-Ungleichung (mean(1/s²) minimal bei Varianz 0).
    robust = [0.129, 0.130, 0.131]
    peaky = [0.115, 0.130, 0.145]
    assert taguchi_sn(robust) > taguchi_sn(peaky)


def test_taguchi_monotone_in_level():
    assert taguchi_sn([0.2, 0.2, 0.2]) > taguchi_sn([0.1, 0.1, 0.1])


def test_pairwise_ari_identical_is_one():
    lab = np.array([0, 0, 1, 1, -1, 2])
    assert pairwise_ari([lab, lab.copy(), lab.copy()]) == 1.0


def test_pairwise_ari_unstable_below_one():
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.array([0, 1, 0, 1, 0, 1])
    assert pairwise_ari([a, b]) < 1.0


def test_truncate_is_prefix_and_unit_norm():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 64)).astype(np.float32)
    Xr, Xn = truncate_norm(X, 16)
    assert Xr.shape == (50, 16)
    assert np.allclose(Xr, X[:, :16])
    assert np.allclose(np.linalg.norm(Xn, axis=1), 1.0, atol=1e-5)


def test_aggregate_point_needs_two_accepted():
    cfg = {"d": 8, "n_neighbors": 15, "min_dist": 0.0,
           "min_cluster_size": 15, "min_samples": 15}
    lab = np.array([0, 0, 1, 1])
    rej = {"accepted": False, "n_clusters": 0, "noise_ratio": 1.0}
    ok = {"accepted": True, "adjusted": 0.12, "n_clusters": 2,
          "noise_ratio": 0.3}
    assert aggregate_point([(cfg, rej, lab), (cfg, ok, lab)]) is None
    a = aggregate_point([(cfg, ok, lab), (cfg, ok, lab)])
    assert a["n_rep"] == 2 and abs(a["mean_adj"] - 0.12) < 1e-9
    assert a["ari_stability"] == 1.0


def test_response_surface_fits_synthetic():
    import pandas as pd

    from design import fit_response_surface

    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({
        "d": rng.integers(4, 21, n),
        "n_neighbors": rng.integers(15, 61, n),
        "min_dist": rng.uniform(0, 0.25, n),
        "min_cluster_size": rng.integers(10, 41, n),
        "min_samples": rng.integers(5, 26, n),
    })
    df["mean_adj"] = (0.1 + 0.001 * df["d"]
                      - 0.00005 * df["d"] ** 2
                      + rng.normal(0, 0.005, n))
    model, anova = fit_response_surface(df)
    assert 0.0 <= model.rsquared <= 1.0
    assert "F" in anova.columns and "PR(>F)" in anova.columns

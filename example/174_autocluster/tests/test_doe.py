"""DoE-Logiktests: LHS-Design, Taguchi-S/N, ARI, Schnittmengen-Trunkierung (CPU)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "doe"))

from data import dedup_map, truncate_norm  # noqa: E402
from design import (FULL_BOUNDS, PARAM_BOUNDS, aggregate_point,  # noqa: E402
                    generate_ccd_design, generate_lhs_design,
                    generate_sobol_design, pairwise_ari, rsm_argmax,
                    taguchi_sn)


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


def test_ccd_structure_cube_axial_center():
    c = {"mcs": 12, "ms": 10, "nn": 42}
    h = {"mcs": 7, "ms": 5, "nn": 18}
    d = generate_ccd_design(c, h, n_center=6)
    assert len(d) == 8 + 6 + 6
    assert all(set(p) == set(c) for p in d)
    assert all(isinstance(v, int) for p in d for v in p.values())
    assert sum(1 for p in d if p == c) == 6  # Zentrum 6-fach
    assert {"mcs": 5, "ms": 5, "nn": 24} in d  # Wuerfelecke - - -
    assert {"mcs": 19, "ms": 15, "nn": 60} in d  # Wuerfelecke + + +
    assert {"mcs": 5, "ms": 10, "nn": 42} in d  # Achspunkt mcs-
    for p in d:  # nur 3 Stufen je Faktor (face-centered)
        assert p["mcs"] in (5, 12, 19)
        assert p["ms"] in (5, 10, 15)
        assert p["nn"] in (24, 42, 60)


def test_dedup_map_roundtrip():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0],
                  [3.0, 4.0]])
    uniq, back = dedup_map(X)
    assert uniq.tolist() == [0, 1, 3]
    assert back.tolist() == [0, 1, 0, 2, 1]
    assert (X == X[uniq][back]).all()  # perfekte Rekonstruktion
    lab = np.array([7, 8, 9])
    assert lab[back].tolist() == [7, 8, 7, 9, 8]  # Dubletten erben


def test_dedup_map_no_dups_identity():
    X = np.arange(12, dtype=float).reshape(4, 3)
    uniq, back = dedup_map(X)
    assert uniq.tolist() == [0, 1, 2, 3]
    assert back.tolist() == [0, 1, 2, 3]


def test_rsm_argmax_finds_known_peak():
    import pandas as pd
    from statsmodels.formula.api import ols

    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.uniform(-5, 5, 300),
                       "z": rng.uniform(-5, 5, 300)})
    df["y"] = 1.0 - (df["x"] - 3.0) ** 2 - (df["z"] + 1.0) ** 2
    m = ols("y ~ x + z + I(x**2) + I(z**2) + x:z", data=df).fit()
    best, pred = rsm_argmax(m, {"x": (-5, 5), "z": (-5, 5)}, resolution=21)
    assert abs(best["x"] - 3.0) < 0.6 and abs(best["z"] + 1.0) < 0.6
    assert abs(pred - 1.0) < 0.1


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


def test_sobol_respects_full_bounds_and_types():
    d = generate_sobol_design(32, seed=20260927)
    assert len(d) == 32
    for cfg in d:
        for name, (lo, hi, typ) in FULL_BOUNDS.items():
            assert lo <= cfg[name] <= hi, (name, cfg[name])
            assert isinstance(cfg[name], typ), (name, cfg[name])


def test_sobol_reproducible_and_unique_rows():
    a = generate_sobol_design(128, seed=20260927)
    b = generate_sobol_design(128, seed=20260927)
    assert a == b
    rows = [tuple(sorted(c.items())) for c in a]
    assert len(set(rows)) == 128  # keine exakten Duplikate nach Rundung


def test_sobol_requires_power_of_two():
    import pytest

    with pytest.raises(ValueError):
        generate_sobol_design(100)


def test_full_formula_fits_synthetic():
    import pandas as pd

    from design import FULL_RESPONSE_FORMULA, fit_response_surface

    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({
        "d": rng.integers(2, 25, n),
        "n_neighbors": rng.integers(10, 81, n),
        "min_dist": rng.uniform(0, 0.4, n),
        "min_cluster_size": rng.integers(5, 61, n),
        "min_samples": rng.integers(3, 31, n),
    })
    df["mean_adj"] = (0.1 + 0.001 * df["min_cluster_size"]
                      - 0.0001 * df["min_samples"] ** 2
                      + rng.normal(0, 0.005, n))
    model, anova = fit_response_surface(df, formula=FULL_RESPONSE_FORMULA)
    assert 0.0 <= model.rsquared <= 1.0
    assert "I(min_dist ** 2)" in anova.index
    assert "I(min_samples ** 2)" in anova.index

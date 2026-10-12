"""Tests for src.analysis.prox_crit (Python port of Sooter's prox_crit_toolkit).

Three layers:
1. MATLAB parity: outputs of the ORIGINAL MATLAB code on identical inputs
   (fixtures from tests/matlab/make_reference.m).
2. Independent math checks: the formulas themselves (time-domain vs spectral
   KL rate, manifold constraints, Hessian, literal MATLAB loops, ...).
3. End-to-end: the values printed in the original tutorial (demo.mlx) for the
   ephys data (slow; deselect with  -m "not slow").

Run:  python -m pytest tests/test_prox_crit.py -v
"""

import math
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
from scipy.optimize import minimize

from src.analysis.prox_crit import (
    DbResult,
    beta_lift,
    beta_project,
    calc_db,
    calc_dbv1,
    explosive,
    get_kernw,
    get_klr_b,
    get_sigmac,
    get_spike_counts,
    matlab_colon,
    minimize_klr_b,
    mle_ar,
    run_ar,
    yw_ar,
)
from src.analysis.prox_crit.kl_rate import _inv_spectrum
from src.analysis.prox_crit.manifold import _ar_roots
from src.config.paths import YMV15_MAT

FIX = Path(__file__).parent / "fixtures"
FITS = ["YuleWalker", "MaxLikelihood"]
LOG2E = np.log2(np.e)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ref():
    """Inputs and MATLAB outputs, plus the Python results for every case."""
    inp = sio.loadmat(FIX / "prox_crit_inputs.mat")
    out = sio.loadmat(FIX / "prox_crit_reference.mat", squeeze_me=True)
    dsets = [[np.ravel(s) for s in d.ravel()] for d in inp["dsets"].ravel()]
    cases = inp["cases"].astype(int)
    py = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d, order, b, fit, err in cases:
            py.append(
                calc_db(
                    dsets[d - 1],
                    order,
                    1.0,
                    b,
                    fit_method=FITS[fit],
                    with_err_bars=bool(err),
                )
            )
    return dict(inp=inp, out=out, dsets=dsets, cases=cases, py=py)


def _cases(ref, pred):
    return [i for i, c in enumerate(ref["cases"]) if pred(*c)]


# --------------------------------------------------------------------------- #
# 1. MATLAB parity
# --------------------------------------------------------------------------- #
def test_ar_fits_match_matlab(ref):
    """kern / sigma / H of yw_ar and mle_ar (deterministic linear algebra)."""
    out = ref["out"]
    for i in _cases(ref, lambda d, o, b, f, e: o >= b // 2):
        r = ref["py"][i]
        # rtol 1e-9: max observed 3e-11 (YW, order 8: errors amplified by
        # cond(Gamma)); different BLAS/LAPACK summation order only.
        np.testing.assert_allclose(r.kern, np.ravel(out["kern"][i]), rtol=1e-9)
        np.testing.assert_allclose(r.sigma, out["sigma"][i], rtol=1e-12)
        if ref["cases"][i][3] == 1:
            np.testing.assert_allclose(r.H, out["H"][i], rtol=1e-12)


def test_calc_dbv1_matches_matlab(ref):
    K = ref["inp"]["dbv1_kerns"]
    for i, row in enumerate(K):
        kern, b = row[2 : 2 + int(row[0])], int(row[1])
        dbv1, kernc = calc_dbv1(kern, b)
        np.testing.assert_allclose(dbv1, ref["out"]["dbv1"][i], rtol=1e-12)
        np.testing.assert_allclose(
            kernc, np.ravel(ref["out"]["dbv1_kernc"][i]), rtol=0, atol=1e-11
        )


def test_db_b2_matches_matlab(ref):
    """b = 2: db, sddb, kernc agree with MATLAB.

    Tolerance db rtol 1e-6 = MATLAB integral()'s RelTol: the observed
    difference (~1.7e-7) is MATLAB's quadrature error; scipy's value agrees
    with a 30-digit mpmath evaluation to ~1e-10 (see test_quadrature_accuracy).
    """
    out = ref["out"]
    for i in _cases(ref, lambda d, o, b, f, e: b == 2):
        r = ref["py"][i]
        assert r.exit_status == out["exit_status"][i] == 0
        np.testing.assert_allclose(r.db, out["db"][i], rtol=1e-6)
        np.testing.assert_allclose(r.kernc, np.ravel(out["kernc"][i]), atol=5e-6)
        if ref["cases"][i][4]:
            np.testing.assert_allclose(np.real(r.sddb), out["sddb"][i], rtol=1e-6)
        else:
            assert np.isnan(r.sddb) and np.isnan(out["sddb"][i])


def test_db_single_point_manifold_matches_matlab(ref):
    """order == b/2: no optimization, M_b is one point; pure quadrature.

    rtol 1e-5: observed 6.6e-6 = MATLAB integral() error at the 3-fold log
    singularity (Python agrees with mpmath to ~2e-8).
    """
    (i,) = _cases(ref, lambda d, o, b, f, e: o == b // 2)
    r, out = ref["py"][i], ref["out"]
    np.testing.assert_array_equal(r.kernc, np.ravel(out["kernc"][i]))  # [3, -3, 1]
    np.testing.assert_allclose(r.db, out["db"][i], rtol=1e-5)


def test_db_higher_b_python_reaches_lower_minimum(ref):
    """b >= 4: the objective is the same, but fmincon stops early (FLAG[15]).

    (a) Python's KL rate evaluated at MATLAB's kernc reproduces MATLAB's db
        (same objective function). rtol 2e-5: at b = 6 both quadratures
        (scipy, MATLAB) deviate up to ~1e-5 from a 25-digit mpmath value.
    (b) Python's db is never above MATLAB's.
    (c) Python's minimum is confirmed by an independent optimizer
        (Nelder-Mead). rtol 2e-5: FLAG[16], finite-difference gradients of
        the ~1e-7-accurate objective let SLSQP stop up to ~1e-5 above the
        minimum at b >= 4 (MATLAB's fmincon uses the same step size).
    """
    out = ref["out"]
    for i in _cases(ref, lambda d, o, b, f, e: b >= 4 and o > b // 2):
        r, b = ref["py"][i], ref["cases"][i][2]
        mkc = np.ravel(out["kernc"][i])
        at_matlab = LOG2E * get_klr_b(r.kern, mkc[: len(mkc) - b // 2], b, "integral")
        np.testing.assert_allclose(at_matlab, out["db"][i], rtol=2e-5)  # (a)
        assert r.db <= out["db"][i] * (1 + 2e-5)  # (b)

        obj = lambda kc: get_klr_b(r.kern, kc, b, "integral")  # noqa: E731
        nm = minimize(
            obj,
            beta_project(r.kernc, b),
            method="Nelder-Mead",
            options=dict(xatol=1e-9, fatol=1e-13, maxiter=20000, maxfev=20000),
        )
        np.testing.assert_allclose(r.db, LOG2E * nm.fun, rtol=2e-5)  # (c)


def test_order_below_b_half(ref):
    """CHANGED[3]: MATLAB crashes on cm.kernc; Python returns NaN, exit 1."""
    (i,) = _cases(ref, lambda d, o, b, f, e: o < b // 2)
    assert "kernc" in ref["out"]["errmsg"][i]  # the MATLAB crash
    r = ref["py"][i]
    assert np.isnan(r.db) and np.isnan(r.kernc) and r.exit_status == 1


def test_spike_counts_match_matlab_histc(ref):
    """Bin edges (MATLAB colon) and histc counts: bit-identical."""
    out = ref["out"]
    d = sio.loadmat(YMV15_MAT, squeeze_me=True, variable_names=["st", "WAKE"])
    st = d["st"]
    windows = {
        "full": (st.min(), st.max()),
        "wake": tuple(d["WAKE"][1].astype(float)),
        "win": (st.min() + 30 * 7, st.min() + 30 * 7 + 60),
    }
    for name, (t1, t2) in windows.items():
        np.testing.assert_array_equal(
            matlab_colon(t1, 0.04, t2), np.ravel(out["bins_" + name])
        )
        np.testing.assert_array_equal(
            get_spike_counts(st, t1, t2, 0.04), np.ravel(out["counts_" + name])
        )


# --------------------------------------------------------------------------- #
# 2. Independent math checks
# --------------------------------------------------------------------------- #
def _ar_kernel_from_roots(roots):
    """kern such that P(z) = 1 - sum kern_k z^k = prod (1 - z / r)."""
    p = np.poly1d([1.0])
    for r in roots:
        p = p * np.poly1d([-1.0 / r, 1.0])
    c = p.coeffs[::-1]  # ascending powers, c[0] = 1
    return -np.real(c[1:])


def _autocov(kern, n_lags):
    """Exact autocovariances gamma_0..gamma_{n_lags} of a stable AR, sigma = 1.

    Time-domain, independent of the spectral code: solve
    gamma_k - sum_j kern_j gamma_|k-j| = delta_k0 for k = 0..n, then recurse.
    """
    n = len(kern)
    M = np.eye(n + 1)
    for k in range(n + 1):
        for j in range(1, n + 1):
            M[k, abs(k - j)] -= kern[j - 1]
    g = list(np.linalg.solve(M, np.r_[1.0, np.zeros(n)]))
    for k in range(n + 1, n_lags + 1):
        g.append(sum(kern[j - 1] * g[k - j] for j in range(1, n + 1)))
    return np.array(g[: n_lags + 1])


@pytest.mark.parametrize("b", [2, 4, 6])
def test_beta_lift_constraints_and_roots(b):
    rng = np.random.default_rng(b)
    red = rng.normal(scale=0.3, size=5)
    phic = beta_lift(red, b)
    n, lags = len(phic), np.arange(1, len(phic) + 1)
    assert n == 5 + b // 2
    np.testing.assert_allclose(phic.sum(), 1, atol=1e-12)  # P(1) = 0
    for j in range(1, b // 2):  # derivative conditions
        np.testing.assert_allclose(lags**j @ phic, 0, atol=1e-9)
    # z = 1 is a root of P of multiplicity b/2: P^(m)(1) = 0 for m < b/2
    P = np.poly1d(np.r_[-phic[::-1], 1.0])
    for m in range(b // 2):
        np.testing.assert_allclose(P.deriv(m)(1.0) if m else P(1.0), 0, atol=1e-8)
    np.testing.assert_array_equal(beta_project(phic, b), red)


@pytest.mark.parametrize("b", [2, 4, 6])
def test_calc_dbv1_is_orthogonal_projection(b):
    rng = np.random.default_rng(10 + b)
    kern = rng.normal(scale=0.3, size=7)
    dbv1, kernc = calc_dbv1(kern, b)
    lags = np.arange(1, 8)
    np.testing.assert_allclose(kernc.sum(), 1, atol=1e-12)
    for j in range(1, b // 2):
        np.testing.assert_allclose(lags**j @ kernc, 0, atol=1e-9)
    # residual orthogonal to every direction within M_b
    red = beta_project(kernc, b)
    for k in range(len(red)):
        e = red.copy()
        e[k] += 1.0
        direction = beta_lift(e, b) - kernc
        np.testing.assert_allclose((kern - kernc) @ direction, 0, atol=1e-10)
    np.testing.assert_allclose(dbv1, np.linalg.norm(kern - kernc), rtol=1e-14)
    # a kernel already on M_b is its own projection
    np.testing.assert_allclose(calc_dbv1(kernc, b)[1], kernc, atol=1e-12)


def test_get_kernw_and_spectrum():
    rng = np.random.default_rng(0)
    kern = rng.normal(size=6)
    w = np.linspace(-np.pi, np.pi, 37)
    direct = sum(kern[t - 1] * np.exp(-1j * w * t) for t in range(1, 7))  # MATLAB loop
    np.testing.assert_allclose(get_kernw(kern, w), direct, rtol=1e-13, atol=1e-14)
    # FFT check on the DFT grid w_m = 2 pi m / N (lag t at FFT index t)
    N = 16
    padded = np.zeros(N)
    padded[1:7] = kern
    np.testing.assert_allclose(
        get_kernw(kern, 2 * np.pi * np.arange(N) / N), np.fft.fft(padded), atol=1e-13
    )
    # AR(1) analytic inverse spectrum: |1 - a e^{-iw}|^2 = 1 - 2 a cos w + a^2
    a = 0.7
    g = _inv_spectrum([a])
    np.testing.assert_allclose(g(w), 1 - 2 * a * np.cos(w) + a**2, rtol=1e-14)
    # scalar fast path (used by quad) == vectorized path
    gk = _inv_spectrum(kern)
    np.testing.assert_allclose([gk(float(v)) for v in w], gk(w), rtol=1e-13)


@pytest.mark.parametrize("b, rtol", [(2, 1e-9), (4, 1e-6)])
def test_kl_rate_equals_time_domain_formula(b, rtol):
    """Spectral KL rate == conditional-likelihood (time-domain) KL rate.

    For the fitted model P (sigma = 1, stable) and critical model Q (kernc,
    sigma_c), with e = [1, -kernc] and Gamma_P the autocovariance matrix of P:
        sigma_c^2 = e' Gamma_P e
        klr       = 1/2 [log sigma_c^2 + e' Gamma_P e / sigma_c^2 - 1]
                  = 1/2 log(e' Gamma_P e)
    This also confirms FLAG[2] (sigma_c is relative to sigma = 1). The b = 4
    case also exercises CHANGED[4] (g_c underflow next to w = 0); its rtol
    1e-6 is the integral's RelTol (observed 1.4e-7), b = 2 is near exact.
    (Valid only for a non-explosive kernc, since it uses int log g_c = 0.)
    """
    kern = _ar_kernel_from_roots([1.6, -2.0, 2.5 + 1j, 2.5 - 1j])
    kc = beta_lift(np.array([0.9, 0.2, -0.3][: 4 - b // 2]), b)
    g = _autocov(kern, len(kc))
    Gamma = np.array(
        [[g[abs(i - j)] for j in range(len(kc) + 1)] for i in range(len(kc) + 1)]
    )
    e = np.r_[1.0, -kc]
    v = e @ Gamma @ e
    np.testing.assert_allclose(get_sigmac(kern, kc) ** 2, v, rtol=1e-9)
    klr = get_klr_b(kern, beta_project(kc, b), b, "integral")
    np.testing.assert_allclose(klr, 0.5 * np.log(v), rtol=rtol)


def test_quadrature_accuracy():
    """scipy quad vs a 30-digit mpmath reference for the b = 2 KL rate."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    kern = [0.8, 0.2, -0.1]
    red = [0.84, 0.23]
    kc = beta_lift(red, 2)

    def g(w, k):
        return (
            abs(1 - mp.fsum(k[t] * mp.e ** (-1j * w * (t + 1)) for t in range(len(k))))
            ** 2
        )

    s2 = mp.quad(lambda w: g(w, kc) / g(w, kern), [-mp.pi, 0, mp.pi]) / (2 * mp.pi)
    ref = mp.quad(
        lambda w: (
            mp.log(s2 * g(w, kern) / g(w, kc)) + g(w, kc) / (s2 * g(w, kern)) - 1
        )
        / (4 * mp.pi),
        [-mp.pi, 0, mp.pi],
    )
    np.testing.assert_allclose(
        get_klr_b(kern, red, 2, "integral"), float(ref), rtol=1e-9
    )


def test_mle_ar_against_literal_loop_and_hessian():
    rng = np.random.default_rng(1)
    segs = [run_ar([0.5, -0.2, 0.1], 1, n, rng=rng) for n in (400, 250, 2)]
    order = 3
    kern, sigma, H = mle_ar(segs, order)

    # literal translation of the MATLAB double loop for the design matrix
    rows, Y = [], []
    for s in segs:
        for j in range(1, len(s) + 1):  # MATLAB j = 1..length
            hist = s[max(1, j - order) - 1 : j - 1]
            rows.append(np.r_[hist[::-1], np.zeros(order - len(hist))])
            Y.append(s[j - 1])
    X, Y = np.array(rows), np.array(Y)
    np.testing.assert_allclose(kern, np.linalg.lstsq(X, Y, rcond=None)[0], rtol=1e-10)
    np.testing.assert_allclose(sigma, np.sqrt(np.mean((Y - X @ kern) ** 2)), rtol=1e-12)

    # H = Hessian of the profile log-likelihood l(k) = -T/2 log(sum(res^2))
    T = len(Y)

    def ll(k):
        return -T / 2 * np.log(np.sum((Y - X @ k) ** 2))

    h = 1e-4
    Hfd = np.zeros((order, order))
    E = np.eye(order) * h
    for i in range(order):
        for j in range(order):
            Hfd[i, j] = (
                ll(kern + E[i] + E[j])
                - ll(kern + E[i] - E[j])
                - ll(kern - E[i] + E[j])
                + ll(kern - E[i] - E[j])
            ) / (4 * h * h)
    np.testing.assert_allclose(H, Hfd, rtol=1e-5)


def test_yw_ar_against_literal_loop():
    rng = np.random.default_rng(2)
    x = [run_ar([0.6, 0.1], 1, n, rng=rng) for n in (500, 300, 3)]
    order = 4
    kern, sigma = yw_ar(x, order)
    # literal MATLAB loops
    gamma = np.zeros(order)
    for lag in range(1, order + 1):
        cntr = 0
        for s in x:
            for k in range(1, len(s) - lag + 1):
                gamma[lag - 1] += s[k - 1] * s[k + lag - 1]
                cntr += 1
        gamma[lag - 1] /= cntr
    Gamma = np.ones((order, order))
    for i in range(1, order + 1):
        for j in range(i + 1, order + 1):
            Gamma[i - 1, j - 1] = Gamma[j - 1, i - 1] = gamma[abs(i - j) - 1]
    np.testing.assert_allclose(Gamma @ kern, gamma, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(sigma, np.sqrt(1 - gamma @ kern), rtol=1e-12)
    with pytest.raises(TypeError):
        yw_ar(np.ones(10), 2)  # FLAG[5c]: no wrapping of a single array


def test_run_ar_matches_matlab_recursion():
    kern = np.array([0.8, 0.2, -0.1])
    noise = np.random.default_rng(3).standard_normal(500)
    x = np.zeros(500)
    for t in range(1, 501):  # MATLAB: hist = fliplr(x(max(1,t-n):(t-1)))
        hist = x[max(1, t - 3) - 1 : t - 1][::-1]
        x[t - 1] = hist @ kern[: len(hist)] + 1.3 * noise[t - 1]
    np.testing.assert_allclose(
        run_ar(kern, 1.3, 500, noise=noise), x, rtol=1e-11, atol=1e-13
    )


def test_explosive_threshold():
    assert not explosive(_ar_kernel_from_roots([1.5, 2.0]))
    assert explosive(_ar_kernel_from_roots([0.5, 2.0]))
    assert explosive(_ar_kernel_from_roots([0.98, 2.0]))
    assert not explosive(_ar_kernel_from_roots([0.995, 2.0]))  # FLAG[4] tolerance
    assert not explosive([0.0, 0.0])  # no roots: MATLAB `if []` is false
    np.testing.assert_allclose(
        sorted(abs(_ar_roots(_ar_kernel_from_roots([1.5, -3.0])))), [1.5, 3.0]
    )


def test_matlab_colon():
    # MATLAB fills symmetrically: left half a + k*d, right half last - k*d,
    # so 0:0.1:0.3 is [0, 0.1, 0.3-0.1, 0.3] (bit-level parity with MATLAB on
    # real data is checked in test_spike_counts_match_matlab_histc).
    np.testing.assert_array_equal(matlab_colon(0, 0.1, 0.3), [0, 0.1, 0.3 - 0.1, 0.3])
    assert matlab_colon(0, 0.1, 0.3)[-1] == 0.3  # end point snapped, arange is not
    assert len(matlab_colon(0, 0.04, 1.0)) == 26
    assert len(matlab_colon(1, 1, 0)) == 0


def test_histc_semantics():
    st = np.array([0.0, 0.5, 1.0, 1.99, 2.0, 2.5, 3.0, 3.5])
    # edges 0,1,2,3 -> [0,1), [1,2), [2,3), ==3
    np.testing.assert_array_equal(get_spike_counts(st, 0, 3, 1.0), [2, 2, 2, 1])


# --------------------------------------------------------------------------- #
# calc_db interface and control flow
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ar3():
    return run_ar([0.8, 0.2, -0.1], 1, 5000, rng=np.random.default_rng(4))


def test_interface_and_error_paths(ar3):
    r = calc_db(ar3, 3, 1.0, 2)
    assert isinstance(r, DbResult) and len(r) == 7
    db, sddb, kern, sigma, H, kernc, exit_status = r
    assert exit_status == 0 and np.isnan(sddb) and np.isnan(H)
    # single array == list with one segment == flat Python list
    assert calc_db([ar3], 3, 1.0, 2).db == db
    assert calc_db(list(ar3), 3, 1.0, 2).db == db
    # delta_t only rescales:  bits/s = log2(e) / delta_t * nats/step
    np.testing.assert_allclose(calc_db(ar3, 3, 0.25, 2).db, 4 * db, rtol=1e-14)
    with pytest.warns(UserWarning, match="invalid fit_method"):
        bad = calc_db(ar3, 3, 1.0, 2, fit_method="OLS")
    assert bad.exit_status == 1 and np.isnan(bad.db)
    with pytest.warns(UserWarning, match="not supported with YuleWalker"):
        yw = calc_db(ar3, 3, 1.0, 2, with_err_bars=True)
    assert yw.exit_status == 1 and np.isnan(yw.sddb) and yw.db == db
    with pytest.raises(ValueError):
        calc_db(ar3, 3, 1.0, 3)  # FLAG[10]: odd b


def test_minimize_klr_b_closest_model(ar3):
    kern = calc_db(ar3, 4, 1.0, 2).kern
    db, cm, ex = minimize_klr_b(kern, 2)
    assert ex == 0
    np.testing.assert_allclose(cm["kernc"].sum(), 1, atol=1e-12)
    np.testing.assert_allclose(cm["sigmac"], get_sigmac(kern, cm["kernc"]), rtol=1e-14)
    assert db <= get_klr_b(kern, beta_project(calc_dbv1(kern, 2)[1], 2), 2, "integral")


def test_speed(ar3):
    t = time.perf_counter()
    calc_db(ar3, 5, 1.0, 2, fit_method="MaxLikelihood", with_err_bars=True)
    elapsed = time.perf_counter() - t
    print(f"calc_db(order 5, b 2, ML + error bars): {elapsed:.2f} s")
    assert elapsed < 3.0


# --------------------------------------------------------------------------- #
# 3. End-to-end: values printed in the original tutorial (demo.mlx)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_ephys_tutorial_values():
    d = sio.loadmat(YMV15_MAT, squeeze_me=True)
    st, dt = d["st"], 0.04
    opts = dict(fit_method="MaxLikelihood", with_err_bars=True)
    expected = {  # MATLAB output recorded in demo.mlx (num2str: 5 significant digits)
        "all": ("1.7122", "0.0081069"),
        "WAKE": ("1.0092", "0.015695"),
        "NREM": ("2.5523", "0.0149"),
        "REM": ("1.4795", "0.032053"),
    }
    r = calc_db(get_spike_counts(st, st.min(), st.max(), dt), 5, dt, 2, **opts)
    got = {"all": r}
    for lab in ("WAKE", "NREM", "REM"):
        x = [
            get_spike_counts(st, a, b, dt)
            for a, b in np.atleast_2d(d[lab]).astype(float)
        ]
        got[lab] = calc_db(x, 5, dt, 2, **opts)
    for k, (db_s, sd_s) in expected.items():
        assert f"{got[k].db:.5g}" == db_s, k
        assert f"{np.real(got[k].sddb):.5g}" == sd_s, k
        assert got[k].exit_status == 0

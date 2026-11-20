"""Tests for src.analysis.prox_crit (Python port of Sooter's prox_crit_toolkit).

Three layers:
1. MATLAB parity: outputs of the ORIGINAL MATLAB code on identical inputs
   (fixtures from tests/matlab/make_reference.m). The AR fits deliberately
   deviate from the original (exact for short segments, NOTICE.md), so these
   tests run calc_db with literal translations of the ORIGINAL estimators
   (_yw_ar_original, _mle_ar_original) patched in.
2. Independent math checks: the formulas themselves (time-domain vs spectral
   KL rate, manifold constraints, Hessian, literal MATLAB loops, ...).
3. End-to-end: the values printed in the original tutorial (demo.mlx) for the
   ephys data (slow; deselect with  -m "not slow").

Run:  python -m pytest tests/test_prox_crit.py -v
"""

import contextlib
import math
import time
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import scipy.io as sio
from scipy.optimize import minimize

from src.analysis.prox_crit import (
    DbResult,
    beta_lift,
    beta_project,
    calc_db,
    calc_db_euclidean,
    explosive,
    get_kernw,
    get_klr_b,
    get_sigma_ratio,
    get_spike_counts,
    matlab_range,
    minimize_klr_b,
    mle_ar,
    yw_ar,
)
from src.analysis.prox_crit.kl_rate import inv_spectrum
from src.analysis.prox_crit.manifold import ar_roots
from src.config.paths import YMV15_MAT
from src.models.autoregression.autoregression_model import simulate_arfima

FIX = Path(__file__).parent / "fixtures"
FITS = ["YuleWalker", "MaxLikelihood"]
LOG2E = np.log2(np.e)


# --------------------------------------------------------------------------- #
# Literal translations of the ORIGINAL estimators (yw_ar.m, mle_ar.m)
# --------------------------------------------------------------------------- #
def _yw_ar_original(x, order):
    """yw_ar.m: autocovariance divided by the number of pairs (N - lag), gamma(0) fixed to 1."""
    x = [np.ravel(seg) for seg in (x if isinstance(x, list) else [x])]
    gamma = np.zeros(order)
    for lag in range(1, order + 1):
        counter = 0
        for seg in x:
            for k in range(1, len(seg) - lag + 1):
                gamma[lag - 1] += seg[k - 1] * seg[k + lag - 1]
                counter += 1
        gamma[lag - 1] /= counter
    Gamma = np.ones((order, order))
    for i in range(1, order + 1):
        for j in range(i + 1, order + 1):
            Gamma[i - 1, j - 1] = Gamma[j - 1, i - 1] = gamma[abs(i - j) - 1]
    kern = np.linalg.solve(Gamma, gamma)
    return kern, np.emath.sqrt(1 - gamma @ kern)


def _mle_ar_original(x, order):
    """mle_ar.m: every sample is a regression row; missing history (before the segment start) is 0."""
    x = [np.ravel(seg) for seg in (x if isinstance(x, list) else [x])]
    rows, Y = [], []
    for seg in x:
        for j in range(1, len(seg) + 1):  # MATLAB j = 1..length
            hist = seg[max(1, j - order) - 1 : j - 1]
            rows.append(np.r_[hist[::-1], np.zeros(order - len(hist))])
            Y.append(seg[j - 1])
    X, Y = np.array(rows), np.array(Y)
    kern = np.linalg.solve(X.T @ X, X.T @ Y)
    resid = Y - X @ kern
    A, B, T = np.sum(resid**2), X.T @ resid, len(Y)
    H = T * (2 * np.outer(B, B) / A**2 - (X.T @ X) / A)
    return kern, np.sqrt(np.mean(resid**2)), H


@contextlib.contextmanager
def _original_estimators():
    """calc_db fits with the original estimators inside this block (for MATLAB parity)."""
    import src.analysis.prox_crit.distance as distance

    with (
        mock.patch.object(distance, "yw_ar", _yw_ar_original),
        mock.patch.object(distance, "mle_ar", _mle_ar_original),
    ):
        yield


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
    with warnings.catch_warnings(), _original_estimators():
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
    """kern / sigma / H of the original estimators (deterministic linear algebra)."""
    out = ref["out"]
    for i in _cases(ref, lambda d, o, b, f, e: o >= b // 2):
        r = ref["py"][i]
        # rtol 1e-9: max observed 3e-11 (YW, order 8: errors amplified by
        # cond(Gamma)); different BLAS/LAPACK summation order only.
        np.testing.assert_allclose(r.kern, np.ravel(out["kern"][i]), rtol=1e-9)
        np.testing.assert_allclose(r.sigma, out["sigma"][i], rtol=1e-12)
        if ref["cases"][i][3] == 1:
            np.testing.assert_allclose(r.H, out["H"][i], rtol=1e-12)


def test_calc_db_euclidean_matches_matlab(ref):
    K = ref["inp"]["dbv1_kerns"]
    for i, row in enumerate(K):
        kern, b = row[2 : 2 + int(row[0])], int(row[1])
        db_euclidean, kernc = calc_db_euclidean(kern, b)
        np.testing.assert_allclose(db_euclidean, ref["out"]["dbv1"][i], rtol=1e-12)
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
    """b >= 4: the objective is the same, but fmincon stops early (NOTICE.md, known issues).

    (a) Python's KL rate evaluated at MATLAB's kernc reproduces MATLAB's db
        (same objective function). rtol 2e-5: at b = 6 both quadratures
        (scipy, MATLAB) deviate up to ~1e-5 from a 25-digit mpmath value.
    (b) Python's db is never above MATLAB's.
    (c) Python's minimum is confirmed by an independent optimizer
        (Nelder-Mead). rtol 2e-5: finite-difference gradients of
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
    """order < b/2: MATLAB crashes on cm.kernc; Python returns NaN, exit 1."""
    (i,) = _cases(ref, lambda d, o, b, f, e: o < b // 2)
    assert "kernc" in ref["out"]["errmsg"][i]  # the MATLAB crash
    r = ref["py"][i]
    assert np.isnan(r.db) and np.isnan(r.kernc) and r.exit_status == 1


def test_spike_counts_match_matlab_histc(ref):
    """Bin edges (MATLAB colon) and histc counts: bit-identical (minus histc's extra last value, dropped by design)."""
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
            matlab_range(t1, 0.04, t2), np.ravel(out["bins_" + name])
        )
        np.testing.assert_array_equal(
            get_spike_counts(st, t1, t2, 0.04), np.ravel(out["counts_" + name])[:-1]
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
    rng = np.random.default_rng(seed=b)
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
def test_calc_db_euclidean_is_orthogonal_projection(b):
    rng = np.random.default_rng(seed=10 + b)
    kern = rng.normal(scale=0.3, size=7)
    db_euclidean, kernc = calc_db_euclidean(kern, b)
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
    np.testing.assert_allclose(db_euclidean, np.linalg.norm(kern - kernc), rtol=1e-14)
    # a kernel already on M_b is its own projection
    np.testing.assert_allclose(calc_db_euclidean(kernc, b)[1], kernc, atol=1e-12)


def test_get_kernw_and_spectrum():
    rng = np.random.default_rng(seed=0)
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
    g = inv_spectrum([a])
    np.testing.assert_allclose(g(w), 1 - 2 * a * np.cos(w) + a**2, rtol=1e-14)
    # scalar fast path (used by quad) == vectorized path
    gk = inv_spectrum(kern)
    np.testing.assert_allclose([gk(float(v)) for v in w], gk(w), rtol=1e-13)


@pytest.mark.parametrize("b, rtol", [(2, 1e-9), (4, 1e-6)])
def test_kl_rate_equals_time_domain_formula(b, rtol):
    """Spectral KL rate == conditional-likelihood (time-domain) KL rate.

    For the fitted model P (sigma = 1, stable) and critical model Q (kernc,
    sigma_c), with e = [1, -kernc] and Gamma_P the autocovariance matrix of P:
        sigma_c^2 = e' Gamma_P e
        klr       = 1/2 [log sigma_c^2 + e' Gamma_P e / sigma_c^2 - 1]
                  = 1/2 log(e' Gamma_P e)
    This also confirms that get_sigma_ratio returns sigma_c relative to sigma = 1. The b = 4
    case also exercises the g_c underflow next to w = 0 (get_klr_b); its rtol
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
    np.testing.assert_allclose(get_sigma_ratio(kern, kc) ** 2, v, rtol=1e-9)
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


@pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
def test_kl_rate_near_criticality(eps):
    """b = 2 close to criticality, where the KL rate sits in a log spike of width ~eps at w = 0 (get_klr_b).

    The original quadrature returned ~0 here from eps ~ 1e-4 on (relative error 1, with IntegrationWarnings); the log-w integration must be accurate and silent.
    (a) AR(1) kern = [1 - eps]: M_2 is one point (kernc = [1]), closed form klr = -1/2 log(1 - eps/2) (see notebooks/prox_crit_branching_network.ipynb). Observed error <= 3e-8 (relative).
    (b) order 5 with a root at 1 + eps, kernc = its L2 projection on M_2 (close, as during the optimization): time-domain formula (test_kl_rate_equals_time_domain_formula). Float autocovariances are ~1/eps large, so their round-off limits (b) to eps >= 1e-4 and to atol 1e-10 (= quad's absolute tolerance; observed 9e-12, while a 40-digit reference gives ~1e-14).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # also turns IntegrationWarnings into failures
        klr = get_klr_b([1 - eps], [], 2, "integral")
        np.testing.assert_allclose(klr, -0.5 * np.log1p(-eps / 2), rtol=1e-7)  # (a)

        if eps < 1e-4:
            return
        kern = _ar_kernel_from_roots([1 + eps, 1.6, -2.0, 2.5 + 1j, 2.5 - 1j])
        kc = calc_db_euclidean(kern, 2)[1]
        g = _autocov(kern, len(kc))
        Gamma = np.array(
            [[g[abs(i - j)] for j in range(len(kc) + 1)] for i in range(len(kc) + 1)]
        )
        e = np.r_[1.0, -kc]
        klr = get_klr_b(kern, beta_project(kc, 2), 2, "integral")
        np.testing.assert_allclose(
            klr, 0.5 * np.log(e @ Gamma @ e), rtol=0, atol=1e-10
        )  # (b)


def test_kl_rate_b4_warns_if_quad_fails():
    """b >= 4 keeps the original quadrature; if quad reports non-convergence, get_klr_b warns (and still returns its value)."""

    def fake_quad(*args, **kwargs):
        # the KL integral asks for full_output; with ier > 0 quad appends a message
        if kwargs.get("full_output"):
            return 0.123, 1e-3, {}, "quad message"
        return 1.0, 0.0  # sigma ratio integral

    with mock.patch("src.analysis.prox_crit.kl_rate.quad", side_effect=fake_quad):
        with pytest.warns(UserWarning, match="unreliable at b >= 4.*Powell"):
            assert get_klr_b([1.9, -0.9], [], 4, "integral") == 0.123


def test_mle_ar_against_literal_loop_and_hessian():
    rng = np.random.default_rng(seed=1)
    segs = [simulate_arfima(phi=[0.5, -0.2, 0.1], T=n, burn_in=0, seed=rng)[0] for n in (400, 250, 2)]
    order = 3
    kern, sigma, H = mle_ar(segs, order)

    # conditional likelihood: targets t = order+1..length (1-based) with full history; the 2-sample segment gives no row
    rows, Y = [], []
    for s in segs:
        for j in range(order + 1, len(s) + 1):
            rows.append(s[j - order - 1 : j - 1][::-1])
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
    rng = np.random.default_rng(seed=2)
    x = [simulate_arfima(phi=[0.6, 0.1], T=n, burn_in=0, seed=rng)[0] for n in (500, 300, 3)]
    order = 4
    kern, sigma = yw_ar(x, order)
    # literal loops: mean of the products over all pairs within segments, lag 0 included
    gamma = np.zeros(order + 1)
    for lag in range(order + 1):
        pairs = [s[k] * s[k + lag] for s in x for k in range(len(s) - lag)]
        gamma[lag] = sum(pairs) / len(pairs)
    Gamma = np.array([[gamma[abs(i - j)] for j in range(order)] for i in range(order)])
    np.testing.assert_allclose(Gamma @ kern, gamma[1:], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(sigma, np.sqrt(gamma[0] - gamma[1:] @ kern), rtol=1e-12)
    # a single array is a single segment (as for mle_ar)
    np.testing.assert_array_equal(yw_ar(x[0], order)[0], yw_ar([x[0]], order)[0])
    # with z-scored data, the only difference to the original is gamma(0) = (N-1)/N instead of 1
    pooled = np.concatenate(x)
    z = [(s - pooled.mean()) / pooled.std(ddof=1) for s in x]
    np.testing.assert_allclose(
        yw_ar(z, order)[0], _yw_ar_original(z, order)[0], atol=1e-2
    )


def test_short_windows():
    """Many short stationary windows of a known AR(2), pooled (NOTICE.md, deviations).

    yw_ar and the conditional mle_ar recover the true kernel; the original mle_ar, which treats the history before each window start as 0, is biased.
    """
    kern_true = np.array([0.6, 0.2])
    rng = np.random.default_rng(seed=5)
    stationary = simulate_arfima(phi=kern_true, T=400_000, burn_in=1000, seed=rng)[0]
    windows = list(stationary.reshape(-1, 10))  # 40000 windows of 10 samples
    kern_yw, kern_mle = yw_ar(windows, 2)[0], mle_ar(windows, 2)[0]
    kern_original = _mle_ar_original(windows, 2)[0]
    print(
        f"true {kern_true}, yw {kern_yw}, mle {kern_mle}, original mle {kern_original}"
    )
    np.testing.assert_allclose(kern_yw, kern_true, atol=0.01)
    np.testing.assert_allclose(kern_mle, kern_true, atol=0.01)
    assert np.all(np.abs(kern_original - kern_true) > 0.02)


def test_simulate_arfima_matches_matlab_recursion():
    kern = np.array([0.8, 0.2, -0.1])
    noise = np.random.default_rng(seed=3).standard_normal(500)
    x = np.zeros(500)
    for t in range(1, 501):  # MATLAB: hist = fliplr(x(max(1,t-n):(t-1)))
        hist = x[max(1, t - 3) - 1 : t - 1][::-1]
        x[t - 1] = hist @ kern[: len(hist)] + 1.3 * noise[t - 1]
    np.testing.assert_allclose(
        simulate_arfima(phi=kern, sigma=1.3, T=500, burn_in=0, seed=3)[0],  # same seed -> same noise
        x,
        rtol=1e-11,
        atol=1e-13,
    )


def test_simulate_arfima_integer_d_is_integrated_ar():
    # (1 - L)^2 x = eps  <=>  x_t = 2 x_{t-1} - x_{t-2} + eps_t; same seed -> same noise
    x_fractional = simulate_arfima(d=2, T=2000, R=2, seed=6)
    x_classical = simulate_arfima(phi=[2, -1], T=2000, R=2, seed=6)
    np.testing.assert_allclose(x_fractional, x_classical, rtol=1e-8, atol=1e-6)


def test_explosive_threshold():
    assert not explosive(_ar_kernel_from_roots([1.5, 2.0]))
    assert explosive(_ar_kernel_from_roots([0.5, 2.0]))
    assert explosive(_ar_kernel_from_roots([0.98, 2.0]))
    assert not explosive(
        _ar_kernel_from_roots([0.995, 2.0])
    )  # tolerance: |z| >= 0.99 is not explosive
    assert not explosive([0.0, 0.0])  # no roots: MATLAB `if []` is false
    np.testing.assert_allclose(
        sorted(abs(ar_roots(_ar_kernel_from_roots([1.5, -3.0])))), [1.5, 3.0]
    )


def test_matlab_range():
    # MATLAB fills symmetrically: left half a + k*d, right half last - k*d,
    # so 0:0.1:0.3 is [0, 0.1, 0.3-0.1, 0.3] (bit-level parity with MATLAB on
    # real data is checked in test_spike_counts_match_matlab_histc).
    np.testing.assert_array_equal(matlab_range(0, 0.1, 0.3), [0, 0.1, 0.3 - 0.1, 0.3])
    assert matlab_range(0, 0.1, 0.3)[-1] == 0.3  # end point snapped, arange is not
    assert len(matlab_range(0, 0.04, 1.0)) == 26
    assert len(matlab_range(1, 1, 0)) == 0


def test_spike_count_bins_are_half_open():
    st = np.array([0.0, 0.5, 1.0, 1.99, 2.0, 2.5, 3.0, 3.5])
    # edges 0,1,2,3 -> bins [0,1), [1,2), [2,3); 3.0 and 3.5 are outside
    np.testing.assert_array_equal(get_spike_counts(st, 0, 3, 1.0), [2, 2, 2])


# --------------------------------------------------------------------------- #
# calc_db interface and control flow
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ar3():
    return simulate_arfima(phi=[0.8, 0.2, -0.1], T=5000, burn_in=0, seed=4)[0]


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
        calc_db(ar3, 3, 1.0, 3)  # odd b is rejected


def test_euclidean_metric(ar3):
    """metric='euclidean': db is the L2 distance of the fitted kernel to M_b, unitless; error bars use the same delta method."""
    kl = calc_db(ar3, 4, 1.0, 2, fit_method="MaxLikelihood")
    r = calc_db(
        ar3,
        4,
        0.25,
        2,
        fit_method="MaxLikelihood",
        metric="euclidean",
        with_err_bars=True,
    )
    db_euclidean, kernc = calc_db_euclidean(kl.kern, 2)
    assert r.exit_status == 0 and r.db == db_euclidean
    np.testing.assert_array_equal(r.kernc, kernc)
    assert np.isfinite(r.sddb) and r.sddb > 0
    with pytest.warns(UserWarning, match="invalid model order"):
        assert np.isnan(calc_db(ar3, 1, 1.0, 4, metric="euclidean").db)
    with pytest.raises(ValueError):
        calc_db(ar3, 4, 1.0, 2, metric="L1")


def test_powell_optimizer(ar3):
    """optimizer='Powell' agrees with SLSQP where SLSQP works, and finds the lower minimum where it does not (b = 6 near criticality, NOTICE.md)."""
    kern = calc_db(ar3, 4, 1.0, 2).kern
    np.testing.assert_allclose(
        minimize_klr_b(kern, 2, optimizer="Powell")[0],
        minimize_klr_b(kern, 2)[0],
        rtol=1e-8,
    )
    near_critical = np.r_[1 - 1e-3, np.zeros(4)]
    db_slsqp = minimize_klr_b(near_critical, 6)[0]
    db_powell, cm, exit_flag = minimize_klr_b(near_critical, 6, optimizer="Powell")
    # SLSQP stops 1.6% above (41% before the sigma ratio was integrated in log w, NOTICE.md)
    assert exit_flag == 0 and db_powell < 0.99 * db_slsqp
    with pytest.raises(ValueError):
        minimize_klr_b(kern, 2, optimizer="BFGS")


@pytest.mark.parametrize("gap, warns", [(5e-4, True), (2e-3, False)])
def test_error_bar_step_crossing_criticality_warns(ar3, gap, warns):
    """Known issue 5: warn iff 0 < 1 - sum(kern) < step (1e-3), ie the + perturbations of the error bars cross M_2. The fit is mocked to a kernel with sum 1 - gap."""
    import src.analysis.prox_crit.distance as distance

    kern = np.r_[0.6, 0.2, 0.1, 0.05, 0.05 - gap]
    fit = (kern, 0.3, -1e5 * np.eye(5))  # (kern, sigma, H) as returned by mle_ar
    with mock.patch.object(distance, "mle_ar", return_value=fit):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calc_db(
                ar3,
                5,
                1.0,
                2,
                fit_method="MaxLikelihood",
                metric="euclidean",
                with_err_bars=True,
            )
    assert any("error bars unreliable" in str(w.message) for w in caught) == warns


def test_minimize_klr_b_closest_model(ar3):
    kern = calc_db(ar3, 4, 1.0, 2).kern
    db, cm, ex = minimize_klr_b(kern, 2)
    assert ex == 0
    np.testing.assert_allclose(cm["kernc"].sum(), 1, atol=1e-12)
    np.testing.assert_allclose(
        cm["sigma_ratio"], get_sigma_ratio(kern, cm["kernc"]), rtol=1e-14
    )
    assert db <= get_klr_b(
        kern, beta_project(calc_db_euclidean(kern, 2)[1], 2), 2, "integral"
    )


def test_speed(ar3):
    t = time.perf_counter()
    calc_db(ar3, 5, 1.0, 2, fit_method="MaxLikelihood", with_err_bars=True)
    elapsed = time.perf_counter() - t
    print(f"calc_db(order 5, b 2, ML + error bars): {elapsed:.2f} s")
    assert elapsed < 3.0


# --------------------------------------------------------------------------- #
# 3. End-to-end: values printed in the original tutorial (demo.mlx)
# --------------------------------------------------------------------------- #
def _histc_series(st, t1, t2, dt):
    """MATLAB-style series: our counts plus histc's extra last value #{st == last edge}."""
    last_edge = matlab_range(t1, dt, t2)[-1]
    return np.r_[get_spike_counts(st, t1, t2, dt), np.sum(st == last_edge)]


@pytest.mark.slow
@pytest.mark.parametrize("matlab_binning", [True, False])
def test_ephys_tutorial_values(matlab_binning):
    """d2 +- sddb on the real data vs the values printed in the original tutorial.

    matlab_binning=True: with MATLAB's exact series (incl. histc's extra value) and the original estimators, the printed digits are reproduced exactly. matlab_binning=False: our defaults (last bin dropped, conditional mle_ar) move d2 by at most 0.05% and sddb by 0.13% (rtol 1e-3 / 1e-2; error bars are 1-2% of d2).
    """
    d = sio.loadmat(YMV15_MAT, squeeze_me=True)
    st, dt = d["st"], 0.04
    series = _histc_series if matlab_binning else get_spike_counts
    opts = dict(fit_method="MaxLikelihood", with_err_bars=True)
    expected = {  # MATLAB output recorded in demo.mlx (num2str: 5 significant digits)
        "all": ("1.7122", "0.0081069"),
        "WAKE": ("1.0092", "0.015695"),
        "NREM": ("2.5523", "0.0149"),
        "REM": ("1.4795", "0.032053"),
    }
    with _original_estimators() if matlab_binning else contextlib.nullcontext():
        r = calc_db(series(st, st.min(), st.max(), dt), 5, dt, 2, **opts)
        got = {"all": r}
        for lab in ("WAKE", "NREM", "REM"):
            x = [series(st, a, b, dt) for a, b in np.atleast_2d(d[lab]).astype(float)]
            got[lab] = calc_db(x, 5, dt, 2, **opts)
    for k, (db_s, sd_s) in expected.items():
        assert got[k].exit_status == 0
        if matlab_binning:
            assert f"{got[k].db:.5g}" == db_s, k
            assert f"{np.real(got[k].sddb):.5g}" == sd_s, k
        else:
            np.testing.assert_allclose(got[k].db, float(db_s), rtol=1e-3, err_msg=k)
            np.testing.assert_allclose(
                np.real(got[k].sddb), float(sd_s), rtol=1e-2, err_msg=k
            )

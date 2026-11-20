"""Spectral Kullback-Leibler divergence rate between two AR models.

NOT OWN WORK. Python translation of the local functions ``get_kernw``, ``get_sigmac`` (here ``get_sigma_ratio``) and ``get_klr_b`` from ``calc_db.m``, prox_crit_toolkit, J. S. Sooter et al. (https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09), licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude). Logic unchanged; two numerical details of the quadrature differ (no effect on the values), and the sigma ratio and (for b = 2) the KL rate are integrated in log w (corrects values close to criticality), see NOTICE.md.

Formulas (for recognition, no derivation), with w = angular frequency in [-pi, pi] (per time step):

* kernel Fourier transform:   kern^(w) = sum_{k=1..n} kern_k e^{-i w k}
* inverse spectral shape:     g(w) = |1 - kern^(w)|^2      (the AR power spectrum is sigma^2 / g(w))
* KL-divergence rate (nats per time step) from the fitted model (kern, sigma) to a critical model (kernc, sigma_c). It depends on the noise levels only through r = sigma_c / sigma::

      klr = 1/(4 pi) * int [ log(r^2 g / g_c) + g_c / (r^2 g) - 1 ] dw

* the ratio r minimizing klr:  r^2 = 1/(2 pi) * int g_c / g dw

Abbreviations: kernw = kernel in the frequency domain (w = omega), klr = KL rate, kernc = kernel of the critical model, _b = on the critical manifold of type b.

Numerical integration: MATLAB's ``integral(f, -pi, pi, 'AbsTol', 1e-10)`` (adaptive Gauss-Kronrod, default RelTol 1e-6, MaxIntervalCount 650) is translated to ``scipy.integrate.quad`` with the same tolerances. The sigma ratio and, for b = 2, the KL rate are integrated in u = log w (``integrate_log_omega``, reasons in ``get_klr_b``): close to criticality the original misses the narrow structure of the integrands at w = 0.
"""

import cmath
import math
import warnings

import numpy as np
from scipy.integrate import quad

from .manifold import beta_lift

# 'AbsTol' as in the original; RelTol and MaxIntervalCount are MATLAB's integral() defaults.
QUAD_OPTS = dict(epsabs=1e-10, epsrel=1e-6, limit=650)


def get_kernw(kern, w):
    """Fourier transform of the kernel, sum_{t=1..n} kern_t exp(-i w t); vectorized over w (result has the shape of w)."""
    kern = np.asarray(kern, dtype=float).ravel()
    w = np.asarray(w, dtype=float)
    lags = np.arange(
        1, len(kern) + 1
    )  # true lags 1..n, the exponent needs them (not indices 0..n-1)
    return np.exp(-1j * np.multiply.outer(w, lags)) @ kern


def inv_spectrum(kern):
    """Return the function g(w) = |1 - kern^(w)|^2 (inverse spectral shape) of an AR kernel.

    Array input uses the vectorized get_kernw. Scalar input, which is what quad passes (one point per call), evaluates kern^(w) = sum_t kern_t z^t with z = e^{-iw} by Horner's rule in plain Python: same formula, ~10x less overhead per point.
    """
    kern = np.asarray(kern, dtype=float).ravel()
    coeffs_high_first = [float(c) for c in kern[::-1]]  # kern_n, ..., kern_1

    def g(w):
        if isinstance(w, float): # if w is a scalar, use Horner's rule for efficiency, else return vector
            z = cmath.exp(-1j * w)
            p = 0j
            for c in coeffs_high_first:  # p = z(kern_1 + z(kern_2 + ... z kern_n))
                p = (p + c) * z
            return abs(1 - p) ** 2
        return np.abs(1 - get_kernw(kern, w)) ** 2

    return g


def integrate_log_omega(integrand):
    """int_{-pi}^{pi} f(w) dw of an even integrand f, integrated in u = log w. Deviation from the original (NOTICE.md); why it is needed: see get_klr_b.

    Even f (real kernels: g(-w) = g(w)), and with w = e^u, dw = e^u du:
        int_{-pi}^{pi} f(w) dw = 2 int_0^pi f(w) dw = 2 int_{-inf}^{log pi} f(e^u) e^u du
    quad then spaces its nodes evenly in log w, so structure of any width delta at w = 0 becomes a smooth step of width O(1) in u at u = log(delta), and a log singularity f ~ -log w becomes ~ -u e^u -> 0 (u -> -inf).
    Cutoff w_min = 1e-15: the dropped part int_0^{w_min} f dw is < ~1e-13 for the integrands here (bounded, or ~log(delta / w) / (2 pi)), far below the absolute tolerance 1e-10.
    """
    omega_min = 1e-15
    half_value, _ = quad(
        lambda u: integrand(math.exp(u)) * math.exp(u),
        math.log(omega_min),
        math.log(np.pi),
        **QUAD_OPTS,
    )
    return 2 * half_value


def get_sigma_ratio(kern, kernc):
    """Ratio r = sigma_c / sigma of the noise stds (critical / fitted model) that minimizes the KL rate: r^2 = 1/(2 pi) int g_c / g dw.

    Original name ``get_sigmac``: the formula sets the fitted model's sigma to 1. The fitted sigma is not 1 in general (z-scoring normalizes the variance of x, not of the noise), so the value is the ratio; absolute noise std of the critical model = sigma * r.

    Integrated in log w (integrate_log_omega), unlike the original: close to criticality g_c / g has a dip to 0 of width ~delta at w = 0 (mirror image of the KL spike, see get_klr_b) that quad on [-pi, pi] misses. The missed area ~pi delta makes r^2 too large by ~delta/2 (relative); klr is stationary in r, so this only costs ~delta^2 / 16 in klr, but that is a relative error of ~delta/4 (2.5e-5 at delta = 1e-4) instead of ~1e-10.
    """
    g = inv_spectrum(kern)
    gc = inv_spectrum(kernc)

    def integrand(w):
        return 1 / (2 * np.pi) * gc(w) / g(w)

    return np.sqrt(integrate_log_omega(integrand))


def get_klr_b(kern, kernc, b, int_method):
    """KL divergence rate from the AR model ``kern`` to the critical model on M_b given by the reduced kernel ``kernc``.

    Parameters
    ----------
    kern : (n,) array
        Kernel of the fitted AR(n) model.
    kernc : (n - b/2,) array
        REDUCED kernel of a model on M_b (lifted to the full kernel inside).
    b : int
        Which critical manifold.
    int_method : {'trapz', 'integral'}
        Fixed 1000-point trapezoid rule, or adaptive quadrature (used for the optimization).

    Returns
    -------
    klr : float
        KL divergence rate in nats per time step, with the optimal noise ratio (see module docstring).
    """
    kernc = beta_lift(kernc, b)
    sigma_ratio_sq = get_sigma_ratio(kern, kernc) ** 2
    g = inv_spectrum(kern)
    gc = inv_spectrum(kernc)

    def integrand(w):
        gw, gcw = g(w), gc(w)
        if isinstance(w, float):  # quad, one point at a time
            if gcw == 0.0:
                # For b >= 4, g_c ~ w^b near w = 0; quad bisects towards that singularity, and for |w| < ~1e-8 floating-point cancellation in 1 - kernc^(w) makes g_c exactly 0, so the integrand and the whole integral become +inf (MATLAB's integral() does not probe that close). Such a node gets the value 0: its weight is negligible (< ~1e-8), and the result matches the exact KL rate to ~1e-7 (see tests).
                return 0.0
            return (
                1
                / (4 * np.pi)
                * (
                    math.log(sigma_ratio_sq * gw / gcw)
                    + gcw / (sigma_ratio_sq * gw)
                    - 1
                )
            )
        return (
            1
            / (4 * np.pi)
            * (np.log(sigma_ratio_sq * gw / gcw) + gcw / (sigma_ratio_sq * gw) - 1)
        )

    if int_method == "trapz":
        omega = np.linspace(
            -np.pi, np.pi, 1000
        )  # even count: w = 0 is not a grid point
        return np.trapezoid(integrand(omega), omega)
    if int_method == "integral":
        if b == 2:
            # Deviation from the original (NOTICE.md): integrate in u = log w instead of w.
            #
            # Why: close to criticality the whole KL rate sits in a narrow log spike at w = 0. Integrand = x - 1 - log x with x = S/S_c = g_c / (r^2 g) (fitted over critical power spectrum). For a fitted kernel with sum 1 - delta, Taylor around w = 0 with a = sum_k k kern_k (a_c same for kernc):
            #   g   ~ delta^2 + a^2 w^2      (fitted spectrum: tall but finite at 0)
            #   g_c ~ a_c^2 w^2              (critical spectrum: diverges at 0)
            #   x   ~ (a_c / (r a))^2 * w^2 / (w^2 + (delta/a)^2)
            # - |w| >> delta/a: both spectra ~ 1/w^2, x ~ const (~1 near the optimum), integrand ~ 0
            # - |w| << delta/a: fitted spectrum saturates, x ~ w^2, integrand ~ -log x / (4 pi) ~ log(delta / (a|w|)) / (2 pi), a log singularity (r ~ 1, a_c ~ a near the optimum)
            # So klr ~ int_{|w| < delta/a} log(delta / (a|w|)) / (2 pi) dw = delta / (pi a): a spike of width delta/a, only logarithmically high. (Check, AR(1): exact klr = -1/2 log(1 - delta/2) ~ delta/4.)
            # QUADPACK's first 21-point rule on [0, pi] has its node closest to 0 at w ~ 0.007, and its extrapolation assumes a clean endpoint singularity, so for delta/a ≲ 1e-3 it can miss the spike and return values orders of magnitude too small (with IntegrationWarnings; observed from delta ~ 3e-4 on). This broke the error bars of near-critical fits (step 0.001 in calc_db pushes kern towards M_2), and, closer still, d_2 itself.
            # Fix: integrate in u = log w (integrate_log_omega), where the spike is a smooth step of width O(1) at u = log(delta/a), wherever delta lies, and the singularity decays like -u e^u.
            #
            # b >= 4 keeps the original quadrature: g_c ~ w^b against g ~ w^2 makes the log region broad (klr = O(0.1), no narrow spike), and in u the noisy g_c at tiny w (cancellation in 1 - kernc^(w)) gets sampled densely, which made SLSQP's finite-difference gradients stop early (+1.6% db in a MATLAB reference case).
            return integrate_log_omega(integrand)
        # b >= 4, as in the original. points=[0.0] splits the interval at w = 0. On M_b, g_c(0) = |1 - sum(kernc)|^2 = 0: an integrable log singularity. QUADPACK's first 21-point rule on [-pi, pi] would evaluate exactly w = 0 (-> inf), MATLAB's integral() never places a node there. The value of the integral is unchanged.
        # quad_out = (klr, abserr, infodict) or, if quad did not converge (ier > 0), (klr, abserr, infodict, message)
        quad_out = quad(
            integrand, -np.pi, np.pi, points=[0.0], full_output=1, **QUAD_OPTS
        )
        if len(quad_out) > 3:
            warnings.warn(
                "kernel close to criticality, KL integral unreliable at b >= 4. optimizer='Powell' is slower but more reliable (NOTICE.md, known issue 2)."
            )
        return quad_out[0]
    # The original only prints this message and then fails on the undefined result.
    raise ValueError(
        'ERROR (get_klr_b): specific integration method. Valid options are "trapz" and "integral"'
    )

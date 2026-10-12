"""Spectral Kullback-Leibler divergence rate between two AR models.

NOT OWN WORK. Python translation of the local functions ``get_kernw``, ``get_sigmac`` and ``get_klr_b`` from ``calc_db.m``, prox_crit_toolkit, J. S. Sooter et al. (https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09), licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude). Logic unchanged except CHANGED[1] and CHANGED[4]; see FLAGS.md.

Formulas (for recognition, no derivation), with w = angular frequency in [-pi, pi] (per time step):

* kernel Fourier transform:   kern^(w) = sum_{k=1..n} kern_k e^{-i w k}
* inverse spectral shape:     g(w) = |1 - kern^(w)|^2      (the AR power spectrum is sigma^2 / g(w))
* KL-divergence rate (nats per time step) from the fitted model (kern, sigma) to a critical model (kernc, sigma_c). It depends on the noise levels only through r = sigma_c / sigma::

      klr = 1/(4 pi) * int [ log(r^2 g / g_c) + g_c / (r^2 g) - 1 ] dw

* the ratio r minimizing klr:  r^2 = 1/(2 pi) * int g_c / g dw

Abbreviations: kernw = kernel in the frequency domain (w = omega), klr = KL rate, kernc = kernel of the critical model, _b = on the critical manifold of type b.

Numerical integration: MATLAB's ``integral(f, -pi, pi, 'AbsTol', 1e-10)`` (adaptive Gauss-Kronrod, default RelTol 1e-6, MaxIntervalCount 650) is translated to ``scipy.integrate.quad`` with the same tolerances.
"""

import cmath
import math

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
        if isinstance(w, float):
            z = cmath.exp(-1j * w)
            p = 0j
            for c in coeffs_high_first:  # p = z(kern_1 + z(kern_2 + ... z kern_n))
                p = (p + c) * z
            return abs(1 - p) ** 2
        return np.abs(1 - get_kernw(kern, w)) ** 2

    return g


def get_sigma_ratio(kern, kernc):
    """Noise ratio r = sigma_c / sigma of the critical model that minimizes the KL rate: r^2 = 1/(2 pi) int g_c / g dw.

    Named ``get_sigmac`` in the original, which suggests the absolute noise std of the critical model. The formula sets the fitted model's sigma to 1, so the value is the ratio sigma_c / sigma (the KL rate depends only on this ratio). The absolute value is sigma * r.
    """
    g = inv_spectrum(kern)
    gc = inv_spectrum(kernc)

    def integrand(w):
        return 1 / (2 * np.pi) * gc(w) / g(w)

    value, _ = quad(integrand, -np.pi, np.pi, **QUAD_OPTS)
    return np.sqrt(value)


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
                # CHANGED[4]: for b >= 4, g_c ~ w^b near w = 0; quad bisects towards that singularity, and for |w| < ~1e-8 floating-point cancellation in 1 - kernc^(w) makes g_c exactly 0, so the integrand and the whole integral become +inf (MATLAB's integral() does not probe that close). Such a node gets the value 0: its weight is negligible (< ~1e-8), and the result matches the exact KL rate to ~1e-7 (see tests).
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
        # CHANGED[1]: points=[0.0] splits the interval at w = 0. On M_b, g_c(0) = |1 - sum(kernc)|^2 = 0: an integrable log singularity. QUADPACK's first 21-point rule on [-pi, pi] would evaluate exactly w = 0 (-> inf), MATLAB's integral() never places a node there. The value of the integral is unchanged.
        klr, _ = quad(integrand, -np.pi, np.pi, points=[0.0], **QUAD_OPTS)
        return klr
    # The original only prints this message and then fails on the undefined result.
    raise ValueError(
        'ERROR (get_klr_b): specific integration method. Valid options are "trapz" and "integral"'
    )

"""Spectral Kullback-Leibler divergence rate between two AR models.

NOT OWN WORK. Python translation of the local functions ``get_kernw``,
``get_sigmac`` and ``get_klr_b`` from ``calc_db.m``, prox_crit_toolkit,
J. S. Sooter et al. (https://github.com/sam-sooter/prox_crit_toolkit,
fetched 2026-10-09), licensed CC BY-NC 4.0. Translated by Andre Potthoff
(with Claude). Logic unchanged except CHANGED[1]; see FLAGS.md.

Formulas (for recognition, no derivation), with w = angular frequency in
[-pi, pi] (per time step):

* kernel Fourier transform     kern^(w) = sum_{k=1..n} kern_k e^{-i w k}
* inverse spectral shape        g(w)     = |1 - kern^(w)|^2
  (the AR power spectrum is sigma^2 / g(w))
* optimal noise of the critical model (minimises the KL rate over sigma_c):
                                sigma_c^2 = 1/(2 pi) * int g_c / g dw
* KL-divergence rate (nats per time step) from the fitted model (sigma := 1)
  to the critical model (kern_c, sigma_c):
      klr = 1/(4 pi) * int [ log(sigma_c^2 g / g_c) + g_c / (sigma_c^2 g) - 1 ] dw

Abbreviations: kernw = kernel in the frequency domain (w = omega),
klr = KL rate, sigmac / kernc = noise std / kernel of the critical model,
_b = on the critical manifold of type b.

Numerical integration: MATLAB ``integral(f, -pi, pi, 'AbsTol', 1e-10)``
(adaptive Gauss-Kronrod, default RelTol 1e-6, MaxIntervalCount 650) is
translated to ``scipy.integrate.quad`` with the same tolerances.
"""

import cmath
import math
import warnings

import numpy as np
from scipy.integrate import quad

from .manifold import beta_lift

# MATLAB integral() settings: 'AbsTol' given in the original; RelTol and
# MaxIntervalCount are MATLAB's defaults.
QUAD_OPTS = dict(epsabs=1e-10, epsrel=1e-6, limit=650)


def get_kernw(kern, w):
    """Fourier transform of the kernel: sum_{t=1..n} kern_t exp(-i w t).

    Vectorized over w (any shape); the result has the shape of w.
    """
    kern = np.asarray(kern, dtype=float).ravel()
    w = np.asarray(w, dtype=float)
    t = np.arange(1, len(kern) + 1)  # true lags 1..n (kern[t-1] <-> kern(t))
    return np.exp(-1j * np.multiply.outer(w, t)) @ kern


def _inv_spectrum(kern):
    """Return the function g(w) = |1 - kern^(w)|^2 (inverse spectral shape).

    Speed only, same formula as ``abs(1 - get_kernw(kern, w)).^2`` in MATLAB:
    array input uses the vectorized get_kernw; scalar input (what
    scipy's quad passes, one point per call) evaluates
    kern^(w) = sum_t kern_t z^t, z = e^{-iw}, with Horner's rule in plain
    Python, avoiding ~10x numpy call overhead per point.
    """
    kern = np.asarray(kern, dtype=float).ravel()
    coeffs_high_first = [float(c) for c in kern[::-1]]  # kern_n, ..., kern_1

    def g(w):
        if isinstance(w, float):  # scalar (quad passes Python floats)
            z = cmath.exp(-1j * w)
            p = 0j
            for c in coeffs_high_first:  # p = z(kern_1 + z(kern_2 + ... z kern_n))
                p = (p + c) * z
            return abs(1 - p) ** 2
        return np.abs(1 - get_kernw(kern, w)) ** 2

    return g


def get_sigmac(kern, kernc):
    """Value of sigma_c that minimizes the KL div. rate for given kern, kernc.

    sigma_c = sqrt( 1/(2 pi) * int_{-pi}^{pi} g_c(w) / g(w) dw )

    FLAG[2]: the fitted model's noise sigma is implicitly 1 here (and in
    get_klr_b), so sigma_c is really the RATIO sigma_c / sigma. This is
    consistent (the KL rate only depends on that ratio) but easy to misread.
    """
    g = _inv_spectrum(kern)  # fitted model
    gc = _inv_spectrum(kernc)  # critical model

    def fun(w):
        return 1 / (2 * np.pi) * gc(w) / g(w)

    val, _ = quad(fun, -np.pi, np.pi, **QUAD_OPTS)
    return np.sqrt(val)


def get_klr_b(kern, kernc, b, int_method):
    """KL div. rate between two AR models, one of which lies on M_b.

    Parameters
    ----------
    kern : (n,) array
        Kernel of an AR(n) model (the fitted model).
    kernc : (n - b/2,) array
        REDUCED kernel of an AR(n) model on M_b (lifted inside, beta_lift).
    b : int
        Which critical manifold.
    int_method : {'trapz', 'integral'}
        Fixed 1000-point trapezoid rule, or adaptive quadrature.

    Returns
    -------
    klr : float
        KL divergence rate in nats per time step (see module docstring).
    """
    kernc = beta_lift(kernc, b)
    sigmac = get_sigmac(kern, kernc)
    g = _inv_spectrum(kern)
    gc = _inv_spectrum(kernc)
    s2 = sigmac**2

    def fun2(w):
        gw, gcw = g(w), gc(w)
        if isinstance(w, float):  # scalar (quad): math.log avoids numpy overhead
            if gcw == 0.0:
                # CHANGED[4]: for b >= 4, g_c ~ w^b near w = 0; quad bisects
                # towards that singularity, and for |w| < ~1e-8 floating-point
                # cancellation in 1 - kernc^(w) makes g_c exactly 0, so the
                # integrand is +inf and the whole integral inf (MATLAB's
                # integral() does not probe that close). Such a node gets the
                # value 0: its weight is negligible (< ~1e-8), and the result
                # matches the exact KL rate to ~1e-7 (see tests).
                return 0.0
            return 1 / (4 * np.pi) * (math.log(s2 * gw / gcw) + gcw / (s2 * gw) - 1)
        return 1 / (4 * np.pi) * (np.log(s2 * gw / gcw) + gcw / (s2 * gw) - 1)

    if int_method == "trapz":
        xx = np.linspace(-np.pi, np.pi, 1000)  # even count: w = 0 not a node
        klr = np.trapezoid(fun2(xx), xx)
    elif int_method == "integral":
        # CHANGED[1]: points=[0.0] splits the interval at w = 0. On M_b,
        # g_c(0) = |1 - sum(kernc)|^2 = 0, an integrable log singularity of
        # the integrand. QUADPACK's first 21-point rule on [-pi, pi] would
        # evaluate exactly w = 0 (-> inf/nan), whereas MATLAB's integral()
        # never places a node there. The value of the integral is unchanged.
        klr, _ = quad(fun2, -np.pi, np.pi, points=[0.0], **QUAD_OPTS)
    else:
        warnings.warn(
            "ERROR (get_klr_b): specific integration method\n"
            'Valid options are "trapz" and "integral"'
        )
        # FLAG[1]: like the MATLAB original, `klr` is undefined on this path,
        # so the return below raises (UnboundLocalError; MATLAB: error).

    return klr

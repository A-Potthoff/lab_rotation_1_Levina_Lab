"""Auxiliary functions of the toolkit demos: AR simulation and spike binning.

NOT OWN WORK. Python translation of the auxiliary functions ``runAR`` and
``get_spike_counts`` from ``demo/demo.mlx`` and ``demo/demo_crit_db.m``,
prox_crit_toolkit, J. S. Sooter et al.
(https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09),
licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude).
Logic unchanged; see FLAGS.md.
"""

import numpy as np
from scipy.signal import lfilter

def run_ar(kern, sigma, T, rng=None, noise=None):
    """Simulate T steps of the AR model x_t = sum_k kern_k x_{t-k} + sigma eps_t.

    Starts from x = 0 before t = 1 (no burn-in), exactly as the MATLAB loop:
    x(t) = fliplr(x(max(1,t-n):(t-1))) * kern(1:...) + sigma*randn.
    Implemented as the equivalent IIR filter (scipy.signal.lfilter with zero
    initial state), which is exact and much faster than a Python loop.

    Parameters
    ----------
    kern : array
        History kernel (kern[k-1] = coefficient of lag k).
    sigma : float
        Noise standard deviation.
    T : int
        Number of time steps.
    rng : numpy.random.Generator, optional
        Random generator for the noise (default: np.random.default_rng()).
        MATLAB's randn stream cannot be reproduced, so simulated series differ
        from MATLAB's sample by sample (not in distribution).
    noise : (T,) array, optional
        Pre-drawn standard-normal noise (overrides rng; used for testing).

    Returns
    -------
    x : (T,) array
    """
    kern = np.asarray(kern, dtype=float).ravel()
    if noise is None:
        rng = np.random.default_rng() if rng is None else rng
        noise = rng.standard_normal(int(T))
    # y_t - sum_k kern_k y_{t-k} = sigma * eps_t
    return lfilter([1.0], np.r_[1.0, -kern], sigma * np.asarray(noise, dtype=float))


def matlab_colon(a, d, b):
    """MATLAB's floating-point colon operator ``a:d:b`` (for d > 0).

    np.arange does not include endpoints.
    This function replicates the behavior of MATLAB's colon operator.
    """
    a, d, b = float(a), float(d), float(b)
    if d <= 0 or a > b:
        return np.array([])
    tol = 2.0 * np.finfo(float).eps * max(abs(a), abs(b))
    n = int(np.round((b - a) / d))
    if a + n * d - b > tol:
        n -= 1
    last = a + n * d
    if last - b > -tol:  # snap the end point onto b
        last = b
    v = np.zeros(n + 1)
    k = np.arange(0, n // 2 + 1)
    v[k] = a + k * d
    v[n - k] = last - k * d
    if n % 2 == 0:
        v[n // 2] = (a + last) / 2
    return v


def get_spike_counts(st, t1, t2, delta_t):
    """Spike-count time series: counts of spike times ``st`` in bins of delta_t.

    Exact translation of MATLAB ``histc(st, t1:deltaT:t2)``: with N edges
    e_1..e_N it returns N counts, count_k = #{e_k <= st < e_{k+1}} for k < N and
    count_N = #{st == e_N}. Spikes outside [t1, e_N] are ignored.

    FLAG[12]: the last count only holds spikes exactly at the final edge, so
    every series ends with a spurious, almost always 0, sample.

    Returns
    -------
    x : (N,) float array
    """
    edges = matlab_colon(t1, delta_t, t2)
    n_edges = len(edges)
    st = np.asarray(st, dtype=float).ravel()
    if n_edges == 0:
        return np.zeros(0)
    st = st[(st >= edges[0]) & (st <= edges[-1])]  # pre-select (speed only)
    k = np.searchsorted(edges, st, side="right") - 1  # edges[k] <= st < edges[k+1]
    # st == edges[-1] gives k = N-1 (the last, closed "bin")
    return np.bincount(k, minlength=n_edges).astype(float)

"""Spike-count time series from spike times (binning).

NOT OWN WORK. Python translation of the auxiliary function ``get_spike_counts`` from ``demo/demo.mlx`` and ``demo/demo_crit_db.m``,
prox_crit_toolkit, J. S. Sooter et al.
(https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09),
licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude).
Logic unchanged; see FLAGS.md.
"""

import numpy as np


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

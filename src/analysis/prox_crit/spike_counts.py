"""Spike-count time series from spike times (binning).

NOT OWN WORK. Python translation of the auxiliary function ``get_spike_counts`` from ``demo/demo.mlx`` and ``demo/demo_crit_db.m``, prox_crit_toolkit, J. S. Sooter et al. (https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09), licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude). Logic unchanged except CHANGED[5]; see FLAGS.md.
"""

import numpy as np


def matlab_colon(a, d, b):
    """MATLAB's floating-point colon operator ``a:d:b`` (for d > 0), bit for bit.

    Why not ``np.arange(a, b + d/2, d)``? It gives the same edges up to ~1e-12, but here those last digits matter: spike times lie on a 1/20000 s sampling grid and bin edges t1 + k*delta_t often lie on the same grid, so many spikes sit *exactly* on an edge, and the rounding of the edge decides their bin. With np.arange, ~0.5% of the bins of the demo session differ from MATLAB by one spike (d2 changes in the 4th digit). This replica keeps the counts identical to MATLAB.

    MATLAB's algorithm (published by MathWorks as colonop.m): choose the number of steps n with a small tolerance, then fill the vector symmetrically, ``a + k*d`` from the left and ``last - k*d`` from the right.
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
    """Spike-count time series: number of spikes ``st`` in consecutive bins of width delta_t.

    Bin edges e_1 < ... < e_N are MATLAB's ``t1:delta_t:t2``; the N-1 bins are half-open, count_k = #{e_k <= st < e_{k+1}}. Spikes outside [e_1, e_N) are ignored.

    CHANGED[5]: MATLAB's ``histc`` returns one more value, count_N = #{st == e_N}, which is almost always 0. That spurious trailing sample is dropped here. Effect on the demo's d2 values: at most 0.07% (WAKE 1.0092 -> 1.0097, error bar 0.016).

    Returns
    -------
    x : (N-1,) float array
    """
    edges = matlab_colon(t1, delta_t, t2)
    if len(edges) < 2:
        return np.zeros(0)
    st = np.asarray(st, dtype=float).ravel()
    spikes = st[(st >= edges[0]) & (st < edges[-1])]
    bin_index = np.searchsorted(edges, spikes, side="right") - 1  # edges[k] <= spike < edges[k+1]
    return np.bincount(bin_index, minlength=len(edges) - 1).astype(float)

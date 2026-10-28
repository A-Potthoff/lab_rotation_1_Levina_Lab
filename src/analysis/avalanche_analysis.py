import warnings
import numpy as np

# =========================================================================== avalanches

def avalanches_from_activity(A, threshold=0):
    """
    Detect complete avalanches in one or more activity recordings.

    An avalanche is a maximal run of consecutive bins for which
    activity > threshold.

    Parameters
    ----------
    A : array, shape (T,) or (R, T)
        Population activity over time. Each row is processed as
        an independent recording.
    threshold : float, default=0
        Bins with activity <= threshold separate avalanches.
        Avalanche size is the sum of activity above threshold.

    Returns
    -------
    sizes : 1-D integer array
        Total excess activity in each complete avalanche.
    durations : 1-D integer array
        Number of bins in each complete avalanche.

    Notes
    -----
    Avalanches touching either recording boundary are discarded,
    because their full duration and size may be unknown.
    """
    A = np.atleast_2d(np.asarray(A))
    sizes, durs = [], []

    for row in A:
        active = row > threshold # True = this time bin belongs to an avalanche.

        # Pad with zeros so that transitions are detected even when activity starts or ends at a recording boundary.
        padded = np.r_[0, active.astype(np.int8), 0]
        # Difference between adjacent bins: +1 = avalanche starts; -1 = avalanche ends.
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)

        # Keep only avalanches that start after bin 0 and finish before the final bin. Boundary-touching runs might be incomplete.
        keep = (starts > 0) & (ends < len(row))
        starts = starts[keep]
        ends = ends[keep]

        if len(starts) == 0:
            continue

        contribution = np.where(active, row - threshold, 0) # activity - threshold if active, otherwise zero.

        excess = np.cumsum(np.r_[0, contribution])
        row_sizes = excess[ends] - excess[starts] # avoids a separate loop per avalanche.
        
        row_durs = ends - starts
        
        sizes.append(row_sizes)
        durs.append(row_durs)

    # No complete avalanches were found in any recording.
    if not sizes:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    # Combine results from all recordings into flat arrays.
    return (
        np.concatenate(sizes).astype(np.int64),
        np.concatenate(durs).astype(np.int64),
    )

 
 
def avalanches_sts_aa(N, sigma, n_avalanches, max_steps=10**5, seed=None,
                      p_max=1.0, refractory=False):
    """
    Separation of timescales (h -> 0), all-to-all: seed one spike, run until A = 0.
    All avalanches run in parallel; dead ones are dropped. Exact (population level, h = 0).
    Returns sizes, durations, truncated (bool: hit max_steps, size/duration are lower bounds).
    """
    rng = np.random.default_rng(seed)
    A = np.ones(n_avalanches, dtype=np.int64)
    sizes = np.ones(n_avalanches, dtype=np.int64)
    durs = np.ones(n_avalanches, dtype=np.int64)
    alive = np.arange(n_avalanches)
    for _ in range(max_steps - 1):
        if alive.size == 0:
            break
        p = np.minimum(sigma * A / N, p_max)
        A = rng.binomial(N - A if refractory else N, p)
        ongoing = A > 0
        alive, A = alive[ongoing], A[ongoing]
        sizes[alive] += A
        durs[alive] += 1
    truncated = np.zeros(n_avalanches, bool)
    truncated[alive] = True
    return sizes, durs, truncated
 
 
def avalanches_sts_er(W, n_avalanches, max_steps=10**5, seed=None, p_max=1.0, refractory=False):
    """
    Separation of timescales on a sparse network: seed one random neuron, run until silent.
    Event-driven: with h = 0 only targets of active neurons can fire, so each step costs
    O(active * K_out) instead of O(N). Exact for the PIF update rule.
    Returns sizes, durations, truncated.
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    Wc = W.tocsc()
    ip, ind, dat = Wc.indptr, Wc.indices, Wc.data
    sizes = np.zeros(n_avalanches, dtype=np.int64)
    durs = np.zeros(n_avalanches, dtype=np.int64)
    truncated = np.zeros(n_avalanches, bool)
    for a in range(n_avalanches):
        active = np.array([rng.integers(N)])
        size, d = 1, 1
        while d < max_steps:
            st = ip[active]
            ln = ip[active + 1] - st
            tot = ln.sum()
            if tot == 0:
                break
            offs = np.repeat(st - np.cumsum(ln) + ln, ln) + np.arange(tot)   # CSC entries of active cols
            targets, inv = np.unique(ind[offs], return_inverse=True)
            V = np.bincount(inv, weights=dat[offs])                          # input per target
            P = np.minimum(V, p_max)
            if refractory:
                P[np.isin(targets, active)] = 0.0
            active = targets[rng.random(targets.size) < P]
            if active.size == 0:
                break
            size += active.size
            d += 1
        else:
            truncated[a] = True
        sizes[a], durs[a] = size, d
    return sizes, durs, truncated



# Claude was motivated to write
# I am not motivated to check

# # =========================================================================== branching ratio
# def branching_ratio_lr(A):
#     """Conventional estimator: slope of the linear regression A_{t+1} ~ A_t."""
#     x, y = A[:-1].astype(float), A[1:].astype(float)
#     return np.cov(x, y, bias=True)[0, 1] / np.var(x)
 
 
# def branching_ratio_mr(A, k_max=100, fix_b=False):
#     """
#     Multistep regression (MR) estimator (Wilting & Priesemann 2018, Nat. Commun.).
#     r_k = slope of A_{t+k} ~ A_t decays as b * m^k; fitting the decay gives m, also under
#     subsampling (where b < 1 biases the conventional estimator but not m).
#     fix_b=True fixes b = 1 (valid only for fully observed activity); this removes the
#     b-m trade-off of the two-parameter fit and lowers the variance of m_hat.
#     Returns m_hat, tau_hat (steps), b_hat, ks, r_k.
#     """
#     from scipy.optimize import curve_fit
#     A = np.asarray(A, float)
#     ks = np.arange(1, k_max + 1)
#     rk = np.array([np.cov(A[:-k], A[k:], bias=True)[0, 1] / np.var(A[:-k]) for k in ks])
#     if fix_b:
#         (m,), _ = curve_fit(lambda k, m: m**k, ks, rk, p0=(0.9,), bounds=([0], [1.5]))
#         b = 1.0
#     else:
#         (b, m), _ = curve_fit(lambda k, b, m: b * m**k, ks, rk, p0=(rk[0], 0.9),
#                               bounds=([-np.inf, 0], [np.inf, 1.5]))
#     return m, -1 / np.log(m) if 0 < m < 1 else np.inf, b, ks, rk
 
 
# def logbinned_pdf(x, bins_per_decade=8):
#     """Normalized histogram on logarithmic bins (for heavy-tailed integer data, x >= 1)."""
#     x = np.asarray(x)
#     edges = np.unique(np.floor(np.logspace(0, np.log10(x.max() + 1), 
#                       int(bins_per_decade * np.log10(x.max() + 1)) + 2)).astype(np.int64))
#     counts, edges = np.histogram(x, bins=edges)
#     widths = np.diff(edges)                       # number of integers per bin
#     centers = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(min=1))
#     pdf = counts / (widths * len(x))
#     keep = counts > 0
#     return centers[keep], pdf[keep]
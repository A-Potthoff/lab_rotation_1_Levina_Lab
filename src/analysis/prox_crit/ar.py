"""Autoregressive (AR) models: fitting them to (possibly segmented) time series, and simulating them.

NOT OWN WORK. Python translation of ``yw_ar.m``, ``mle_ar.m`` and of ``runAR`` (from the demos) from prox_crit_toolkit, J. S. Sooter et al. (https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09), licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude). See FLAGS.md for flagged points.

AR(n) model, the object fitted everywhere in this package::

    x_t = sum_{k=1..n} kern_k * x_{t-k} + sigma * eps_t,   eps_t ~ N(0, 1)

``kern`` (the "history kernel" phi) is a length-n array; ``kern[k-1]`` is the coefficient of lag k.

Segments: a time series may consist of several separate pieces (e.g. all WAKE episodes of a session). They are treated as realizations of ONE AR process: one kernel and one sigma are fitted jointly, but the history of a sample never reaches into another segment.

Abbreviations: yw = Yule-Walker, mle = maximum-likelihood estimate, H = Hessian of the log-likelihood.
"""

import numpy as np
from scipy.signal import lfilter


def as_segments(x):
    """Return ``x`` as a list of 1-D float arrays, one per segment.

    A list/tuple whose elements are arrays is a list of separate segments (MATLAB: cell array ``{seg1, seg2}``). Anything else (a 1-D array, or a flat list of numbers) is a single time series. Each segment is flattened with ``ravel`` because MATLAB vectors arrive as (1, N) or (N, 1) arrays; never pass several series as one 2-D array, they would be strung together.
    """
    if (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(np.ndim(seg) >= 1 for seg in x)
    ):
        return [np.asarray(seg, dtype=float).ravel() for seg in x]
    return [np.asarray(x, dtype=float).ravel()]


def yw_ar(x, order):
    """Fit an AR model with the Yule-Walker method (method of moments).

    Solves the Yule-Walker equations ``Gamma @ kern = gamma``: gamma[lag-1] is the sample autocovariance at lag 1..order, Gamma the Toeplitz matrix Gamma[i, j] = gamma(|i-j|) with gamma(0) = 1. Then sigma^2 = 1 - gamma . kern (variance not explained by the history).

    Parameters
    ----------
    x : array or list of arrays
        Time series or list of segments. Must already be z-scored (mean 0, variance 1), as done in ``calc_db``; the diagonal of Gamma is set to 1 on that assumption.
    order : int
        AR model order n.

    Returns
    -------
    kern : (order,) array
        History kernel.
    sigma : float
        Noise standard deviation (complex if 1 - gamma.kern < 0, FLAG[8]).
    """
    # The original requires a MATLAB cell array here (unlike mle_ar); both now accept the same inputs.
    x = as_segments(x)

    # Autocovariance: average of x_k * x_{k+lag} over all pairs, pooled over all segments. Pairs never straddle a segment border.
    gamma = np.zeros(order)
    for lag in range(1, order + 1):
        total = 0.0
        counter = 0  # number of pairs (x_k, x_{k+lag}) used
        for seg in x:
            n_pairs = len(seg) - lag
            if n_pairs > 0:
                total += np.dot(seg[:n_pairs], seg[lag:])
                counter += n_pairs
        # FLAG[3]: divides by the number of pairs (N - lag), not by N. The 1/N estimator guarantees a positive-definite Gamma (hence a stable fit); this one does not.
        gamma[lag - 1] = np.float64(total) / counter

    # Toeplitz matrix of autocovariances, Gamma[i, j] = gamma(|i - j|).
    # FLAG[4]: diagonal fixed to 1 (unit variance), although calc_db z-scores with the N-1 std, so the actual lag-0 moment of the data is (N-1)/N.
    idx = np.arange(order)
    lag_matrix = np.abs(idx[:, None] - idx[None, :])
    Gamma = np.ones((order, order))
    off_diagonal = lag_matrix > 0
    Gamma[off_diagonal] = gamma[lag_matrix[off_diagonal] - 1]

    kern = np.linalg.solve(Gamma, gamma)
    # FLAG[8]: a negative argument gives a complex sigma, as MATLAB's sqrt does (np.sqrt would give nan).
    sigma = np.emath.sqrt(1 - gamma @ kern)
    return kern, sigma


def mle_ar(x, order):
    """Fit an AR model by (conditional) maximum likelihood, which for Gaussian noise is least squares.

    Regression ``Y = X @ kern + noise``; row t of the design matrix X holds the history (x_{t-1}, ..., x_{t-order}) of sample Y_t::

        kern  = (X'X)^-1 X'Y                     (minimizes the sum of squared prediction errors)
        sigma = sqrt(mean(residual^2))           (MLE, divides by T)
        H     = Hessian of the profile log-likelihood l(kern) = -T/2 log A(kern),  A = sum(residual^2):
                H = T (2 B B' / A^2 - X'X / A),  B = X' residual

    -H^-1 approximates the covariance of the kernel estimate; calc_db uses it for error bars.

    Parameters
    ----------
    x : array or list of arrays
        Time series or list of segments (each with its own history).
    order : int
        AR model order.

    Returns
    -------
    kern : (order,) array
    sigma : float
    H : (order, order) array
    """
    x = as_segments(x)
    T = sum(len(seg) for seg in x)  # total number of samples

    # Design matrix, one block per segment, stacked. Column k-1 holds lag k: X[t, k-1] = seg[t-k].
    # FLAG[5]: EVERY sample is a regression row, also the first `order` samples of each segment; their missing history is filled with zeros (= the mean, after z-scoring), and they count in T.
    X_blocks = []
    for seg in x:
        n = len(seg)
        X_seg = np.zeros((n, order))
        for k in range(1, order + 1):
            X_seg[k:, k - 1] = seg[: max(n - k, 0)]
        X_blocks.append(X_seg)
    X = np.vstack(X_blocks)
    Y = np.concatenate(x)

    kern = np.linalg.solve(
        X.T @ X, X.T @ Y
    )  # the original computes inv(X'X) X'Y explicitly; same result, solve is more stable
    resid = Y - X @ kern
    sigma = np.sqrt(np.mean(resid**2))

    A = np.sum(resid**2)
    B = X.T @ resid
    H = T * (2 * np.outer(B, B) / A**2 - (X.T @ X) / A)
    return kern, sigma, H


def run_ar(kern, sigma, T, rng=None, noise=None):
    """Simulate T steps of the AR model x_t = sum_k kern_k x_{t-k} + sigma eps_t.

    Starts from x = 0 before t = 1 (no burn-in), like the original loop. Implemented as the equivalent linear filter (scipy.signal.lfilter with zero initial state), which is exact and much faster than a Python loop.

    Parameters
    ----------
    kern : array
        History kernel (kern[k-1] = coefficient of lag k).
    sigma : float
        Noise standard deviation.
    T : int
        Number of time steps.
    rng : numpy.random.Generator, optional
        Random generator for the noise, e.g. ``np.random.default_rng(seed=0)``. MATLAB's randn stream cannot be reproduced, so simulated series differ from MATLAB's sample by sample (not in distribution).
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
    # lfilter solves  x_t - sum_k kern_k x_{t-k} = sigma * eps_t  for x
    return lfilter([1.0], np.r_[1.0, -kern], sigma * np.asarray(noise, dtype=float))

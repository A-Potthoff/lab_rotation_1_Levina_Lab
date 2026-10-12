"""Fitting autoregressive (AR) models to (possibly segmented) time series.

NOT OWN WORK. Python translation of ``yw_ar.m`` and ``mle_ar.m`` from
prox_crit_toolkit, J. S. Sooter et al.
(https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09),
licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude).
Logic unchanged; see FLAGS.md for flagged points.

AR(n) model (the object fitted everywhere in this package)::
    x_t = sum_{k=1..n} kern_k * x_{t-k} + sigma * eps_t,   eps_t ~ N(0, 1)

``kern`` ("history kernel", phi) is a length-n array; ``kern[k-1]`` is the
coefficient of lag k (Python is 0-based, MATLAB's ``kern(k)`` is 1-based).

Abbreviations: yw = Yule-Walker, mle = maximum-likelihood estimate,
H = Hessian of the log-likelihood.
"""

import numpy as np


def _as_segments(x):
    """Return ``x`` as a list of 1-D float arrays (MATLAB: cell array).

    A list/tuple whose elements are arrays is treated as separate segments
    (MATLAB ``x = {seg1, seg2}``); anything else (a 1-D array, or a flat list of numbers) is a single time series (MATLAB ``x = {x}``).
    """
    if (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(np.ndim(seg) >= 1 for seg in x)
    ):
        return [np.asarray(seg, dtype=float).ravel() for seg in x]
    return [np.asarray(x, dtype=float).ravel()]


def yw_ar(x, order):
    """Fit an AR model with the Yule-Walker method.

    Solves the Yule-Walker equations  Gamma @ kern = gamma, where
    gamma[lag-1] is the sample autocovariance at lag 1..order and Gamma is
    the Toeplitz matrix Gamma[i, j] = gamma(|i-j|) with gamma(0) = 1.
    Then sigma^2 = 1 - gamma . kern (residual variance of a unit-variance
    process).

    Parameters
    ----------
    x : list of 1-D arrays
        Time-series segments. Must already be z-scored (mean 0, variance 1), as done in ``calc_db`` (original comment: "x is already z-scored").
        FLAG[5c]: unlike ``mle_ar``, a single array is NOT wrapped into a list (MATLAB indexes ``x{j}`` and errors on a non-cell input; here a TypeError is raised for the same case).
    order : int
        AR model order n.

    Returns
    -------
    kern : (order,) array
        History kernel.
    sigma : float
        Noise standard deviation (complex if 1 - gamma.kern < 0, FLAG[14]).
    """
    if not isinstance(x, (list, tuple)):
        raise TypeError("yw_ar expects a list of segments (MATLAB cell array).")
    x = [np.asarray(seg, dtype=float).ravel() for seg in x]

    # Estimate autocovariance up to lag `order`: products x_k * x_{k+lag},
    # pooled over all segments (pairs never straddle segment borders).
    # gamma[lag-1] <-> MATLAB gamma(lag).
    gamma = np.zeros(order)
    for lag in range(1, order + 1):
        total = np.float64(0.0)
        cntr = 0  # number of pairs used
        for seg in x:
            n_pairs = len(seg) - lag  # MATLAB: k = 1:(length(x{j})-lag)
            if n_pairs > 0:
                total += np.dot(seg[:n_pairs], seg[lag:])
                cntr += n_pairs
        # FLAG[5a]: divides by the number of pairs (N - lag), not by N. The
        # 1/N estimator guarantees a positive-definite Gamma; this one does not.
        gamma[lag - 1] = total / cntr

    # Toeplitz matrix of autocovariances. FLAG[5b]: diagonal fixed to 1 (unit
    # variance), although calc_db z-scores with the N-1 std, so the actual
    # lag-0 moment of the data is (N-1)/N.
    idx = np.arange(order)
    lags = np.abs(idx[:, None] - idx[None, :])  # |i - j|, same in 0/1-based
    Gamma = np.ones((order, order))
    off = lags > 0
    Gamma[off] = gamma[lags[off] - 1]  # MATLAB: Gamma(i,j) = gamma(abs(i-j))

    kern = np.linalg.solve(Gamma, gamma)  # MATLAB: Gamma \ gamma(:)
    # FLAG[14]: MATLAB sqrt returns a complex number for a negative argument;
    # np.emath.sqrt reproduces that instead of returning nan.
    sigma = np.emath.sqrt(1 - gamma @ kern)
    return kern, sigma


def mle_ar(x, order):
    """Fit an AR model by (conditional) maximum likelihood = least squares.

    Regression  Y = X @ kern + noise, where row t of the design matrix X holds
    the history (x_{t-1}, ..., x_{t-order}) of sample Y_t:

        kern  = (X'X)^-1 X'Y
        sigma = sqrt(mean(residual^2))          (MLE, divides by T)
        H     = Hessian of the profile log-likelihood w.r.t. kern,
                l(kern) = -T/2 * log(A(kern)) + const,  A = sum(residual^2):
                H_jk = T * (2 B_j B_k / A^2 - (X'X)_jk / A),  B = X' residual

    Parameters
    ----------
    x : array or list of arrays
        Time series, or list of segments (each fitted with its own history).
    order : int
        AR model order.

    Returns
    -------
    kern : (order,) array
    sigma : float
    H : (order, order) array
        Hessian of the log-likelihood at ``kern`` (used for error bars of db).
    """
    x = _as_segments(x)  # MATLAB: if ~iscell(x); x = {x}; end
    T = sum(len(seg) for seg in x)  # total number of samples

    # Design matrix, built per segment and stacked (vectorized form of the
    # MATLAB double loop). Column k-1 holds lag k: X[t, k-1] = seg[t-k].
    # FLAG[6a]: EVERY sample is a regression row, also the first `order`
    # samples of each segment, whose missing history is zero-padded (MATLAB:
    # [fliplr(hist) zeros(1, order-length(hist))]); they also count in T.
    X_blocks = []
    for seg in x:
        n = len(seg)
        Xs = np.zeros((n, order))
        for k in range(1, order + 1):
            Xs[k:, k - 1] = seg[: max(n - k, 0)]
        X_blocks.append(Xs)
    X = np.vstack(X_blocks)  # rows are different samples
    Y = np.concatenate(x)

    # FLAG[6b]: explicit inverse (MATLAB inv(X'*X)*X'*Y, evaluated left to
    # right) instead of a linear solve; kept for faithfulness.
    kern = (np.linalg.inv(X.T @ X) @ X.T) @ Y
    resid = Y - X @ kern
    sigma = np.sqrt(np.mean(resid**2))

    A = np.sum(resid**2)
    B = X.T @ resid  # MATLAB: B(j) = sum((Y-X*kern).*X(:,j))
    H = T * (2 * np.outer(B, B) / A**2 - (X.T @ X) / A)
    return kern, sigma, H

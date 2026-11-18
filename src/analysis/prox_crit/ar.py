"""Autoregressive (AR) models: fitting them to (possibly segmented) time series. Simulation (``run_ar``) lives in src/models/autoregression/autoregression_model.py.

NOT OWN WORK. Python translation of ``yw_ar.m`` and ``mle_ar.m`` from prox_crit_toolkit, J. S. Sooter et al. (https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09), licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude). Both estimators deviate from the original where it is not exact for short segments (see the docstrings and NOTICE.md).

AR(n) model, the object fitted everywhere in this package::

    x_t = sum_{k=1..n} kern_k * x_{t-k} + sigma * eps_t,   eps_t ~ N(0, 1)

``kern`` (the "history kernel" phi) is a length-n array; ``kern[k-1]`` is the coefficient of lag k.

Segments: a time series may consist of several separate pieces (e.g. all WAKE episodes of a session). They are treated as realizations of ONE AR process: one kernel and one sigma are fitted jointly, but the history of a sample never reaches into another segment.

Abbreviations: yw = Yule-Walker, mle = maximum-likelihood estimate, H = Hessian of the log-likelihood.
"""

import numpy as np


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

    Solves the Yule-Walker equations ``Gamma @ kern = gamma``: gamma[lag-1] is the sample autocovariance at lag 1..order, Gamma the Toeplitz matrix Gamma[i, j] = gamma(|i-j|). Then sigma^2 = gamma(0) - gamma . kern (variance not explained by the history).

    Autocovariance: gamma(lag) = mean of x_t x_{t+lag} over all pairs within segments, ie the sum divided by the number of pairs (N - lag), for every lag including 0. Unbiased for each lag, also when pooling many short segments. Deviation from the original, which fixes gamma(0) = 1 instead of estimating it: after calc_db's z-scoring (std with N - 1) the data's gamma(0) is (N - 1)/N, so the effect is O(1/N), but this way all entries are estimated consistently and x need not have unit variance.

    Not chosen: dividing every lag by N (the "biased" estimator) makes Gamma positive definite, which guarantees a stable fit and a real sigma. But each segment then contributes (L - lag)/L of its true covariance (L = segment length), which, pooled over short segments, pulls the fit away from criticality (AR(1) 0.95 fitted as 0.86 from 10-sample windows). With N - lag, Gamma can be indefinite when the order is a sizeable fraction of the segment length (eg order 5, segments of 10-20 samples); the fit can then be explosive (calc_db returns NaN) or sigma complex.

    Parameters
    ----------
    x : array or list of arrays
        Time series or list of segments. Must be centered (mean 0, as after the z-scoring in ``calc_db``); the variance can be anything (gamma(0) is estimated).
    order : int
        AR model order n.

    Returns
    -------
    kern : (order,) array
        History kernel.
    sigma : float
        Noise standard deviation, in units of x (complex if gamma(0) - gamma.kern < 0, see NOTICE.md, known issues).
    """
    # The original requires a MATLAB cell array here (unlike mle_ar); both now accept the same inputs.
    x = as_segments(x)

    # gamma[lag] for lag = 0..order, pooled over all segments. Pairs never straddle a segment border.
    gamma = np.zeros(order + 1)
    for lag in range(order + 1):
        total = 0.0
        counter = 0  # number of pairs (x_k, x_{k+lag}) used
        for seg in x:
            n_pairs = len(seg) - lag
            if n_pairs > 0:
                total += np.dot(seg[:n_pairs], seg[lag:])
                counter += n_pairs
        gamma[lag] = total / counter

    # Toeplitz matrix of autocovariances, Gamma[i, j] = gamma(|i - j|).
    idx = np.arange(order)
    Gamma = gamma[np.abs(idx[:, None] - idx[None, :])]

    kern = np.linalg.solve(Gamma, gamma[1:])
    # Known issue (NOTICE.md): a negative argument gives a complex sigma, as MATLAB's sqrt does (np.sqrt would give nan).
    sigma = np.emath.sqrt(gamma[0] - gamma[1:] @ kern)
    return kern, sigma


def mle_ar(x, order):
    """Fit an AR model by conditional maximum likelihood, which for Gaussian noise is least squares.

    Regression ``Y = X @ kern + noise``; row t of the design matrix X holds the history (x_{t-1}, ..., x_{t-order}) of sample Y_t::

        kern  = (X'X)^-1 X'Y                     (minimizes the sum of squared prediction errors)
        sigma = sqrt(mean(residual^2))           (MLE, divides by T)
        H     = Hessian of the profile log-likelihood l(kern) = -T/2 log A(kern),  A = sum(residual^2):
                H = T (2 B B' / A^2 - X'X / A),  B = X' residual

    -H^-1 approximates the covariance of the kernel estimate; calc_db uses it for error bars.

    Conditional likelihood: the first ``order`` samples of each segment are conditioned on (they only enter as history), so T = sum over segments of (length - order). Deviation from the original, which also uses those first samples as regression targets with their missing history set to 0. That treats x = 0 before each segment start as observed, which biases kern towards 0 by O(order / segment length): negligible for long segments, not for short windows.

    Parameters
    ----------
    x : array or list of arrays
        Time series or list of segments (each with its own history). Segments with at most ``order`` samples contribute nothing.
    order : int
        AR model order.

    Returns
    -------
    kern : (order,) array
    sigma : float
    H : (order, order) array
    """
    x = as_segments(x)

    # Design matrix, one block per segment, stacked. Column k-1 holds lag k: X[row, k-1] = seg[t-k] for targets t = order..len-1.
    X_blocks, Y_blocks = [], []
    for seg in x:
        n_rows = len(seg) - order
        if n_rows <= 0:
            continue
        X_blocks.append(
            np.column_stack(
                [seg[order - k : order - k + n_rows] for k in range(1, order + 1)]
            )
        )
        Y_blocks.append(seg[order:])
    X = np.vstack(X_blocks)
    Y = np.concatenate(Y_blocks)
    T = len(Y)  # number of regression rows

    kern = np.linalg.solve(
        X.T @ X, X.T @ Y
    )  # the original computes inv(X'X) X'Y explicitly; same result, solve is more stable
    resid = Y - X @ kern
    sigma = np.sqrt(np.mean(resid**2))

    A = np.sum(resid**2)
    B = X.T @ resid
    H = T * (2 * np.outer(B, B) / A**2 - (X.T @ X) / A)
    return kern, sigma, H

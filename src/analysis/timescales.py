from scipy.signal import correlate
import numpy as np

def _area_until_threshold(acf, dt, threshold):
    """
    Trapezoidal area under a 1D `acf` from lag 0 to its first crossing of
    `threshold`, with the crossing located by linear interpolation.

    Returns (area, crossing_lag) in units of dt, or (nan, nan) if the
    threshold is never reached.
    """
    below = np.flatnonzero(acf <= threshold)
    if below.size == 0:
        return np.nan, np.nan

    k = below[0]
    if k == 0:
        return 0.0, 0.0

    y0, y1 = acf[k - 1], acf[k]  # y0 > threshold >= y1
    frac = (y0 - threshold) / (y0 - y1)  # in (0, 1]

    seg = acf[:k]
    area = seg.sum() - 0.5 * (seg[0] + seg[-1])  # full intervals 0..k-1
    area += 0.5 * (y0 + threshold) * frac  # partial interval to crossing

    return float(area * dt), float((k - 1 + frac) * dt)


def get_autocorrelation(A, max_lag, dt=1.0, average_acf=True):
    """
    Normalized autocorrelation and timescale measures.

    Parameters
    ----------
    A : array-like, shape (R, T) or (T,)
        R replicas of a time series, time along the last axis. Each replica
        is centered on its own mean and normalized so its ACF starts at 1.
    max_lag : int
        Maximum lag in samples, inclusive. Must satisfy 0 <= max_lag < T.
    dt : float, optional
        Sampling interval (e.g. 1/fs). Lags and measures are in these units.
    average_acf : bool, optional
        If True (default), average the per-replica ACFs and compute the
        measures once from the mean ACF. If False, compute them per replica.

    Returns
    -------
        (autocorr, lags, [INT, INT_01, ACW_0, ACW_01]). autocorr has shape
        (max_lag + 1,) if average_acf, else (R, max_lag + 1). Measures are
        floats if average_acf, else arrays of shape (R,). NaN where the
        threshold is not reached within max_lag.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.ndim != 2:
        raise ValueError("A must have shape (R, T) or (T,).")

    R, n = A.shape
    if R < 1 or n < 2:
        raise ValueError("A needs at least one replica of at least two samples.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values.")
    if isinstance(max_lag, (bool, np.bool_)) or not isinstance(
        max_lag, (int, np.integer)
    ):
        raise TypeError("max_lag must be an integer.")
    if not 0 <= max_lag < n:
        raise ValueError("max_lag must satisfy 0 <= max_lag < T.")
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt must be a positive finite number.")
    # Exact test; a variance == 0 check can miss constant signals because
    # mean subtraction leaves rounding residue.
    if np.any(np.ptp(A, axis=1) == 0):
        raise ValueError(
            "At least one replica is constant; its autocorrelation is undefined."
        )

    # scipy.signal.correlate has no axis argument, so correlate each replica.
    # Zero lag sits at index n - 1 of the full output.
    acf = np.empty((R, max_lag + 1))
    for r, row in enumerate(A):
        x = row - row.mean()
        acf[r] = correlate(x, x, mode="full", method="fft")[n - 1 : n + max_lag]

    acf /= acf[:, :1]  # normalize each replica so acf[:, 0] == 1
    lags = np.arange(max_lag + 1) * dt

    if average_acf:
        mean_acf = acf.mean(axis=0)
        INT, ACW_0 = _area_until_threshold(mean_acf, dt, 0.0)
        INT_01, ACW_01 = _area_until_threshold(mean_acf, dt, 0.1)
        return mean_acf, lags, [INT, INT_01, ACW_0, ACW_01]

    INT, ACW_0 = np.array([_area_until_threshold(a, dt, 0.0) for a in acf]).T
    INT_01, ACW_01 = np.array([_area_until_threshold(a, dt, 0.1) for a in acf]).T

    return acf, lags, [INT, INT_01, ACW_0, ACW_01]
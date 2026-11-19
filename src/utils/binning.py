import numpy as np


def binning(x, fuse=2, mode="sum"):
    """
    Rebin a time series into bins of width `fuse` time steps (sum or mean per bin), along the last axis.

    Each time step t is treated as the unit interval [t, t+1) with its value spread uniformly over it.
    Bin k covers [k*fuse, (k+1)*fuse), so for non-integer fuse a time step straddling a bin edge is split between both bins in proportion to the overlap (e.g. fuse=2.5: bin 0 = x0 + x1 + 0.5*x2, bin 1 = 0.5*x2 + x3 + x4).
    For integer fuse this is plain summing of `fuse` consecutive steps.
    The incomplete trailing bin is dropped, so there are floor(T / fuse) bins and every bin has width fuse.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (..., T), e.g. (T,) or (R, T).
    fuse : int or float, default=2
        Bin width in time steps. Must be positive (fuse < 1 upsamples).
    mode : {"sum", "mean"}, default="sum"
        "sum": total per bin (counts are conserved). "mean": sum / fuse, i.e. the average per original time step.

    Returns
    -------
    np.ndarray of float, shape (..., floor(T / fuse)).
    """
    x = np.asarray(x)
    if x.ndim == 0:
        raise ValueError("x must have at least one (time) axis.")
    if not fuse > 0:
        raise ValueError("fuse must be positive.")
    if mode not in ("sum", "mean"):
        raise ValueError("mode must be 'sum' or 'mean'.")

    T = x.shape[-1]
    # small tolerance s.t. eg T=3, fuse=0.3 (3 / 0.3 = 9.999...) still gives 10 bins
    n_bins = int(np.floor(T / fuse + 1e-9))
    if n_bins == 0:
        return np.empty(x.shape[:-1] + (0,))

    # cumulative sum C(t) = sum(x[..., :t]) at integer t, linearly interpolated at the bin edges;
    # linear interpolation within a step == spreading x[t] uniformly over [t, t+1)
    cumulative = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1, dtype=float)], axis=-1)
    edges = np.minimum(np.arange(n_bins + 1) * fuse, T)
    lower = np.minimum(np.floor(edges).astype(int), T - 1)
    fraction = edges - lower  # in [0, 1]; = 1 only at edge == T
    cumulative_at_edges = cumulative[..., lower] + fraction * x[..., lower]
    sums = np.diff(cumulative_at_edges, axis=-1)

    return sums if mode == "sum" else sums / fuse

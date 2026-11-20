"""Simulation of (fractionally integrated) autoregressive models. Fitting lives in src/analysis/prox_crit/ar.py, the exact power spectrum in src/analysis/power_spectrum.py."""

import numpy as np
from scipy.signal import fftconvolve
from statsmodels.tsa.arima_process import arma_generate_sample, lpol_fima


def simulate_arfima(phi=(), d=0.0, sigma=1.0, T=10_000, R=1, burn_in=None, seed=None):
    """
    Simulate R independent (fractional) AR series of length T:
        x_t = sum_k phi_k x_{t-k} + u_t
    Classical AR (d = 0): u_t = eps_t, white noise with standard deviation sigma.
    Fractional (d > 0): u_t = weighted sum of all past noise, sum_j psi_j eps_{t-j}, with slowly decaying weights psi_j ~ j^(d-1) (long memory).
    Same arguments as power_spectrum(phi=, d=, sigma=) in src/analysis/power_spectrum.py, which gives the exact PSD of this process.

    Examples:
        phi = [0.9]              AR(1), not critical
        phi = [1]  or  d = 1     random walk (brown noise); phi = [2, -1] or d = 2: integrated random walk
        d = 0.5                  pink noise (PSD ~ 1/f)
        0 < d < 0.5              stationary, with long memory
    Walks (critical AR, d >= 0.5) never settle into a stationary state: their variance keeps growing, so their statistics depend on burn_in (= how long ago they started).

    How:
    1. AR part: statsmodels' arma_generate_sample, started from x = 0.
    2. Fractional part: statsmodels' lpol_fima gives the weights psi_j; their weighted sum over past noise is computed as an FFT convolution (scipy, fast).
       statsmodels has no fractional simulator of its own. The order of steps 1 and 2 does not matter (both are linear filters).

    Parameters
    ----------
    phi : array-like
        AR coefficients, phi[k-1] = coefficient of lag k.
    d : float
        Fractional integration order, d >= 0.
    sigma : float
        Noise standard deviation.
    T, R : int
        Recorded time steps, independent replicas.
    burn_in : int, optional
        Steps simulated before the T recorded ones and discarded; the simulation starts from x = 0. Default T, so the slowest part of the long memory (lags up to ~T) is built up before recording. burn_in = 0 reproduces Sooter's runAR.
    seed : int or numpy.random.Generator, optional

    Returns
    -------
    x : (R, T) float array
    """
    T, R = int(T), int(R)
    burn_in = T if burn_in is None else int(burn_in)
    n_steps = burn_in + T
    rng = np.random.default_rng(seed=seed)
    # arma_generate_sample takes the lag polynomials incl. lag 0: ar = phi(L) = [1, -phi_1, ...], ma = [1] (no MA part)
    x = arma_generate_sample(
        ar=np.r_[1.0, -np.asarray(phi, dtype=float)],
        ma=[1.0],
        nsample=(R, n_steps),
        scale=sigma,
        distrvs=rng.standard_normal,
        axis=-1,
    )
    if d != 0:
        psi = lpol_fima(d, n=n_steps)  # psi_0 = 1, ..., psi_{n_steps - 1}; all ones for d = 1 (cumulative sum)
        x = fftconvolve(x, psi[None, :], axes=-1)[:, :n_steps]  # x_t <- sum_{k <= t} psi_k x_{t-k}
    return x[:, burn_in:]

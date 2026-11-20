"""Power spectrum and Hurst exponent: estimated from a time series, or exact (theoretical) for an AR model.

Examples
--------
    from src.analysis.power_spectrum import power_spectrum, plot_power_spectrum

    spec = power_spectrum(x)                        # estimate from data, x: (T,) or (R, T)
    spec = power_spectrum(phi=[0.9])                # exact PSD of the AR(1) x_t = 0.9 x_{t-1} + eps_t
    spec = power_spectrum(d=0.5)                    # exact PSD of pink noise
    spec.freqs, spec.psd, spec.beta, spec.hurst

    ax = plot_power_spectrum(power_spectrum(x), label="data")
    plot_power_spectrum(power_spectrum(d=0.5), ax=ax, label="exact, d = 0.5")

Conventions
-----------
Frequency f in cycles per time step, from 0 to 0.5 (the fastest oscillation a sampled series can show: up, down, up, ...).
PSD (power spectral density): how much of the variance sits at each frequency; summed (integrated) over all frequencies it gives the variance. Estimate and exact PSD have the same units, so they can be overlaid.
Model arguments phi, d, sigma: as in simulate_arfima in src/models/autoregression.
"""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import freqz, welch


class Spectrum(NamedTuple):
    freqs: np.ndarray  # (n_freqs,) frequency f, cycles/step
    psd: np.ndarray  # (n_freqs,) power spectral density
    beta: float  # spectral exponent: psd ~ f^(-beta) in the fit band
    hurst: float  # (beta + 1) / 2, see power_spectrum
    fit_freqs: np.ndarray  # frequencies in the fit band
    fit_psd: np.ndarray  # fitted power law on fit_freqs, for plotting


def power_spectrum(x=None, phi=(), d=0.0, sigma=1.0, freqs=None, nperseg=None, fit_band=(None, 0.05)):
    """
    Power spectral density (PSD) and Hurst exponent, either estimated from a time series x, or exact for an AR model.

    Give either x (estimate) or model parameters phi, d (exact), not both; neither gives white noise.
    Classical AR: phi only. Fractional models: d (optionally with phi). d = 0.5 is pink noise, d = 1 a random walk.

    Parameters
    ----------
    x : (T,) or (R, T) array, optional
        Time series; with R replicas the spectra are averaged.
    phi : array-like
        AR coefficients, phi[k-1] = coefficient of lag k.
    d : float
        Fractional integration order.
    sigma : float
        Noise standard deviation of the model; scales the exact PSD only.
    freqs : array, optional
        Frequencies for the exact PSD. Default: 500 log-spaced values in [1e-4, 0.5].
    nperseg : int, optional
        Segment length for the estimate (Welch's method: cut x into overlapping segments, take the spectrum of each, average them).
        Default T // 8. Longer segments resolve slower frequencies (lowest: 1 / nperseg), but fewer segments are averaged, so the estimate is noisier.
    fit_band : (f_min, f_max)
        Frequency band of the power-law fit; f_min = None: lowest available frequency.
        For models that are not critical, the spectrum is a power law only over part of the frequencies, so beta depends on this band.

    Returns
    -------
    Spectrum(freqs, psd, beta, hurst, fit_freqs, fit_psd), a named tuple:
        freqs, psd: the spectrum
        beta: spectral exponent of the power-law fit psd ~ f^(-beta); 0 white noise, 1 pink noise, 2 random walk
        hurst: Hurst exponent (beta + 1) / 2; 0.5 white noise, 1 pink noise, 1.5 random walk (the convention of DFA). For random walks and steeper (beta > 1), the textbook Hurst exponent is hurst - 1.
        fit_freqs, fit_psd: the fitted power law over the fit band, for plotting
    """
    if x is not None and (len(phi) > 0 or d != 0):
        raise ValueError("give either x (estimate) or phi, d (exact model), not both")

    if x is not None:
        x = np.atleast_2d(np.asarray(x, dtype=float))  # (R, T)
        nperseg = x.shape[-1] // 8 if nperseg is None else nperseg
        # fs = 1: f in cycles/step; psd (R, n_freqs)
        # linear detrend per segment: the in-segment trend of non-stationary walks leaks into low f through the window (integrated random walk: lowest bins x100, beta 4.6 instead of 4 with mean removal only); negligible for stationary series
        freqs, psd = welch(x, nperseg=nperseg, detrend="linear", axis=-1)
        freqs, psd = freqs[1:], psd.mean(axis=0)[1:]  # average replicas; drop f = 0, where the mean was removed
    else:
        freqs = np.logspace(-4, np.log10(0.5), 500) if freqs is None else np.asarray(freqs, dtype=float)
        # frequency response of the AR model (scipy), times the extra 1 / f^(2d)-like factor of the fractional part
        _, response = freqz(b=1.0, a=np.r_[1.0, -np.asarray(phi, dtype=float)], worN=freqs, fs=1.0)
        psd = 2 * sigma**2 * np.abs(response) ** 2 * (2 * np.sin(np.pi * freqs)) ** (-2 * d)  # factor 2: same scaling as Welch

    f_min = freqs[0] if fit_band[0] is None else fit_band[0]
    in_band = (freqs >= f_min) & (freqs <= fit_band[1])
    beta, amplitude = _fit_power_law(freqs[in_band], psd[in_band])
    return Spectrum(
        freqs=freqs,
        psd=psd,
        beta=beta,
        hurst=(beta + 1) / 2,
        fit_freqs=freqs[in_band],
        fit_psd=amplitude * freqs[in_band] ** (-beta),
    )


def _fit_power_law(freqs, psd):
    """
    Fit the power law psd = amplitude * f^(-beta) by maximum likelihood.

    Why not a straight line through log-log: it is a cruder estimator (least squares assumes the scatter in log-log is Gaussian and equal everywhere, which it is not for spectra). Maximum likelihood uses the known statistics of spectral estimates (the Whittle likelihood).
    The amplitude has a closed-form best value for each beta, so only beta is searched numerically (scipy's minimize_scalar).
    Each frequency counts with the spacing around it, so an exact PSD on a log grid gets the same fit as an estimate on Welch's even grid would converge to.

    Parameters
    ----------
    freqs, psd : (n,) arrays
        Frequencies and PSD values to fit (already restricted to the fit band).

    Returns
    -------
    beta : float
        Spectral exponent.
    amplitude : float
        Prefactor, so that amplitude * freqs^(-beta) is the fitted line.
    """
    weights = np.gradient(freqs)  # spacing around each frequency; all equal on an even grid

    def neg_log_likelihood(beta):  # best amplitude for this beta already inserted
        return np.log(np.average(psd * freqs**beta, weights=weights)) - beta * np.average(np.log(freqs), weights=weights)

    beta = minimize_scalar(neg_log_likelihood, bounds=(-2.0, 8.0), method="bounded").x
    amplitude = np.average(psd * freqs**beta, weights=weights)
    return beta, amplitude


def plot_power_spectrum(spectrum, ax=None, label=None, color=None, show_fit=True, **plot_kwargs):
    """
    Plot a PSD on log-log axes, optionally with its power-law fit (dashed, over the fit band). beta and hurst go into the legend.
    Call repeatedly with the same ax to overlay spectra (eg estimate and exact).

    Parameters
    ----------
    spectrum : Spectrum
        Output of power_spectrum.
    ax : matplotlib Axes, optional
        Axes to draw on; default: a new figure.
    label : str, optional
        Legend label of the PSD line; beta and hurst are appended.
    color : optional
        Colour of the PSD and the fit; default: next colour of the cycle.
    show_fit : bool
        Draw the fitted power law as a dashed line (legend entry "power-law fit").
    **plot_kwargs
        Passed on to the PSD line, eg linewidth, alpha.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    fit_info = rf"$\beta$ = {spectrum.beta:.2f}, H = {spectrum.hurst:.2f}"
    (line,) = ax.loglog(
        spectrum.freqs,
        spectrum.psd,
        color=color,
        label=f"{label} ({fit_info})" if label else fit_info,
        **{"linewidth": 1, **plot_kwargs},
    )
    if show_fit:
        ax.loglog(
            spectrum.fit_freqs, spectrum.fit_psd, color=line.get_color(), linestyle="--", linewidth=1.5, label="power-law fit"
        )
    ax.set(xlabel="Frequency f (cycles/step)", ylabel="PSD")
    ax.legend(frameon=False)
    return ax

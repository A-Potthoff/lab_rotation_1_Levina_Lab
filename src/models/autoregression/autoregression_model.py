import numpy as np
# import matplotlib.pyplot as plt  # unused
from scipy.signal import lfilter
from scipy.special import comb

def get_ar_coeffs(m):
    """
    Returns AR coefficients phi = [phi_1, ..., phi_m] for (1 - L)^m x_t = e_t.
    By binomial expansion: phi_k = (-1)^(k+1) * comb(m, k) for k = 1..m.
    """
    if m == 0:
        return np.array([])
    k = np.arange(1, m + 1)
    return ((-1) ** (k + 1)) * comb(m, k, exact=False)


def verify_moment_conditions(max_m=3):
    """
    Taylor expansion of Leftover A(w) = 1 - sum_{k=1}^m phi_k e^{-i w k} around w=0:
    - Level (w^0):     1 - sum(phi_k) = 0        <=> sum(phi_k) = 1
    - Slope (w^1):     sum(k * phi_k) = 0
    - Curvature (w^2): sum(k^2 * phi_k) = 0
    - Jerk (w^3):      sum(k^3 * phi_k) = 0
    """
    print("=" * 78)
    print("ALGEBRAIC DIAGNOSTIC: MOMENT CONDITIONS ON ECHO COEFFICIENTS (phi_k)")
    print("=" * 78)
    header = (f"{'m':<3} | {'phi coefficients':<18} | {'Sum(phi)=1?':<12} | "
              f"{'Sum(k*phi)=0?':<14} | {'Sum(k^2*phi)=0?':<16} | {'Sum(k^3*phi)=0?'}")
    print(header)
    print("-" * 78)
    
    for m in range(max_m + 1):
        phi = get_ar_coeffs(m)
        k = np.arange(1, m + 1)
        m0 = np.sum(phi) if m > 0 else 0.0
        m1 = np.sum(k * phi) if m > 0 else 0.0
        m2 = np.sum((k ** 2) * phi) if m > 0 else 0.0
        m3 = np.sum((k ** 3) * phi) if m > 0 else 0.0
        
        phi_str = str(phi.astype(int).tolist())
        print(f"{m:<3} | {phi_str:<18} | {m0:<12.1f} | "
              f"{m1:<14.1f} | {m2:<16.1f} | {m3:<14.1f}")
    print("=" * 78 + "\n")

def simulate_ar(phi, n_steps, init_history, noise_scale=1.0, rng=None):
    """
    Simulates x_t = sum_{k=1}^p phi_k x_{t-k} + e_t.
    Returns full array of length (len(init_history) + n_steps).
    """
    p = len(phi)
    n_init = len(init_history)
    if n_init < p:
        raise ValueError(f"Need at least {p} initial values, got {n_init}.")
        
    x = np.zeros(n_init + n_steps, dtype=float)
    x[:n_init] = init_history
    
    if noise_scale > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        e = rng.normal(0.0, noise_scale, size=n_init + n_steps)
    else:
        e = np.zeros(n_init + n_steps)
        
    for t in range(n_init, n_init + n_steps):
        if p > 0:
            # x[t-p:t][::-1] orders lags as [x_{t-1}, x_{t-2}, ..., x_{t-p}]
            x[t] = np.dot(phi, x[t - p:t][::-1]) + e[t]
        else:
            x[t] = e[t]
            
    return x

def exact_theoretical_std(m, n_steps, noise_scale=1.0):
    """
    Computes the exact analytical standard deviation of x_t at each step t >= 0
    using the Green's function h_j = comb(j + m - 1, m - 1):
    Var(x_t) = sigma^2 * sum_{j=0}^{t} h_j^2
    """
    j = np.arange(n_steps)
    if m == 0:
        return np.full(n_steps, noise_scale)
    h_j = comb(j + m - 1, m - 1)
    var_t = (noise_scale ** 2) * np.cumsum(h_j ** 2)
    return np.sqrt(var_t)


def run_ar(kern, sigma, T, rng=None, noise=None):
    """Simulate T steps of the AR model x_t = sum_k kern_k x_{t-k} + sigma eps_t.

    NOT OWN WORK. Python translation of ``runAR`` (from the demos of prox_crit_toolkit, J. S. Sooter et al., https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09), licensed CC BY-NC 4.0, unlike the rest of this file. Translated by Andre Potthoff (with Claude); see src/analysis/prox_crit/NOTICE.md.

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

"""prox_crit: proximity to criticality d_b of a time series (Sooter et al.).

NOT OWN WORK. Python translation of the MATLAB prox_crit_toolkit by J. S. Sooter et al., "Defining and measuring proximity to criticality", bioRxiv (2025), doi:10.1101/2025.08.03.668332. Source: https://github.com/sam-sooter/prox_crit_toolkit (fetched 2026-10-09); the original is kept in src/external/proxy_crit_toolkit_sam_sooter_2026_10_09/. License: CC BY-NC 4.0 (NOT the MIT license of the surrounding repository). Translated by Andre Potthoff (with Claude); see NOTICE.md (file/function mapping) and FLAGS.md (flagged issues and the marked CHANGED items).

Idea: fit an AR(n) model to the data; d_b is the smallest KL-divergence rate from that model to any AR(n) model that is critical of type b (characteristic polynomial with a b/2-fold root at z = 1). Units: bits/s. "How distinguishable is this system from a system at criticality?"

Modules: distance (calc_db, the entry point), ar (fit and simulate AR models), manifold (geometry of the critical manifold), kl_rate (the KL-rate formula), plotting (QC plot), spike_counts (binning spike times).

Glossary of abbreviations used throughout:

    db           distance (proximity) to criticality of type b, bits/s
    sddb         standard deviation (error bar) of db
    b, beta      type/order of criticality (2, 4, 6, ...); M_b = critical manifold
    kern         history kernel phi of the fitted AR model (kern[k-1] = lag k)
    kernc        kernel of the closest critical model (phi_c); "reduced" = its first n - b/2 entries
    kernw        Fourier transform of a kernel (w = omega)
    sigma        noise std of the fitted model
    sigma_ratio  noise std of the critical model relative to sigma
    klr          KL-divergence rate
    dbv1         "version 1" distance: plain L2 distance to M_b (starting point)
    yw / mle     Yule-Walker / maximum-likelihood AR fit
    H            Hessian of the log-likelihood (for error bars)
    con          constraint function of the optimizer
    QC           quality control

Usage::

    from src.analysis.prox_crit import calc_db
    db, sddb, kern, sigma, H, kernc, exit_status = calc_db(x, order=5, delta_t=0.04, b=2, fit_method="MaxLikelihood", with_err_bars=True)
"""

from .ar import as_segments, mle_ar, run_ar, yw_ar
from .distance import DbResult, calc_db, minimize_klr_b
from .kl_rate import get_kernw, get_klr_b, get_sigma_ratio, inv_spectrum
from .manifold import (
    ar_roots,
    beta_lift,
    beta_project,
    calc_dbv1,
    con,
    explosive,
    half_order,
)
from .spike_counts import get_spike_counts, matlab_colon

__all__ = [
    "calc_db",
    "DbResult",
    "minimize_klr_b",
    "as_segments",
    "yw_ar",
    "mle_ar",
    "run_ar",
    "ar_roots",
    "half_order",
    "calc_dbv1",
    "beta_lift",
    "beta_project",
    "con",
    "explosive",
    "get_kernw",
    "inv_spectrum",
    "get_sigma_ratio",
    "get_klr_b",
    "get_spike_counts",
    "matlab_colon",
]

"""prox_crit: proximity to criticality d_b of a time series (Sooter et al.).

NOT OWN WORK. Python translation of the MATLAB prox_crit_toolkit by
J. S. Sooter et al., "Defining and measuring proximity to criticality",
bioRxiv (2025), doi:10.1101/2025.08.03.668332.
Source: https://github.com/sam-sooter/prox_crit_toolkit (fetched 2026-10-09),
original kept in src/external/proxy_crit_toolkit_sam_sooter_2026_10_09/.
License: CC BY-NC 4.0 (NOT the MIT license of the surrounding repository).
Translated by Andre Potthoff (with Claude); see NOTICE.md (file/function
mapping) and FLAGS.md (flagged issues and the few marked CHANGED items).

Idea: fit an AR(n) model to the data; d_b is the smallest KL-divergence rate
from that model to any AR(n) model that is critical of type b (characteristic
polynomial with a b/2-fold root at z = 1). Units: bits/s. "How
distinguishable is this system from a system at criticality?"

Glossary of abbreviations used throughout:

    db       distance (proximity) to criticality of type b, bits/s
    sddb     standard deviation (error bar) of db
    b, beta  type/order of criticality (2, 4, 6, ...); M_b = critical manifold
    kern     history kernel phi of the fitted AR model (kern[k-1] = lag k)
    kernc    kernel of the closest critical model (phi_c); "reduced" = first
             n - b/2 entries
    kernw    Fourier transform of a kernel (w = omega)
    sigma(c) noise std of the fitted (critical) model
    klr      KL-divergence rate
    dbv1     "version 1" distance: plain L2 distance to M_b (start point)
    yw / mle Yule-Walker / maximum-likelihood AR fit
    H        Hessian of the log-likelihood (for error bars)
    con      constraint function of the optimizer
    QC       quality control

Usage::

    from src.analysis.prox_crit import calc_db
    db, sddb, kern, sigma, H, kernc, exit_status = calc_db(
        x, order=5, delta_t=0.04, b=2, fit_method="MaxLikelihood",
        with_err_bars=True)
"""

from .ar_fit import mle_ar, yw_ar
from .demo_aux import get_spike_counts, matlab_colon, run_ar
from .distance import DbResult, calc_db, minimize_klr_b
from .kl_rate import get_kernw, get_klr_b, get_sigmac
from .manifold import beta_lift, beta_project, calc_dbv1, con, explosive

__all__ = [
    "calc_db",
    "DbResult",
    "minimize_klr_b",
    "yw_ar",
    "mle_ar",
    "calc_dbv1",
    "beta_lift",
    "beta_project",
    "con",
    "explosive",
    "get_kernw",
    "get_sigmac",
    "get_klr_b",
    "run_ar",
    "get_spike_counts",
    "matlab_colon",
]

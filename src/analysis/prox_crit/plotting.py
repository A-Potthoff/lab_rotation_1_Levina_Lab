"""Quality-control (QC) plot of the d_b optimization.

NOT OWN WORK. Python translation of the local function ``QC_plot`` from
``calc_db.m``, prox_crit_toolkit, J. S. Sooter et al.
(https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09),
licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude).
Logic unchanged; see FLAGS.md.

What it shows (from the original tutorial): one panel per free component of
the critical kernel. Each panel scans that component around the optimum,
keeping the others fixed, and plots the KL rate to the fitted model. The
optimization succeeded if the black lines (optimum, db) meet at the visual
minimum, at a cyan (stable) rather than red (explosive) point.
"""

import matplotlib.pyplot as plt
import numpy as np

from .kl_rate import get_klr_b
from .manifold import beta_lift, explosive, half_order


def qc_plot(kern, kernc, db, b, delta_t):
    """Plot KL rate (bits/s) along each free coordinate of kernc.

    Parameters
    ----------
    kern : (n,) array
        Fitted kernel.
    kernc : (n,) array
        Full closest critical kernel (as returned by calc_db).
    db : float
        d_b in bits/s (horizontal line).
    b : int
        Type of criticality.
    delta_t : float
        Time step in s (for the unit conversion to bits/s).

    Returns
    -------
    matplotlib.figure.Figure
    """
    kern = np.asarray(kern, dtype=float).ravel()
    kernc = np.asarray(kernc, dtype=float).ravel()
    dim = len(kern) - half_order(b)  # number of free (reduced) coordinates

    fig = plt.figure(figsize=(3.5 * max(dim, 1), 3))
    for i in range(dim):  # MATLAB i = 1..dim -> component i+1 in the label
        ax = fig.add_subplot(1, dim, i + 1)
        # FLAG[7b]: hard-coded scan range of +-0.2 around the optimum.
        xx = np.linspace(kernc[i] - 0.2, kernc[i] + 0.2, 100)
        klr = np.zeros_like(xx)
        expl = np.zeros_like(xx, dtype=bool)
        for j in range(len(xx)):
            kernc_temp = kernc[:dim].copy()
            kernc_temp[i] = xx[j]
            kernc_temp = beta_lift(kernc_temp, b)
            # FLAG[7a]: uses 'trapz' while the optimization used 'integral'.
            # 1000 trapezoid points under-resolve the log singularity at w = 0,
            # so the curve lies ~1% BELOW the db line even at the optimum.
            klr[j] = (
                np.log2(np.e) / delta_t * get_klr_b(kern, kernc_temp[:dim], b, "trapz")
            )
            expl[j] = explosive(kernc_temp)
        ax.plot(xx[expl], klr[expl], ".r", markersize=10)
        ax.plot(xx[~expl], klr[~expl], ".c", markersize=10)

        ax.axvline(kernc[i], color="k", linewidth=2)
        ax.axhline(db, color="k", linewidth=2)
        ax.tick_params(labelsize=12)
        ax.set_xlabel(rf"$(\phi_c)_{i + 1}$", fontsize=17)
        ax.set_ylabel("KL rate (bits/s)", fontsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.set_ylim(0, ax.get_ylim()[1])
    fig.tight_layout()
    return fig

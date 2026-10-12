"""Quality-control (QC) plot of the d_b optimization.

NOT OWN WORK. Python translation of the local function ``QC_plot`` from ``calc_db.m``, prox_crit_toolkit, J. S. Sooter et al. (https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09), licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude). See FLAGS.md.

What it shows (from the original tutorial): one panel per free component of the critical kernel. Each panel scans that component around the optimum, keeping the others fixed, and plots the KL rate to the fitted model. The optimization succeeded if the black lines (optimum, db) meet at the visual minimum, at a cyan (stable) rather than red (explosive) point.
"""

import matplotlib.pyplot as plt
import numpy as np

from .kl_rate import get_klr_b
from .manifold import beta_lift, explosive, half_order


def qc_plot(kern, kernc, db, b, delta_t, scan_width=0.2):
    """Plot the KL rate (bits/s) along each free coordinate of kernc.

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
        Time step in s (for the conversion to bits/s).
    scan_width : float, default 0.2
        Each coordinate is scanned over kernc_i +- scan_width (fixed to 0.2 in the original).

    Returns
    -------
    matplotlib.figure.Figure
    """
    kern = np.asarray(kern, dtype=float).ravel()
    kernc = np.asarray(kernc, dtype=float).ravel()
    n_free = len(kern) - half_order(b)  # number of free (reduced) coordinates

    fig = plt.figure(figsize=(3.5 * max(n_free, 1), 3))
    for i in range(n_free):
        ax = fig.add_subplot(1, n_free, i + 1)
        scan = np.linspace(kernc[i] - scan_width, kernc[i] + scan_width, 100)
        klr = np.zeros_like(scan)
        is_explosive = np.zeros_like(scan, dtype=bool)
        for j, value in enumerate(scan):
            kernc_scan = kernc[:n_free].copy()
            kernc_scan[i] = value
            # FLAG[6]: 'trapz' here, while the optimization used 'integral'. 1000 trapezoid points under-resolve the log singularity at w = 0, so the curve lies ~1% BELOW the db line even at the optimum; judge the location of the minimum, not its height.
            klr[j] = np.log2(np.e) / delta_t * get_klr_b(kern, kernc_scan, b, "trapz")
            is_explosive[j] = explosive(beta_lift(kernc_scan, b))
        ax.plot(scan[is_explosive], klr[is_explosive], ".r", markersize=10)
        ax.plot(scan[~is_explosive], klr[~is_explosive], ".c", markersize=10)

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

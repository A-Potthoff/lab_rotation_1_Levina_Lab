"""
Shared pieces of the confound notebooks (notebooks/prox_crit_confounds/): d2 of the all-to-all BN vs distance to criticality, swept over one covariate (bin width, model order, N, T, recorded fraction, ...).
Only the parts that are identical in all notebooks live here; each notebook keeps its own parameters and generation loop.
"""

import matplotlib.pyplot as plt
import numpy as np

# calc_db settings shared by all confound notebooks; override single entries with eg dict(CALC_DB_DEFAULTS, order=8)
CALC_DB_DEFAULTS = dict(
    order=5,
    delta_t=1,
    b=2,
    fit_method="MaxLikelihood",
    metric="kl_rate",
    optimizer="SLSQP",
    with_err_bars=False,  # not needed here, and they only create warnings
    with_qc=False,
)


def plot_d2_sweep(distances, db_results, values, legend_title, title):
    """
    d2 vs distance to criticality, one colour per value of the swept covariate, plus the theory line w/o confound.

    db_results[i][j][r] = DbResult for distances[i], values[j], replica r.
    Returns fig, ax, so follow-up plots can draw on top.
    """
    # db[i, j, r] = d2 for distances[i], values[j], replica r
    db = np.array(
        [[[result.db for result in replicas] for replicas in row] for row in db_results]
    )

    # one colour per value: dark = first value, light = last value
    # (0.85 instead of 1 skips the pale yellow end of the colour map)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(values)))

    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    for j, value in enumerate(values):
        # every replica as a dot
        for r in range(db.shape[2]):
            ax.scatter(distances, db[:, j, r], color=colors[j], alpha=0.5, s=10)

        # mean over replicas as a thin line (this line also gives the legend entry)
        mean_over_replicas = np.nanmean(db[:, j, :], axis=1)
        ax.plot(
            distances, mean_over_replicas, color=colors[j], linewidth=1, label=f"{value:g}"
        )

    # theory w/o confound: d2 of the true AR(1) kernel [1 - eps], small-eps limit (see prox_crit_branching_network.ipynb)
    ax.plot(
        distances,
        distances / (4 * np.log(2)),
        color="tab:gray",
        linestyle="--",
        linewidth=2,
        label=r"$\epsilon / (4 \ln 2)$",
    )

    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"Distance to criticality $\epsilon = 1 - \sigma_{{eff}}$",
        ylabel=r"$d_2$ (bits/step)",
        title=title,
    )
    ax.legend(title=legend_title, frameon=False)
    ax.grid(axis="y", alpha=0.5, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax

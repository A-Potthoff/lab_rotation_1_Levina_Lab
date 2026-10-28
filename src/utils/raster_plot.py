from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

import numpy as np
from numpy.typing import ArrayLike, NDArray
from collections.abc import Callable

def raster_plot(
    recording: ArrayLike,
    time_axis: ArrayLike | None = None,
    *,  # forces that all following arguments must be passed as keyword arguments
    ax=None,
    sort: str | ArrayLike | Callable | None = None,
    descending: bool = False,
    activity_threshold: float = 0.5,
    cmap: str | None = None,
    spike_color: str = "black",
    background_color: str = "white",
    marginal_color: str = "tab:blue",
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = "Neuron",
    show_marginals: bool = False,
    rasterized: bool = True,  # flattens the data layer into a single image, keeping pdfs small
):
    """Plot a neuron-by-time activity matrix as a raster plot, optionally with marginals.

    The recording is binarised (`activity >= activity_threshold`) and drawn with `imshow`, one row per neuron and one column per time step.
    Row k (counted from the top, starting at 1) sits at y = k, column i is centred on `time_axis[i]`.

    Parameters
    ----------
    recording : array_like, shape (n_neurons, n_times)
        Activity matrix. Values >= `activity_threshold` count as a spike.
    time_axis : array_like, shape (n_times,), optional
        Time stamp of each column. Assumed to be uniformly spaced. Defaults to `0, 1, ..., n_times - 1`.
    ax : matplotlib Axes, optional
        Axes to draw the raster into. A new figure is created if None. With `show_marginals=True`, the marginal axes are carved out of this axes' space.
    sort : {None, "firing_rate", "similarity"}, array_like or callable, optional
        Ordering of the neurons (rows):
        - None: keep the original order.
        - "firing_rate": ascending mean activity per neuron.
        - "similarity": leaf order of a Ward hierarchical clustering on pairwise correlation distances, so neurons with similar time courses end up next to each other. Costs O(N^2 T).
        - array_like of length n_neurons: an explicit permutation.
        - callable: `sort(activity) -> permutation`, receives the raw (non-thresholded) recording.
    descending : bool
        Reverse the ordering, e.g. highest firing rate on top.
    activity_threshold : float
        Values at or above this count as active.
    cmap : str or Colormap, optional
        Overrides `spike_color` / `background_color`.
    spike_color, background_color : str
        Colours of active / inactive bins when `cmap` is None.
    marginal_color : str
        Fill colour of the marginal bar plots.
    title, xlabel, ylabel : str
        Axis annotations. With marginals, the title is placed above the time marginal.
    show_marginals : bool
        Add two marginal bar plots:
        - right: per-neuron activity rate, i.e. fraction of time steps in which the neuron is active (aligned with the raster rows, after sorting).
        - top: population activity, i.e. fraction of neurons active at each time step (aligned with the raster columns).
        Both are drawn with `stairs`, which stays cheap and readable even for very long recordings, unlike a line or one bar artist per bin.
    rasterized : bool
        Rasterise the image layer when saving to vector formats.

    Returns
    -------
    ax : matplotlib Axes
        The raster axes. The marginal axes (if any) are reachable via `ax.figure.axes`.
    neuron_order : ndarray, shape (n_neurons,)
        Permutation applied to the rows: row k of the plot shows neuron `neuron_order[k]`.
    """
    activity = np.asarray(recording)

    if activity.ndim != 2:
        raise ValueError("recording must have shape (neurons, time).")

    n_neurons, n_times = activity.shape

    if time_axis is None:
        time_axis = np.arange(n_times)
    else:
        time_axis = np.asarray(time_axis)
        if time_axis.size != n_times:
            raise ValueError("time_axis must have one value per time point.")

    # Bin edges so that column i is centred on time_axis[i] (previously the extent ended at time_axis[-1], which shifted columns by up to one step)
    time_step = (time_axis[-1] - time_axis[0]) / (n_times - 1) if n_times > 1 else 1
    time_edges = np.linspace(
        time_axis[0] - time_step / 2, time_axis[-1] + time_step / 2, n_times + 1
    )
    neuron_edges = np.arange(n_neurons + 1) + 0.5  # row k spans [k - 0.5, k + 0.5]

    if cmap is None:
        cmap = ListedColormap([background_color, spike_color])

    raster = activity >= activity_threshold

    if sort is None:
        neuron_order = np.arange(n_neurons)
    elif isinstance(sort, str):
        if sort == "firing_rate":
            neuron_order = np.argsort(raster.mean(axis=1))
        elif sort == "similarity":
            # This simple implementation will suffice for my purposes
            # Compute pairwise correlation distances (O(N^2 T))
            # Constant neurons (silent or always active) have std = 0 -> distance 0/0 = NaN; treat them as uncorrelated (distance 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                distances = np.nan_to_num(
                    pdist(activity, metric="correlation"), nan=1.0
                )
            # Build the hierarchical tree
            Z = linkage(distances, method="average")
            # Get the sorted order
            neuron_order = leaves_list(Z)
        else:
            raise ValueError(
                "sort must be None, 'firing_rate', 'similarity', an order, or a callable."
            )
    elif callable(sort):
        neuron_order = np.asarray(sort(activity))
    else:
        neuron_order = np.asarray(sort)

    if neuron_order.shape != (
        n_neurons,
    ):  # (n_neurons,) is a tuple, so we assert if it is a 1d array of length N
        raise ValueError("The neuron ordering must contain one entry per neuron.")

    if descending:
        neuron_order = neuron_order[::-1]

    raster = raster[neuron_order]

    if ax is None:
        height = max(3, np.log2(n_neurons + 1)) + (1.5 if show_marginals else 0)
        fig, ax = plt.subplots(figsize=(12, height))

    image = ax.imshow(
        raster,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        origin="upper",
        extent=(time_edges[0], time_edges[-1], neuron_edges[-1], neuron_edges[0]),
        rasterized=rasterized,
    )

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        yticks=[],
    )
    ax.spines[["top", "right"]].set_visible(False)

    if not show_marginals:
        ax.set_title(title)
        return ax, neuron_order

    # Marginal axes share the raster's axis, so zooming / xlim changes stay aligned
    divider = make_axes_locatable(ax)
    ax_neurons = divider.append_axes("right", size="12%", pad=0.08, sharey=ax)
    ax_time = divider.append_axes("top", size="22%", pad=0.08, sharex=ax)

    # Per-neuron rate: fraction of time steps active, shape (n_neurons,)
    neuron_rates = raster.mean(axis=1)
    ax_neurons.stairs(
        neuron_rates,
        neuron_edges,
        orientation="horizontal",
        fill=True,
        color=marginal_color,
    )
    ax_neurons.set_xlim(0, max(neuron_rates.max(), 1e-12) * 1.05)
    ax_neurons.set_xlabel("Activity rate")
    # Drop the 0 tick, it would collide with the raster's last time tick label
    ax_neurons.xaxis.set_major_locator(MaxNLocator(nbins=2, prune="lower"))
    ax_neurons.tick_params(labelleft=False, left=False)
    ax_neurons.spines[["top", "right", "left"]].set_visible(False)

    # Population activity: fraction of neurons active per time step, shape (n_times,)
    population_activity = raster.mean(axis=0)
    ax_time.stairs(population_activity, time_edges, fill=True, color=marginal_color)
    ax_time.set_ylim(0, max(population_activity.max(), 1e-12) * 1.05)
    ax_time.set_ylabel("Fraction\nactive")
    ax_time.set_title(title)
    ax_time.tick_params(labelbottom=False, bottom=False)
    ax_time.spines[["top", "right", "bottom"]].set_visible(False)

    return ax, neuron_order
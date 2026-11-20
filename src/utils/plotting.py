"""Plotting helpers: colour palettes, colormaps, a self-placing inset, and saving figures.

Examples
--------
    from src.utils.plotting import PET6, PET8, PET10, pet, OWN, VIRIDIS, RDBU_R, TWILIGHT
    from src.utils.plotting import add_linear_inset, save_fig

    # Colours: Petroff's accessible sequences, one set per number of series
    PET10.use()                       # default colour cycle for all following plots
    pet(4).use()                      # smallest set with >= 4 colours, here PET6
    ax.plot(x, y, color=PET10.red)    # one colour by name ...
    PET10[0], list(PET10)             # ... or by position
    PET10                             # as the last line of a cell: named swatches
    PET10.show()                      # swatches, also as seen with colour-vision deficiencies
    OWN.use(); OWN.grey               # own palette, same interface; grey is not in its cycle

    # Colormaps for continuous data (alternatives are listed where they are defined)
    ax.imshow(z, cmap=VIRIDIS)
    ax.imshow(z, cmap=RDBU_R, norm=CenteredNorm())   # from matplotlib.colors; white at 0

    # Linear inset in the emptiest corner of a (log-log) plot
    ax_in = add_linear_inset(ax, x, y)                      # scatter x, y into it
    add_linear_inset(ax, draw=draw, corner="upper left")    # draw(a) plots into the inset

    # Saving; the default folder is FIGURES_DIR (results/figures)
    save_fig(fig, "bn/d2_vs_eps")                       # -> results/figures/bn/d2_vs_eps.png
    save_fig(fig, "bn/d2_vs_eps", fmt=("png", "pdf"))   # several formats at once
    save_fig(fig, "bn/d2_vs_eps.svg")                   # an extension in the name picks the format
    save_fig(fig, "bn/d2_vs_eps", aspect=16 / 9)        # saved file 16:9, the figure stays as it is
    save_fig(None, "quick_look")                        # the current figure
    save_fig(fig, "bn/d2_vs_eps", keep_old=0)           # replace without keeping the old file
    save_fig(fig, "draft", folder="/tmp")               # any other folder
    save_fig(plotly_fig, "bn/interactive")              # plotly: self-contained .html + .png

A replaced file moves to old/ inside the figures folder, in the same subfolders:
results/figures/bn/d2_vs_eps.png -> results/figures/old/bn/d2_vs_eps__<YYYY-MM-DD_HHMMSS>.png.
Only the newest `keep_old` versions (default 5) stay there. A figure identical to the
saved file is not written again, so re-running a cell does not fill old/ with copies.
"""

import io
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from src.config.paths import FIGURES_DIR, PROJECT_ROOT


# ============================================================================
# Categorical palettes
# ============================================================================
class Palette(tuple):
    """Colours in cycle order that can also be reached by name.

    Behaves like a tuple of hex strings (pal[0], list(pal), len(pal)) and also
    gives pal.blue or pal["blue"]. Colours in `extra` (e.g. a grey for
    reference lines) can be reached by name but are not part of the cycle.
    In a notebook, a palette on the last line of a cell shows as swatches.
    """

    def __new__(cls, cycle, extra=None):
        self = super().__new__(cls, cycle.values())
        self.names = {**cycle, **(extra or {})}  # every colour by name, cycle first
        return self

    def __getitem__(self, key):
        return self.names[key] if isinstance(key, str) else super().__getitem__(key)

    def __getattr__(self, name):  # only called when normal attribute lookup fails
        if name.startswith("_") or name == "names":  # keep copy/pickle/numpy probes working
            raise AttributeError(name)
        try:
            return self.names[name]
        except KeyError:
            raise AttributeError(f"no colour {name!r}; choose from {', '.join(self.names)}") from None

    def __reduce__(self):  # lets copy and pickle rebuild it from (cycle, extra)
        items = list(self.names.items())
        return Palette, (dict(items[: len(self)]), dict(items[len(self) :]))

    def __repr__(self):
        return "Palette(" + ", ".join(f"{n}={c}" for n, c in self.names.items()) + ")"

    def _repr_html_(self):
        """Swatches with name and hex code, shown by Jupyter."""
        cells = "".join(
            f'<div style="display:inline-block;width:72px;margin-right:4px;font:11px monospace">'
            f'<div style="height:28px;background:{c};border-radius:4px"></div>{n}<br>{c}</div>'
            for n, c in self.names.items()
        )
        return f"<div>{cells}</div>"

    def use(self):
        """Make this palette the default colour cycle for all following plots."""
        plt.rcParams["axes.prop_cycle"] = cycler(color=list(self))

    def show(self):
        """Show the swatches as seen with normal vision and with the three colour-vision
        deficiencies (full severity, i.e. the worst case); also works outside Jupyter."""
        rgb = np.array([to_rgb(c) for c in self.names.values()])
        rows = {"normal": rgb, **{kind: simulate_cvd(rgb, kind) for kind in _MACHADO}}
        fig, ax = plt.subplots(figsize=(0.9 * len(rgb) + 1, 0.4 + 0.45 * len(rows)), layout="constrained")
        ax.imshow(np.stack(list(rows.values())), aspect="auto")
        ax.set_yticks(range(len(rows)), list(rows))
        ax.set_xticks(range(len(rgb)), list(self.names))
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        plt.show()


# Petroff, "Accessible Color Sequences for Data Visualization" (arXiv:2107.02270).
# Each set is optimised as a whole for its size: its colours stay distinct under
# simulated colour-vision deficiencies, have enough contrast on white, and are chosen
# and ordered with an aesthetics model fitted to a crowd-sourced survey. The sets are
# not subsets of each other, so use the smallest one that has enough colours (pet(n)),
# or PET10 everywhere if a series should keep its colour across all figures.
# Matplotlib ships them as the styles "petroff10" (>= 3.10), "petroff6", "petroff8" (>= 3.11).
PET6 = Palette(
    {
        "blue": "#5790fc",
        "orange": "#f89c20",
        "red": "#e42536",
        "plum": "#964a8b",
        "grey": "#9c9ca1",
        "violet": "#7a21dd",
    }
)
PET8 = Palette(
    {
        "blue": "#1845fb",
        "orange": "#ff5e02",
        "red": "#c91f16",
        "magenta": "#c849a9",
        "khaki": "#adad7d",
        "cyan": "#86c8dd",
        "lightblue": "#578dff",
        "grey": "#656364",
    }
)
PET10 = Palette(
    {
        "blue": "#3f90da",
        "amber": "#ffa90e",
        "red": "#bd1f01",
        "grey": "#94a4a2",
        "purple": "#832db6",
        "brown": "#a96b59",
        "orange": "#e76300",
        "khaki": "#b9ac70",
        "slate": "#717581",
        "cyan": "#92dadd",
    }
)


def pet(n):
    """The smallest Petroff palette with at least n colours: PET6, PET8 or PET10.

    Note: in a worst-case check (smallest distance between any two colours, for normal
    vision and three simulated deficiencies), PET8 is the weakest of the three sets: its
    orange and red are the closest pair. PET10 is at least as distinct as PET8 for any
    number of series, so for 7 or 8 series PET10 is just as good a choice.
    """
    for palette in (PET6, PET8, PET10):
        if n <= len(palette):
            return palette
    raise ValueError(f"the Petroff palettes have at most 10 colours, not {n}")


# Own palette, refined: colours differ in lightness, so red and green stay apart for
# colour-blind readers; tuned so that the worst case over normal vision and simulated
# protan-, deutan- and tritanopia is as large as possible, especially for the first
# colours, with a shared moderate chroma so that no colour glows. The most distinct
# option for up to 4 series. Grey is kept out of the cycle so it can mark references.
OWN = Palette(
    {
        "blue": "#6eb3fd",
        "orange": "#f18820",
        "purple": "#6435a1",
        "brown": "#773912",
        "red": "#c8533f",
        "green": "#7cb986",
        "pink": "#d283af",
    },
    extra={"grey": "#7c7b7a"},
)

# --- Colour-vision deficiency simulation, used by Palette.show() ---
# Machado, Oliveira & Fernandes (2009), severity 1.0; the matrices act on linear RGB.
_MACHADO = {
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def simulate_cvd(rgb, kind):
    """How sRGB colours (rows of values in [0, 1]) look with a colour-vision
    deficiency: kind is "deutan", "protan" or "tritan"."""
    rgb = np.asarray(rgb, dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)  # sRGB -> linear
    lin = np.clip(lin @ np.array(_MACHADO[kind]).T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)  # linear -> sRGB


# ============================================================================
# Colormaps for continuous data: all perceptually ordered and readable with
# colour-vision deficiencies. Alternatives in the comments below each; the
# cmc.* maps need `pip install cmcrameri` and `import cmcrameri` before use.
# ============================================================================
VIRIDIS = "viridis"  # sequential: one quantity from low to high
# "cividis"                     like viridis, optimised for colour-vision deficiencies, a bit duller
# "magma", "inferno", "plasma"  same family, more contrast, near-black low end
# "cmc.batlow"                  Crameri's all-round sequential map, more colourful

RDBU_R = "RdBu_r"  # diverging: signed quantity around a meaningful centre, blue < centre < red;
#                    pair with norm=matplotlib.colors.CenteredNorm() so the centre is white
# "coolwarm"                    softer, light grey centre
# "PuOr", "BrBG"                ColorBrewer diverging maps, also colour-blind safe
# "cmc.vik"                     Crameri's diverging map
# "berlin", "managua", "vanimo" dark centre, so values near it recede (matplotlib >= 3.10)

TWILIGHT = "twilight"  # cyclic: angles, phases, time of day (both ends have the same colour)
# "twilight_shifted"            same, dark instead of light at the ends
# "cmc.romaO"                   Crameri's cyclic map


# ============================================================================
# Linear inset that places itself in the emptiest corner of a plot
# ============================================================================

CORNERS = ("upper left", "upper right", "lower right", "lower left")


def add_linear_inset(
    ax,
    x=None,
    y=None,
    *,
    draw=None,
    corner="auto",
    size=(0.36, 0.38),
    min_scale=0.6,
    overlap="avoid",
    tolerance=0,
    avoid=None,
    pad=0.02,
    label_room=(0.06, 0.08),
    **scatter_kw,
):
    """Add a linear-axes inset in a free corner of `ax`.

    Call this after everything else is drawn on `ax` and its scales are set,
    because the placement is computed from where things sit right now.

    The inset keeps clear of all scatter points and lines already on `ax`
    (lines along their whole length, not just at their data points). For
    anything else (text, fills, error bars), pass extra points via `avoid`.
    A legend with the default loc="best" also steers clear of the inset.

    What goes into the inset (use either or both):
      - x, y: scattered with the defaults color="tab:blue", alpha=0.5, s=6,
        overridable via **scatter_kw.
      - draw: a function draw(a) that plots into the axes `a` it is given.
        Call the same function on `ax` yourself, and the inset shows exactly
        what the main plot shows, just on linear axes.

    Search order: the biggest inset first, and at each size the corners in
    your order of preference. The first box that covers at most `tolerance`
    points wins.

    Parameters
    ----------
    ax : matplotlib Axes
        The parent plot (any scales, e.g. log-log).
    x, y : array-like, optional
        Data to scatter into the inset.
    draw : callable, optional
        Function that draws into a given axes, called as draw(ax_in).
    corner : "auto", str, or list of str
        "auto": try "upper left", "upper right", "lower right", "lower left".
        One name, e.g. "upper right": try that corner first, the others as fallbacks.
        A list, e.g. ["upper right", "lower right"]: try only these, in this order.
    size : (width, height)
        Full inset size, as fractions of the parent axes.
    min_scale : float
        How far the inset may shrink (relative to `size`) to find a free
        corner. 1 = never shrink.
    overlap : "allow", "avoid", or "forbid"
        "allow":  ignore what is drawn; use the first corner at full size.
        "avoid":  look for a free corner; if there is none, use the box that
                  covers the fewest points and warn.
        "forbid": look for a free corner; if there is none, draw no inset,
                  warn, and return None.
    tolerance : int
        How many points a box may cover and still count as free. A line
        counts as one point per ~1% of the axes it runs through.
    avoid : (x, y) or None
        Extra points to keep clear of, in data coordinates.
    pad : float
        Gap between the inset (tick labels included) and the parent's frame.
    label_room : (float, float)
        Room for the inset's tick labels, left of and below the box.
        Increase it for bigger tick labels or a smaller figure.
    **scatter_kw
        Passed on to the scatter of x, y.

    Returns
    -------
    ax_in : matplotlib Axes or None
        The inset, for further styling (labels, limits, ...).
        None if overlap="forbid" found no free corner.
    """
    # --- 1. Validate the options, so a typo fails with a clear message ---
    if overlap not in ("allow", "avoid", "forbid"):
        raise ValueError(f"overlap must be 'allow', 'avoid' or 'forbid', not {overlap!r}")
    if (x is None) != (y is None):
        raise ValueError("give both x and y, or neither")
    if corner == "auto":
        order = list(CORNERS)
    elif isinstance(corner, str):  # one preferred corner, the others as fallbacks
        order = [corner] + [c for c in CORNERS if c != corner]
    else:  # an explicit list: only these corners, in this order
        order = list(corner)
    bad = [c for c in order if c not in CORNERS]
    if bad:
        raise ValueError(f"unknown corner(s) {bad}; choose from {CORNERS}")

    # --- 2. Collect what to keep clear of, in parent-axes fractions ---
    # (0, 0) is the parent's lower left corner and (1, 1) its upper right,
    # whatever the axis scales are, so log axes need no special treatment.
    ax.autoscale_view()  # fix the parent's axis limits before measuring
    pts = _drawn_points(ax)
    if avoid is not None:  # extra points, given in data coordinates
        xy = np.column_stack([np.ravel(avoid[0]), np.ravel(avoid[1])]).astype(float)
        pts = np.vstack([pts, ax.transAxes.inverted().transform(ax.transData.transform(xy))])
    pts = pts[np.isfinite(pts).all(axis=1)]  # drop points a log axis cannot show (<= 0)

    # --- 3. Build all candidate boxes and count the points under each ---
    w_full, h_full = size
    mx, my = label_room
    # Sizes to try, from full size down to min_scale in 5 steps (just full size if min_scale >= 1).
    scales = np.linspace(1, min_scale, 5) if min_scale < 1 else [1.0]

    def box(name, scale):
        """Inset bounds (x0, y0, w, h) for one corner and size, in parent-axes fractions."""
        w, h = scale * w_full, scale * h_full
        # The inset's tick labels sit left of and below the box, so the left
        # and lower corners leave label_room between the box and the frame.
        x0 = mx + pad if "left" in name else 1 - w - pad
        y0 = my + pad if "lower" in name else 1 - h - pad
        return x0, y0, w, h

    def n_covered(x0, y0, w, h):
        """How many points lie under the box or under its tick labels."""
        in_x = (pts[:, 0] > x0 - mx) & (pts[:, 0] < x0 + w)
        in_y = (pts[:, 1] > y0 - my) & (pts[:, 1] < y0 + h)
        return int(np.sum(in_x & in_y))

    # Search order: outer loop over sizes (biggest first), inner loop over
    # corners (most preferred first). A big inset in another corner thus beats
    # a shrunken one in the preferred corner.
    candidates = [(name, scale, box(name, scale)) for scale in scales for name in order]
    counts = [n_covered(*bounds) for _, _, bounds in candidates]

    # --- 4. Pick a box according to the overlap rule ---
    limit = np.inf if overlap == "allow" else tolerance  # with "allow" every box counts as free
    free = [i for i, n in enumerate(counts) if n <= limit]
    if free:
        i = free[0]  # first free box in search order: biggest size, most preferred corner
    elif overlap == "forbid":
        warnings.warn(
            f"No corner covers <= {tolerance} points, even at {min_scale:.0%} size, so no "
            "inset was drawn. Try a higher tolerance, a lower min_scale, or overlap='avoid'.",
            stacklevel=2,
        )
        return None
    else:  # "avoid": take the least bad box; argmin returns the first of
        i = int(np.argmin(counts))  # several equal minima, i.e. biggest and most preferred
    name, scale, bounds = candidates[i]
    if not free:  # only reachable with overlap="avoid"
        warnings.warn(
            f"No free corner; placed the inset {name} at {scale:.0%} size, "
            f"where it covers {counts[i]} points.",
            stacklevel=2,
        )

    # --- 5. Reserve the spot so that a legend avoids it ---
    # legend(loc="best") is placed at draw time and dodges rectangles on the
    # axes, so an invisible rectangle over the inset and its tick labels keeps
    # the legend off the inset. Its transform is transAxes, so it does not
    # change the parent's axis limits.
    x0, y0, w, h = bounds
    ax.add_patch(
        Rectangle(
            (x0 - mx, y0 - my), w + mx, h + my,
            transform=ax.transAxes, visible=False, label="_inset_area",
        )
    )

    # --- 6. Create the inset: style first, so that `draw` can still override it ---
    ax_in = ax.inset_axes(bounds)
    ax_in.tick_params(labelsize=8)
    ax_in.grid(alpha=0.4, linewidth=0.5)
    if x is not None:
        style = {"color": "tab:blue", "alpha": 0.5, "s": 6, **scatter_kw}  # your kwargs win
        ax_in.scatter(x, y, **style)
    if draw is not None:
        draw(ax_in)
    return ax_in


def _drawn_points(ax, step=0.01):
    """Positions of the scatter points and lines on `ax`, in parent-axes fractions."""
    to_axes = ax.transAxes.inverted()
    chunks = [np.empty((0, 2))]  # so vstack works on an empty plot
    for coll in ax.collections:
        if isinstance(coll, PathCollection):  # what ax.scatter creates
            offsets = np.ma.filled(coll.get_offsets(), np.nan).astype(float)
            chunks.append(to_axes.transform(coll.get_offset_transform().transform(offsets)))
    for line in ax.lines:  # ax.plot, axhline, ...; each brings its own transform
        xy = to_axes.transform(line.get_transform().transform(line.get_xydata()))
        chunks.append(_along_line(xy, step))
    return np.vstack(chunks)


def _along_line(xy, step):
    """Sample a polyline every `step` (in axes fractions).

    Without this, a line with few data points could pass right through the
    inset between two of them.
    """
    if len(xy) < 2:
        return xy
    a, b = xy[:-1], xy[1:]  # start and end of each segment
    n = np.nan_to_num(np.hypot(*(b - a).T) / step, nan=1, posinf=1)
    n = np.clip(np.ceil(n), 1, 200).astype(int)  # samples per segment; capped for far-off points
    pieces = [np.linspace(p, q, k, endpoint=False) for p, q, k in zip(a, b, n)]
    return np.vstack(pieces + [xy[-1:]])




# ============================================================================
# Saving figures
# ============================================================================
_PLOTLY_FORMATS = {"html", "png", "jpg", "jpeg", "webp", "svg", "pdf"}
_STAMP = "%Y-%m-%d_%H%M%S"  # same time stamps as src.utils.persist


def save_fig(
    fig, name, *, fmt=None, folder=FIGURES_DIR, aspect=None, dpi=300, keep_old=5, close=False, **kwargs
):
    """Save a matplotlib or plotly figure in one or more formats and return the paths.

    fig:      matplotlib Figure, plotly Figure, or None for the current matplotlib figure.
    name:     file name without extension; may contain subfolders ("bn/d2_vs_eps").
              An extension ("bn/d2_vs_eps.pdf") picks the format.
    fmt:      "pdf", ("png", "pdf"), ... Default: "png" for matplotlib, ("html", "png")
              for plotly (images of plotly figures need the kaleido package).
    folder:   root folder; default results/figures, any path works. Created if missing.
    aspect:   width / height of the saved file, e.g. 1.5 or 16 / 9: the width stays and
              the height adapts. Only the file changes; the figure keeps its size. With the
              default tight cropping the file's ratio can be off by the trimmed margin.
              None (default): save the figure as it is.
    dpi:      resolution of raster formats (png, jpg).
    keep_old: how many replaced versions of each file to keep in <folder>/old/
              (0: none, None: all).
    close:    close the matplotlib figure afterwards. Keep False in notebooks (a closed
              figure is not displayed), True in loops that make many figures.
    **kwargs: passed on to fig.savefig (matplotlib) or fig.to_image (plotly).
    """
    # --- 1. Which figure, and is it matplotlib or plotly? ---
    if fig is None:
        fig = plt.gcf()
        if not fig.axes:
            raise ValueError("No figure to save: pass one, or plot something first.")
    is_plotly = hasattr(fig, "write_html")  # plotly figures have it, matplotlib ones don't

    # --- 2. Which formats? An extension in the name wins over fmt ---
    root = Path(folder)
    path = root / name
    allowed = _PLOTLY_FORMATS if is_plotly else set(fig.canvas.get_supported_filetypes())
    ext = path.suffix.lstrip(".").lower()
    if ext in allowed:  # "d2.pdf" -> pdf, but "eps_0.5" stays a name
        fmts, path = [ext], path.with_suffix("")
    elif fmt is None:
        fmts = ["html", "png"] if is_plotly else ["png"]
    else:
        fmts = [fmt] if isinstance(fmt, str) else list(fmt)
    unknown = [f for f in fmts if f not in allowed]
    if unknown:
        raise ValueError(f"unsupported format(s) {unknown}; choose from {sorted(allowed)}")
    path.parent.mkdir(parents=True, exist_ok=True)

    # --- 3. Render each format in memory; write it only if it differs from the saved file ---
    written = []
    for f in fmts:
        target = path.parent / f"{path.name}.{f}"  # not with_suffix: "eps_0.5" must keep its ".5"
        data = _render(fig, f, is_plotly, aspect, dpi, kwargs)
        if target.exists() and target.read_bytes() == data:
            status = "unchanged"  # keep the saved file, and its date
        else:
            status = "new"
            if target.exists():
                status = "replaced"
                if keep_old != 0:
                    _retire(target, root, keep_old)
                    status = "replaced, previous moved to old/"
            target.write_bytes(data)
        written.append(target)
        print(f"Saved {_shown(target)} ({_size(target)}, {status})")

    if close and not is_plotly:
        plt.close(fig)
    return written


def _render(fig, fmt, is_plotly, aspect, dpi, kwargs):
    """The figure as the bytes of a file in format `fmt`."""
    if is_plotly:
        if fmt == "html":
            return fig.to_html(include_plotlyjs=True).encode()  # self-contained, works offline
        size = {}
        if aspect:
            width = fig.layout.width or 700  # plotly's default size is 700 x 500 px
            size = {"width": width, "height": round(width / aspect)}
        return fig.to_image(format=fmt, **{"scale": 2, **size, **kwargs})

    opts = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.05, **kwargs}
    if not opts.get("transparent"):
        opts.setdefault("facecolor", "white")  # no see-through jpgs or dark-mode surprises
    # PDF and SVG normally store their creation date, and SVG uses random ids. Without
    # them the same figure always gives the same bytes, so "unchanged" can be detected.
    if fmt in ("pdf", "svg"):
        no_date = {"CreationDate": None} if fmt == "pdf" else {"Date": None}
        opts["metadata"] = {**no_date, **kwargs.get("metadata", {})}
    buf = io.BytesIO()
    size = fig.get_size_inches().copy()
    try:
        if aspect:
            fig.set_size_inches(size[0], size[0] / aspect)  # constrained layout re-flows on save
        with plt.rc_context({"svg.hashsalt": plt.rcParams["svg.hashsalt"] or "save_fig"}):
            fig.savefig(buf, format=fmt, **opts)
    finally:
        fig.set_size_inches(size)  # the figure itself keeps its size
    return buf.getvalue()


def _retire(target, root, keep):
    """Move `target` to <root>/old/, keeping its subfolders, as <name>__<stamp>.<ext> (the
    stamp is when it was saved); then keep only the newest `keep` versions (None: all).

    results/figures/bn/d2.png -> results/figures/old/bn/d2__2026-11-19_143012.png
    """
    rel = target.relative_to(root) if target.is_relative_to(root) else Path(target.name)
    old_dir = root / "old" / rel.parent
    old_dir.mkdir(parents=True, exist_ok=True)
    stem, ext = rel.name.rsplit(".", 1)
    stamp = datetime.fromtimestamp(target.stat().st_mtime).strftime(_STAMP)
    dest, n = old_dir / f"{stem}__{stamp}.{ext}", 1
    while dest.exists():  # two versions within the same second
        n += 1
        dest = old_dir / f"{stem}__{stamp}_{n}.{ext}"
    target.replace(dest)
    if keep is not None:
        version = re.compile(rf"{re.escape(stem)}__\d{{4}}-\d\d-\d\d_\d{{6}}(?:_\d+)?\.{re.escape(ext)}")
        versions = sorted(p for p in old_dir.iterdir() if version.fullmatch(p.name))  # oldest first
        for p in versions[:-keep]:
            p.unlink()


def _size(path):
    """File size as a short string: 840 B, 84 kB, 8.4 MB."""
    n = path.stat().st_size
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB" if n >= 1e3 else f"{n} B"


def _shown(path):
    """The path relative to the repository root when it lies inside it."""
    return path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
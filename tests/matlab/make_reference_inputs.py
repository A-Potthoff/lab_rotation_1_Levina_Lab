"""Write the inputs for make_reference.m (step 1 of 2 of fixture generation).

Usage (from the repository root):
    python tests/matlab/make_reference_inputs.py
    matlab -batch "run('tests/matlab/make_reference.m')"

The time series are generated here (numpy RNG, fixed seed) so that MATLAB and
Python see bit-identical inputs; MATLAB then only runs the ORIGINAL toolkit
functions on them and stores the outputs in tests/fixtures/.
"""

from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import lfilter

rng = np.random.default_rng(seed=20261011)
out = Path(__file__).resolve().parents[1] / "fixtures" / "prox_crit_inputs.mat"


def ar(kern, T, sigma=1.0):
    return lfilter([1.0], np.r_[1.0, -np.asarray(kern)], sigma * rng.standard_normal(T))


# A: the tutorial's AR(3) model, single series
xA = ar([0.8, 0.2, -0.1], 10_000)
# B: segmented AR(2) with mean offset; last segment shorter than the AR order
xB = [ar([0.5, 0.3], n) + 3.0 for n in (3000, 1500, 800, 4)]
# C: short series close to criticality
xC = ar([0.6, 0.38], 2000)


def cell(segments):
    """MATLAB cell array of row vectors <- 1-D numpy object array."""
    c = np.empty(len(segments), dtype=object)
    for i, s in enumerate(segments):
        c[i] = s.reshape(1, -1)
    return c


dsets = cell([np.empty(0)] * 3)  # cell of cells
dsets[0], dsets[1], dsets[2] = cell([xA]), cell(xB), cell([xC])

# cases: dataset (1-based), order, b, fit (0 = YuleWalker, 1 = MaxLikelihood), err bars
cases = np.array(
    [
        [1, 3, 2, 0, 0],
        [1, 3, 2, 1, 1],
        [1, 5, 4, 1, 1],
        [1, 8, 6, 0, 0],
        [2, 5, 2, 1, 1],
        [2, 5, 2, 0, 0],
        [3, 4, 2, 1, 1],
        [3, 4, 4, 0, 0],
        [1, 3, 6, 0, 0],  # order == b/2: M_b is a single point
        [1, 2, 6, 0, 0],  # order < b/2: crashes in MATLAB (CHANGED[3])
    ],
    dtype=float,
)

# kernels for calc_dbv1 (padded with NaN to a matrix; length in column 0)
dbv1_kerns = [
    ([0.8, 0.2, -0.1], 2),
    ([0.8, 0.2, -0.1], 4),
    ([0.5, -0.2, 0.3, 0.1, -0.05, 0.02], 2),
    ([0.5, -0.2, 0.3, 0.1, -0.05, 0.02], 4),
    ([0.5, -0.2, 0.3, 0.1, -0.05, 0.02], 6),
    ([1.2, -0.4, 0.3, 0.1, -0.2, 0.05, 0.02, -0.01, 0.03, 0.01], 8),
]
K = np.full((len(dbv1_kerns), 12), np.nan)
for i, (k, b) in enumerate(dbv1_kerns):
    K[i, 0], K[i, 1] = len(k), b
    K[i, 2 : 2 + len(k)] = k

savemat(out, {"dsets": dsets, "cases": cases, "dbv1_kerns": K}, do_compression=True)
print("wrote", out)

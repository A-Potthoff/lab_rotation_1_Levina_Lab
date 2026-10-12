"""Loading the Senzai et al. (2019) demo recording YMV15 as readable tables.

Data: single-unit extracellular recordings from mouse visual cortex during natural sleep and wakefulness (Senzai, Fernandez-Ruiz & Buzsaki, Neuron 2019), as distributed with Sooter's prox_crit_toolkit demo. Not own data.

The .mat file holds:
    st       spike times in seconds, all units pooled, sorted (1 x 7,850,337)
    clu      unit id of each spike (same length as st; 72 distinct ids between 2 and 80)
    WAKE, NREM, REM   behavioral-state episodes, one row [start, end] in whole seconds each
    Session  recording name ("YMV15")
"""

import numpy as np
import pandas as pd
import scipy.io as sio

from src.config.paths import DATA_PROCESSED, YMV15_MAT

STATES = ("WAKE", "NREM", "REM")


def load_ymv15(path=YMV15_MAT):
    """Load the recording as two pandas DataFrames.

    Returns
    -------
    spikes : DataFrame
        One row per spike: ``time_s`` (float, s), ``unit`` (int id).
    states : DataFrame
        One row per behavioral-state episode, sorted by start: ``state`` (WAKE / NREM / REM), ``start_s``, ``end_s``, ``duration_s``.
    """
    # squeeze_me drops MATLAB's extra singleton dimension: (1, N) arrays become (N,)
    data = sio.loadmat(path, squeeze_me=True)
    spikes = pd.DataFrame({"time_s": data["st"], "unit": data["clu"]})

    episodes = []
    for state in STATES:
        for start, end in np.atleast_2d(data[state]):
            episodes.append(
                {"state": state, "start_s": float(start), "end_s": float(end)}
            )
    states = pd.DataFrame(episodes).sort_values("start_s", ignore_index=True)
    states["duration_s"] = states["end_s"] - states["start_s"]
    return spikes, states


def export_ymv15_csv(out_dir=DATA_PROCESSED):
    """Write the recording as two CSV files and return their paths.

    ``senzai2019_YMV15_spikes.csv`` (~125 MB, 7.85 million rows; git-ignored because it exceeds GitHub's 100 MB file limit and can be regenerated) and ``senzai2019_YMV15_states.csv`` (72 rows).
    """
    spikes, states = load_ymv15()
    spikes_path = out_dir / "senzai2019_YMV15_spikes.csv"
    states_path = out_dir / "senzai2019_YMV15_states.csv"
    spikes.to_csv(
        spikes_path, index=False, float_format="%.5f"
    )  # spike times are on a 0.05 ms grid: 5 decimals are exact
    states.to_csv(states_path, index=False)
    return spikes_path, states_path

"""Project paths, resolved relative to the repository root.

Import these instead of hard-coding paths, so notebooks and scripts work from
any working directory:

    from src.config.paths import YMV15_MAT, FIGURES_DIR
"""

from pathlib import Path

# src/config/paths.py -> parents[2] is the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

RESULTS_OBJECTS = PROJECT_ROOT / "results" / "objects"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
EXTERNAL_CODE_DIR = PROJECT_ROOT / "src" / "external"

# Single-unit ephys from mouse visual cortex (Senzai et al., Neuron 2019),
# shipped with Sooter's prox_crit_toolkit demo. Variables: st (spike times, s),
# clu (unit ids), WAKE / NREM / REM ([start, end] windows in s), Session.
YMV15_MAT = DATA_EXTERNAL / "senzai2019_YMV15.mat"

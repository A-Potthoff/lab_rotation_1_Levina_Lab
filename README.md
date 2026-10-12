# Code for a lab rotation in the lab of [Anna Levina](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/self-organization-and-optimality-in-neuronal-networks/)

## Purpose
We've set out to work with, and analyse the measure of distance to criticality, as proposed by Sam Sooter et. al. (2026) (DOI: [10.1101/2025.08.03.668332](https://www.biorxiv.org/content/10.1101/2025.08.03.668332v2)).

## Architecture

### Core Execution
- `src/`: Core logic, mathematical models, simulations and functions.
- `notebooks/`: Exploratory data analysis, prototyping, and result visualization.
- `archive/`: for deprecated scripts and redundant logic that might be useful later.

### Data & Output
- `data/`: Local storage for datasets.
- `results/objects/`: Save computed objects (e.g., `.pkl`, `.npy`, `.h5`).
- `models/saved/`: eg. model weights and states.
- `reports/figures/`: plots and figures
- `logs/`

### Documentation
- `docs/annotations/`: Markdown explanations linked to source code. Used for mathematical derivations and literature references.

## Third-party code
- `src/external/proxy_crit_toolkit_sam_sooter_2026_10_09/`: original MATLAB toolkit by J. S. Sooter et al. ([prox_crit_toolkit](https://github.com/sam-sooter/prox_crit_toolkit)), CC BY-NC 4.0.
- `src/analysis/prox_crit/`: **Python translation of that toolkit (not own work)**, also CC BY-NC 4.0. See its `NOTICE.md` and `FLAGS.md` (issues found in the original). It is tested against the MATLAB original in `tests/test_prox_crit.py`, and the demos are ported in `notebooks/prox_crit_demo*.ipynb`.

  ```python
  from src.analysis.prox_crit import calc_db
  db, sddb, kern, sigma, H, kernc, exit_status = calc_db(x, order=5, delta_t=0.04, b=2,
                                                         fit_method="MaxLikelihood", with_err_bars=True)
  ```

## Setup
This project uses a Conda environment.
```bash
conda config --set channel_priority strict
conda env create -f environment.yml
conda activate neuro_env
pip install -e .
```

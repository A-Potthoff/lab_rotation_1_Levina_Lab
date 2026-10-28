# Code for a lab rotation in the lab of [Anna Levina](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/self-organization-and-optimality-in-neuronal-networks/)

## Purpose
We've set out to work with, and analyse the measure of distance to criticality, as proposed by Sam Sooter et. al. (2026) (DOI: [10.1101/2025.08.03.668332](https://www.biorxiv.org/content/10.1101/2025.08.03.668332v2)).

## Architecture

### Code
- `src/config/`: Code for loading and interpreting configurations, e.g. project paths (`paths.py`).
- `src/models/`: Model implementations and dynamics, e.g. the probabilistic integrate-and-fire branching network (`branching_model/`) and autoregressive models (`autoregression/`).
- `src/analysis/`: Reusable analysis functions for models, simulations, and data, e.g. avalanches, autocorrelation timescales, and `prox_crit/`.
- `src/utils/`: General-purpose helpers, e.g. plotting and data loading.
- `src/external/`: Externally sourced or copied code; preserve attribution, license terms, and relevant original documentation.
- `tests/`: Unit and integration tests for reusable code.

### Exploration & experiments
- `notebooks/`: Exploratory scratch work, prototypes, and analyses not yet organized into a named experiment.
- `experiments/`: Named research investigations; each can contain notebooks, scripts, configs, and experiment-specific outputs.
- `scripts/`: Stand-alone runnable scripts, e.g. data conversions and maintenance tasks.
- `legacy/`: Deprecated code snippets retained for reference; Git history handles ordinary code evolution.

### Data & output
- `data/raw/`, `data/interim/`, `data/processed/`, `data/external/`: raw inputs, intermediate preprocessing steps, analysis-ready data, and data from publications or databases.
- `results/figures/`, `results/tables/`: final or cross-experiment figures and summary tables.
- `results/objects/`: heavy computed outputs, e.g. pickles and HDF5 files shared across experiments.
- `results/checkpoints/`: model weights and saved states intended for reuse.
- `logs/`: logs from scripts, experiments, and computational jobs.

### Documentation
- `docs/annotations/`: Longer derivations, conceptual explanations, implementation notes, and literature references.

## Third-party code
- `src/external/proxy_crit_toolkit_sam_sooter_2026_10_09/`: original MATLAB toolkit by J. S. Sooter et al. ([prox_crit_toolkit](https://github.com/sam-sooter/prox_crit_toolkit)), CC BY-NC 4.0.
- `src/analysis/prox_crit/`: **Python translation of that toolkit (not own work)**, also CC BY-NC 4.0. See its `NOTICE.md` and `FLAGS.md` (issues found in the original). It is tested against the MATLAB original in `tests/test_prox_crit.py`, and the demos are ported in `notebooks/proxy_crit_Sooter/prox_crit_demo*.ipynb`.

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

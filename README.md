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

## Setup
This project uses a Conda environment.
```bash
conda config --set channel_priority strict
conda env create -f environment.yml
conda activate neuro_env
pip install -e .
```

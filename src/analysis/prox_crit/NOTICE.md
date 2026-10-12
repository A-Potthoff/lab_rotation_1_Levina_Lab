# NOTICE: third-party code (not own work)

This package is a **Python translation** of the MATLAB toolkit `prox_crit_toolkit`.

- **Authors of the original:** J. S. Sooter et al. Contact: Sam Sooter (jssooter@gmail.com), Woodrow Shew (shew@uark.edu).
- **Publication:** Sooter et al., *Defining and measuring proximity to criticality*, bioRxiv (2025), doi:[10.1101/2025.08.03.668332](https://www.biorxiv.org/content/10.1101/2025.08.03.668332v2).
- **Source:** https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09. The unmodified original is kept in `src/external/proxy_crit_toolkit_sam_sooter_2026_10_09/`.
- **License:** [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/); full text in `src/external/proxy_crit_toolkit_sam_sooter_2026_10_09/LICENSE.md`. This license applies to this package. The MIT license of the surrounding repository does **not** apply here.
- **Modifications:** translated to Python by Andre Potthoff (with Claude), October 2026. The logic is kept unchanged, except for four changes needed to run, which are listed with all flagged issues in [FLAGS.md](FLAGS.md). Comments and docstrings were added for explanation.
- **Demo data** (`data/external/senzai2019_YMV15.mat`): Senzai et al., *Neuron* (2019), distributed with the toolkit's demo.

## Mapping MATLAB → Python

| MATLAB (file: function) | Python (`src.analysis.prox_crit.`) |
|---|---|
| `calc_db.m: calc_db(x, order, deltaT, b, dbopt)` | `distance.calc_db(x, order, delta_t, b, *, fit_method, with_err_bars, with_qc, with_parallel)` returns `DbResult` |
| `calc_db.m: minimize_klr_b` | `distance.minimize_klr_b` |
| `calc_db.m: get_klr_b`, `get_kernw`, `get_sigmac` | `kl_rate.get_klr_b`, `kl_rate.get_kernw`, `kl_rate.get_sigmac` |
| `calc_db.m: con`, `beta_project`, `beta_lift`, `explosive` | `manifold.con`, `manifold.beta_project`, `manifold.beta_lift`, `manifold.explosive` |
| `calc_db.m: QC_plot` | `plotting.qc_plot` |
| `calc_dbv1.m: calc_dbv1` | `manifold.calc_dbv1` |
| `yw_ar.m`, `mle_ar.m` | `ar_fit.yw_ar`, `ar_fit.mle_ar` |
| `demo.mlx` / `demo_crit_db.m: runAR`, `get_spike_counts` | `demo_aux.run_ar`, `demo_aux.get_spike_counts` (plus `demo_aux.matlab_colon`) |
| `demo.mlx` | `notebooks/prox_crit_demo.ipynb` |
| `demo_crit_db.m` | `notebooks/prox_crit_demo_crit_db.ipynb` |
| `dbopt.with_QC` | `with_qc` |
| `fmincon('sqp')` | `scipy.optimize.minimize(method='SLSQP')` |
| `integral` | `scipy.integrate.quad` (same AbsTol/RelTol) |

Indexing: lag k (1-based in MATLAB, `kern(k)`) is `kern[k-1]` in Python. Every place where 1-based indices enter a formula as lag numbers is commented in the code.

# NOTICE: third-party code (not own work)

This package is a **Python translation** of the MATLAB toolkit `prox_crit_toolkit`.

- **Authors of the original:** J. S. Sooter et al. Contact: Sam Sooter (jssooter@gmail.com), Woodrow Shew (shew@uark.edu).
- **Publication:** Sooter et al., *Defining and measuring proximity to criticality*, bioRxiv (2025), doi:[10.1101/2025.08.03.668332](https://www.biorxiv.org/content/10.1101/2025.08.03.668332v2).
- **Source:** https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09. The unmodified original is kept in `src/external/proxy_crit_toolkit_sam_sooter_2026_10_09/`.
- **License:** [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/); full text in `src/external/proxy_crit_toolkit_sam_sooter_2026_10_09/LICENSE.md`. This license applies to this package, as to the rest of the repository.
- **Modifications:** translated to Python by Andre Potthoff (with Claude), October 2026. The logic is kept unchanged, except for the deviations listed below (numerical necessities, a crash fix, exact AR estimators for short segments, and a few decisions by the user). Some names were made more descriptive and small inconsistencies fixed. Comments and docstrings were added for explanation.
- **Demo data** (`data/external/senzai2019_YMV15.mat`): Senzai et al., *Neuron* (2019), distributed with the toolkit's demo.

Evidence for the statements below comes from `tests/test_prox_crit.py`, which compares against MATLAB output and high-precision mpmath values.

## Mapping MATLAB → Python

| MATLAB (file: function) | Python (`src.analysis.prox_crit.`) |
|---|---|
| `calc_db.m: calc_db(x, order, deltaT, b, dbopt)` | `distance.calc_db(x, order, delta_t, b, *, fit_method, metric, optimizer, with_err_bars, with_qc, with_parallel)` returns `DbResult` |
| `calc_db.m: minimize_klr_b` | `distance.minimize_klr_b` |
| `calc_db.m: get_klr_b`, `get_kernw`, `get_sigmac` | `kl_rate.get_klr_b`, `kl_rate.get_kernw`, `kl_rate.get_sigma_ratio` |
| `calc_db.m: con`, `beta_project`, `beta_lift`, `explosive` | `manifold.stability_constraint`, `manifold.beta_project`, `manifold.beta_lift`, `manifold.explosive` |
| `calc_db.m: QC_plot` | `plotting.qc_plot` |
| `calc_dbv1.m: calc_dbv1` | `manifold.calc_db_euclidean` |
| `yw_ar.m`, `mle_ar.m` | `ar.yw_ar`, `ar.mle_ar` |
| `demo.mlx` / `demo_crit_db.m: runAR`, `get_spike_counts` | `src.models.autoregression.autoregression_model.run_ar` (re-exported here), `spike_counts.get_spike_counts` (plus `spike_counts.matlab_range`) |
| `demo.mlx` | `notebooks/proxy_crit_Sooter/prox_crit_demo.ipynb` |
| `demo_crit_db.m` | `notebooks/proxy_crit_Sooter/prox_crit_demo_crit_db.ipynb` |
| `dbopt.with_QC` | `with_qc` |
| `fmincon('sqp')` | `scipy.optimize.minimize(method='SLSQP')` (default; `optimizer='Powell'` as an alternative) |
| `integral` | `scipy.integrate.quad` (same AbsTol/RelTol) |

Indexing: lag k (1-based in MATLAB, `kern(k)`) is `kern[k-1]` in Python. Where lag numbers enter a formula (powers, exponentials), the code builds them explicitly as `np.arange(1, n + 1)`.

## Known issues

Real limitations of the results. The code marks them with `Known issue (NOTICE.md)`.

1. **b ≥ 4: the port gives lower db than MATLAB.** The original's fmincon stops early, barely past the L2 starting point. In the reference cases its db is 0.07%, 0.03% and **5.7%** (b = 6) above the true minimum, with kernc off by up to 0.2 (verified with mpmath at both solutions and an independent Nelder-Mead run). SLSQP reaches the minimum in these cases, so MATLAB values at b ≥ 4, and especially their error bars (88% different in one case), are not reproduced. For b = 2 the port agrees with MATLAB to ≤ 2e-7 (db) and ≤ 5e-9 (sddb).
2. **SLSQP can fail for b = 6 close to criticality.** The objective is an adaptive quadrature (accurate to ~1e-7 at b ≥ 4) and its gradient a finite difference with step ~1.5e-8 (as in fmincon), so the gradient is noisy. Usually this costs ≤ 1e-5 (relative), but for a kernel close to M_2 (eg the AR(1) kernel 1 − ε, ε ≤ 1e-3) at b = 6, SLSQP stopped 40% to 9× above the minimum, or ended on an explosive model (NaN). `optimizer='Powell'` (derivative-free) found the minimum in all tested cases, at 10–20× the run time (~2–5 s per minimization at order 5, 20–70 s at order 10). Gradient-based alternatives (BFGS, L-BFGS-B, other step sizes, a smooth constraint) did not help. Error bars at b ≥ 4 inherit the optimizer noise through their finite differences.
3. **Complex outputs.** MATLAB's `sqrt` of a negative number is complex, and `np.emath.sqrt` reproduces this: `sddb` is complex when H is not negative definite, the Yule-Walker `sigma` when γ(0) − γ·φ < 0 (possible when the AR order is a sizeable fraction of the segment length, see `ar.yw_ar`). Such values are unusable; check `np.iscomplexobj` / the imaginary part before using `.real`.
4. **Resolution close to criticality.** The adaptive quadrature misses the narrow low-frequency peak of the integrand (width ~ε) for processes with relaxation times above a few thousand time steps; d_2 then comes out ~0 (see `notebooks/Sooter_x_BN.ipynb`, section 0). Kept as in the original (same tolerances as MATLAB's `integral`).

## Deviations from the original

| Where | What | Effect on results |
|---|---|---|
| `distance.minimize_klr_b` | fmincon's constraint `c ≤ 0` with `ConstraintTolerance = 0.01` is passed to SLSQP as `0.01 − c ≥ 0`: SLSQP has no tolerance option, and on M_b, c ≈ 0 with round-off of either sign by construction (z = 1 is always a root), so a hard `c ≤ 0` can be infeasible. Feasible = no root with \|z\| < 0.99, the same rule as `explosive`. | none (constraint inactive at the optimum in all tests) |
| `distance.minimize_klr_b` | Optimizer: SLSQP instead of fmincon (see known issues 1, 2); option `optimizer='Powell'`, where the stability constraint is a barrier (objective = inf for explosive models). | see known issues |
| `distance.calc_db` | When order < b/2, `minimize_klr_b` deliberately returns NaN, exit = 1 and a warning, but MATLAB then crashes reading the missing field `kernc`. Python returns the intended NaN with exit_status = 1. | crash fixed |
| `distance.calc_db` | Added option `metric='euclidean'`: L2 distance of the fitted kernel to M_b (the starting point of the KL minimization), with the same delta-method error bars. | addition |
| `kl_rate.get_klr_b` | `quad(..., points=[0.0])` splits the integral at ω = 0, where g_c(0) = 0 gives an integrable log singularity. QUADPACK's first rule on [−π, π] would evaluate exactly ω = 0 (inf); MATLAB's `integral` never places a node there. | none |
| `kl_rate.get_klr_b` | For b ≥ 4, `quad` bisects toward ω = 0, and for \|ω\| ≲ 1e-8 floating-point cancellation makes g_c exactly 0 (integral → inf). Such a node contributes 0 (its weight is < ~1e-8). | none (matches the exact time-domain KL rate to 1.4e-7) |
| `plotting.qc_plot` | KL rate integrated adaptively (`'integral'`), as in the optimization; the original's 1000-point `'trapz'` under-resolves the ω = 0 singularity, so its curve lay below the db line (~1%, up to ~25% close to criticality). | QC plot only |
| `ar.yw_ar` | γ(0) is estimated from the data like the other lags (mean of x_t², σ² = γ(0) − γ·φ); the original fixes Γ's diagonal to 1, while `calc_db`'s z-scoring (N − 1 std) gives γ(0) = (N − 1)/N. Also correct for data without unit variance. | O(1/N) |
| `ar.mle_ar` | Conditional maximum likelihood: the first `order` samples of each segment only serve as history, T = Σ(length − order). The original also uses them as regression targets with their missing history set to 0, ie it treats x = 0 before every segment start as observed, which biases the kernel towards 0 by O(order / segment length). | tutorial d2 ≤ 0.05%, sddb ≤ 0.13%; on many 10-sample windows of an AR(2) [0.6, 0.2] the original gives [0.63, 0.17], the conditional MLE [0.60, 0.20] (`test_short_windows`) |
| `spike_counts.get_spike_counts` | MATLAB's `histc` returns one extra final value counting only spikes exactly at the last edge, so every series ended with a spurious ~0 sample. It is dropped (decided by the user). | tutorial d2 values ≤ 0.07%, far below the error bars |

## Estimator details kept from the original

Kept on purpose; they matter little for typical series.

- `ar.yw_ar`: the autocovariance is normalized by the number of pairs N − lag (unbiased per lag, also pooled over short segments). Γ is then not guaranteed positive definite: when the AR order is a sizeable fraction of the segment length (eg order 5 with segments of 10–20 samples), the fit can be explosive (calc_db returns NaN) or have a complex sigma. The alternative 1/N normalization would guarantee a stable fit, but pooled over short segments it pulls the fit away from criticality (AR(1) 0.95 fitted as 0.86 from 10-sample windows), so it is not used.
- `manifold.explosive`: tolerance 1e-2, so a root with \|z\| in [0.99, 1) counts as not explosive (by design).
- `distance.calc_db` error bars: the gradient of db is a forward difference with step 1e-3 (bias of that order).

## Small fixes (no effect on results)

| Original | Python |
|---|---|
| `cntr`, `b` (as right-hand side in `beta_lift`, clashing with the manifold order), `temp`, `cme`, `con`, `calc_dbv1`, ... | descriptive names (`counter`, `rhs`, `stability_constraint`, `calc_db_euclidean`, ...) |
| `get_klr_b` with an unknown `int_method` prints an error and then fails on an undefined result | raises `ValueError` with the same message |
| `yw_ar` errors on a single array (needs a cell array), unlike `mle_ar` | both accept a single array or a list of segments |
| `inv(X'X) X'Y` and `inv(H)` | `np.linalg.solve` (same result, numerically more stable) |
| QC plot scan range fixed to ±0.2 | parameter `scan_width=0.2` |
| `minimize_klr_b` notes end mid-sentence ("To go from") | completed (reduced ↔ full kernel via `beta_lift` / `beta_project`) |
| odd `b` is not validated and errors deep inside | `half_order` raises a clear `ValueError` |
| warning "drifted off of critical manifold" (the lifted kernel is always on M_b) | "closest critical model is explosive (optimizer left the non-explosive part of M_b)" |
| `demo_crit_db.m`: comment says "d2" with b = 6; requests error bars with YuleWalker (always fails); dead `if (0)` block | notebook says d6, no error bars requested, dead block omitted |

`get_sigmac` → `get_sigma_ratio` (and the field `closest_model['sigmac']` → `'sigma_ratio'`): the formula sets the fitted model's σ to 1, but the fitted σ is not 1 (z-scoring normalizes the variance of x, not of the noise), so the value is the ratio σ_c/σ. The KL rate only depends on that ratio, so db is unaffected.

Not changes, but done for speed or idiom: vectorized loops, a plain-Python Horner evaluation of g(ω) for scalar quad nodes, `lfilter` for the AR simulation, keyword arguments instead of `dbopt`, a NamedTuple output, and `warnings.warn` instead of `disp`. All are covered by tests against literal loop translations.

## Numerical agreement with the MATLAB original (summary of the tests)

| Quantity | Agreement |
|---|---|
| AR fits (kern, sigma, H) of the original estimators (literal translations in the tests, patched into calc_db for all parity checks), `calc_db_euclidean` | ≤ 3e-11 relative (linear algebra round-off) |
| Spike-count bins and counts on the real data | **bit-identical** (apart from the dropped last value) |
| d2 (b = 2) | ≤ 2e-7 relative; the residual is MATLAB `integral`'s own error (Python matches mpmath to 1e-10) |
| sddb (b = 2) | ≤ 5e-9 relative |
| Tutorial ephys values (whole session, WAKE, NREM, REM) | identical to all printed digits with MATLAB's binning and the original estimators; with the defaults (dropped last value, `yw_ar` / `mle_ar` deviations) d2 ≤ 0.05% and sddb ≤ 0.13% different |
| b ≥ 4 | see known issues 1 and 2 |

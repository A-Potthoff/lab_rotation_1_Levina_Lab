# Flagged points and changes in the Python port

The code carries the same tags as comments (`# FLAG[n]: ...`, `# CHANGED[n]: ...`), so `grep -rn "FLAG\[" src/analysis/prox_crit` lists them all.

- **FLAG**: a bug, risk or confusing point in the original. It was **kept as is**, so the Python port behaves like the MATLAB code there.
- **CHANGED**: a change that was required for the Python version to run correctly. Each one is made only where the original author's intent is unambiguous.

Evidence for the statements below comes from `tests/test_prox_crit.py`, which compares against MATLAB output and high-precision mpmath values.

## CHANGED (4)

| # | Where | What and why |
|---|---|---|
| 1 | `kl_rate.get_klr_b` | `quad(..., points=[0.0])` splits the KL integral at ω = 0. On M_b, g_c(0) = 0, which is an integrable log singularity. QUADPACK's first rule on [−π, π] evaluates exactly ω = 0 and returns inf, whereas MATLAB's `integral` never places a node there. The integral's value is unchanged. |
| 2 | `distance.minimize_klr_b` | The constraint `c ≤ 0` (fmincon with `ConstraintTolerance = 0.01`) is passed to SLSQP as `0.01 − c ≥ 0`. SLSQP has no constraint tolerance, and c ≈ 0 holds on M_b by construction (FLAG 3), so a hard `c ≤ 0` can be infeasible from round-off alone. In all tests the constraint is inactive at the optimum. |
| 3 | `distance.calc_db` | When order < b/2, `minimize_klr_b` deliberately returns NaN, exit = 1 and a warning, but MATLAB then crashes reading the missing field `cm.kernc` ("Unrecognized field name"; confirmed in MATLAB). Python returns the intended NaN with exit_status = 1. |
| 4 | `kl_rate.get_klr_b` | For b ≥ 4, `quad` bisects toward ω = 0. For \|ω\| ≲ 1e-8, floating-point cancellation makes g_c exactly 0, so the integrand and then the whole integral become inf. Such a node now contributes 0 (its weight is < ~1e-8). The result matches the exact time-domain KL rate to 1.4e-7. MATLAB's `integral` happens not to probe that close to 0. |

Not logic changes, but done for speed or idiom:
- Vectorized loops.
- A plain-Python Horner evaluation of g(ω) for scalar quad nodes.
- `lfilter` for the AR simulation.
- snake_case names, keyword arguments instead of `dbopt`, a NamedTuple output, and `warnings.warn` instead of `disp`.

All of these are covered by tests against literal loop translations.

## FLAG (16)

| # | Where | What |
|---|---|---|
| 1 | `get_klr_b` | An invalid `int_method` leaves `klr` undefined, so the function errors (MATLAB likewise). |
| 2 | `get_sigmac`, `get_klr_b` | The fitted model's noise σ is implicitly 1, so `sigmac` is really the **ratio** σ_c/σ. The math is consistent (KL depends only on that ratio, confirmed by the time-domain test), but the name is misleading. |
| 3 | `con` | On M_b, z = 1 is always a root, so c = 1 − min\|root\| ≈ 0 permanently. The constraint is degenerate and non-smooth, and its finite-difference gradient is root-finding noise for b ≥ 4 (~1e-8 at b = 4, ~1e-5 at b = 6). |
| 4 | `explosive` | Tolerance 1e-2: a root with \|z\| in [0.99, 1) counts as **not** explosive. |
| 5a | `yw_ar` | The autocovariance is normalized by N − lag, not N, so the Toeplitz matrix Γ is not guaranteed positive definite. |
| 5b | `yw_ar` | Γ's diagonal is fixed to 1, while `calc_db` z-scores with the N − 1 std, so the data's actual lag-0 moment is (N − 1)/N. |
| 5c | `yw_ar` | A single array is not wrapped into a list, unlike `mle_ar`. MATLAB errors on that input; Python raises TypeError. `yw_ar` also assumes z-scored input. |
| 6a | `mle_ar` | The first `order` samples of every segment are regression rows with zero-padded history, and they count in T. |
| 6b | `mle_ar` | Uses `inv(X'X)` instead of a solve. |
| 7a | `qc_plot` | Uses `trapz`, while the optimization used `integral`. 1000 points under-resolve the ω = 0 singularity, so the curve lies ~1% **below** the db line. |
| 7b | `qc_plot` | The ±0.2 scan range is hard-coded. |
| 8 | `minimize_klr_b` | The notes comment is truncated ("To go from"). |
| 9 | `beta_lift` | The right-hand-side vector is named `b`, shadowing the manifold order (renamed `rhs` in Python). |
| 10 | `half_order` | Odd b is never validated in MATLAB, which errors deep inside. Python raises a ValueError up front. |
| 11 | `demo_crit_db.m` | The comment says "calculate d2" but sets b = 6. It also combines `with_err_bars = true` with YuleWalker, which always hits the "not supported" path (exit_status = 1), and contains a dead `if (0)` block. |
| 12 | `get_spike_counts` | `histc` semantics: the last count holds only spikes exactly at t2, so every series ends with a spurious ~0 sample. |
| 13 | `calc_db` error bars | The gradient of db is a forward difference (STEP = 1e-3, O(STEP) bias), and stability is checked only for the +STEP direction. |
| 14 | `calc_db`, `yw_ar` | `sqrt` of a negative number gives a complex result in MATLAB (sddb when H is not negative definite; sigma when 1 − γ·φ < 0). `np.emath.sqrt` reproduces this. |
| 15 | `minimize_klr_b` | **For b ≥ 4 the original's fmincon stops early.** In the reference cases its db is 0.07%, 0.03% and **5.7%** (b = 6) above the true minimum, with kernc off by up to 0.2. This was verified by evaluating mpmath at both solutions and by an independent Nelder-Mead run; the likely cause is FLAG 3. SLSQP reaches the minimum, so **for b ≥ 4 this port gives lower db than MATLAB**, and MATLAB error bars at b ≥ 4 differ strongly (88% in one case). For b = 2 the port agrees with MATLAB to ≤ 2e-7 (db) and ≤ 5e-9 (sddb). |
| 16 | `minimize_klr_b` | The finite-difference gradient (step ~1.5e-8, as in fmincon) of a quadrature-based objective that is only ~1e-7 accurate at b ≥ 4 is noisy, so the optimizer may stop up to ~1e-5 (relative) above the minimum. Error bars at b ≥ 4 inherit this noise. |

## Numerical agreement with the MATLAB original (summary of the tests)

| Quantity | Agreement |
|---|---|
| AR fits (kern, sigma, H), `calc_dbv1` | ≤ 3e-11 relative (linear algebra round-off) |
| Spike-count bins and counts on the real data | **bit-identical** |
| d2 (b = 2) | ≤ 2e-7 relative; the residual is MATLAB `integral`'s own error (Python matches mpmath to 1e-10) |
| sddb (b = 2) | ≤ 5e-9 relative |
| Tutorial ephys values (whole session, WAKE, NREM, REM) | identical to all printed digits |
| b ≥ 4 | see FLAG 15 and 16 |

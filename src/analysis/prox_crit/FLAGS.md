# Flagged points and changes in the Python port

The code carries the same tags as comments (`# FLAG[n]: ...`, `# CHANGED[n]: ...`), so `grep -rn "FLAG\[" src/analysis/prox_crit` lists them all.

- **FLAG**: a bug, risk or questionable choice in the original that can affect results. It was **kept as is**, so the port behaves like the MATLAB code there.
- **CHANGED**: a deliberate deviation from the original's behavior, either needed to run correctly or decided by the user.

Small inconsistencies were simply fixed without a flag, each with a short comment in the code where the reason isn't obvious: misleading names, a truncated comment, an unvalidated odd `b`, an undefined result for an unknown integration method, an explicit matrix inverse, a hard-coded plot range, and a demo script whose settings always failed. They are listed at the end.

Evidence for the statements below comes from `tests/test_prox_crit.py`, which compares against MATLAB output and high-precision mpmath values.

## CHANGED (5)

| # | Where | What and why |
|---|---|---|
| 1 | `kl_rate.get_klr_b` | `quad(..., points=[0.0])` splits the KL integral at ω = 0. On M_b, g_c(0) = 0, which is an integrable log singularity. QUADPACK's first rule on [−π, π] evaluates exactly ω = 0 and returns inf, whereas MATLAB's `integral` never places a node there. The integral's value is unchanged. |
| 2 | `distance.minimize_klr_b` | The constraint `c ≤ 0` (fmincon with `ConstraintTolerance = 0.01`) is passed to SLSQP as `0.01 − c ≥ 0`. SLSQP has no constraint tolerance, and c ≈ 0 holds on M_b by construction (FLAG 1), so a hard `c ≤ 0` can be infeasible from round-off alone. In all tests the constraint is inactive at the optimum. |
| 3 | `distance.calc_db` | When order < b/2, `minimize_klr_b` deliberately returns NaN, exit = 1 and a warning, but MATLAB then crashes reading the missing field `kernc` ("Unrecognized field name"; confirmed in MATLAB). Python returns the intended NaN with exit_status = 1. |
| 4 | `kl_rate.get_klr_b` | For b ≥ 4, `quad` bisects toward ω = 0. For \|ω\| ≲ 1e-8, floating-point cancellation makes g_c exactly 0, so the integrand and then the whole integral become inf. Such a node now contributes 0 (its weight is < ~1e-8). The result matches the exact time-domain KL rate to 1.4e-7. MATLAB's `integral` happens not to probe that close to 0. |
| 5 | `spike_counts.get_spike_counts` | MATLAB's `histc` returns one extra final value counting only spikes exactly at the last edge, so every series ended with a spurious ~0 sample. It is dropped (decided by the user). Effect on the tutorial's d2 values: ≤ 0.07%, far below the error bars. |

## FLAG (10)

| # | Where | What |
|---|---|---|
| 1 | `manifold.con` | On M_b, z = 1 is always a root, so c = 1 − min\|root\| ≈ 0 permanently. The constraint is degenerate and non-smooth, and its finite-difference gradient is root-finding noise for b ≥ 4 (~1e-8 at b = 4, ~1e-5 at b = 6). |
| 2 | `manifold.explosive` | Tolerance 1e-2: a root with \|z\| in [0.99, 1) counts as **not** explosive. |
| 3 | `ar.yw_ar` | The autocovariance is normalized by N − lag, not N, so the Toeplitz matrix Γ is not guaranteed positive definite. |
| 4 | `ar.yw_ar` | Γ's diagonal is fixed to 1, while `calc_db` z-scores with the N − 1 std, so the data's actual lag-0 moment is (N − 1)/N. |
| 5 | `ar.mle_ar` | The first `order` samples of every segment are regression rows with zero-padded history (zero = the mean after z-scoring), and they count in T. A small bias that grows with many short segments. |
| 6 | `plotting.qc_plot` | Uses `trapz`, while the optimization used `integral`. 1000 points under-resolve the ω = 0 singularity, so the curve lies ~1% **below** the db line. Judge the location of the minimum, not its height. |
| 7 | `distance.calc_db` error bars | The gradient of db is a forward difference (step 1e-3, bias of that order), and stability is checked only for the + direction. |
| 8 | `distance.calc_db`, `ar.yw_ar` | `sqrt` of a negative number gives a complex result in MATLAB (sddb when H is not negative definite; sigma when 1 − γ·φ < 0). `np.emath.sqrt` reproduces this. |
| 9 | `distance.minimize_klr_b` | **For b ≥ 4 the original's fmincon stops early.** In the reference cases its db is 0.07%, 0.03% and **5.7%** (b = 6) above the true minimum, with kernc off by up to 0.2. This was verified by evaluating mpmath at both solutions and by an independent Nelder-Mead run; the likely cause is FLAG 1. SLSQP reaches the minimum, so **for b ≥ 4 this port gives lower db than MATLAB**, and MATLAB error bars at b ≥ 4 differ strongly (88% in one case). For b = 2 the port agrees with MATLAB to ≤ 2e-7 (db) and ≤ 5e-9 (sddb). |
| 10 | `distance.minimize_klr_b` | The finite-difference gradient (step ~1.5e-8, as in fmincon) of a quadrature-based objective that is only ~1e-7 accurate at b ≥ 4 is noisy, so the optimizer may stop up to ~1e-5 (relative) above the minimum. Error bars at b ≥ 4 inherit this noise. |

## Fixed without a flag (no effect on results)

| Original | Python |
|---|---|
| `get_sigmac` / `sigmac`: suggests the critical model's absolute noise std, but it is the ratio σ_c/σ (the fitted σ is set to 1 in the formula) | `get_sigma_ratio` / `"sigma_ratio"` |
| `cntr`, `b` (as right-hand side in `beta_lift`, clashing with the manifold order), `temp`, `cme`, ... | descriptive names (`counter`, `rhs`, ...) |
| `get_klr_b` with an unknown `int_method` prints an error and then fails on an undefined result | raises `ValueError` with the same message |
| `yw_ar` errors on a single array (needs a cell array), unlike `mle_ar` | both accept a single array or a list of segments |
| `inv(X'X) X'Y` and `inv(H)` | `np.linalg.solve` (same result, numerically more stable) |
| QC plot scan range fixed to ±0.2 | parameter `scan_width=0.2` |
| `minimize_klr_b` notes end mid-sentence ("To go from") | completed (reduced ↔ full kernel via `beta_lift` / `beta_project`) |
| odd `b` is not validated and errors deep inside | `half_order` raises a clear `ValueError` |
| `demo_crit_db.m`: comment says "d2" with b = 6; requests error bars with YuleWalker (always fails); dead `if (0)` block | notebook says d6, no error bars requested, dead block omitted |

Not changes, but done for speed or idiom: vectorized loops, a plain-Python Horner evaluation of g(ω) for scalar quad nodes, `lfilter` for the AR simulation, keyword arguments instead of `dbopt`, a NamedTuple output, and `warnings.warn` instead of `disp`. All are covered by tests against literal loop translations.

## Numerical agreement with the MATLAB original (summary of the tests)

| Quantity | Agreement |
|---|---|
| AR fits (kern, sigma, H), `calc_dbv1` | ≤ 3e-11 relative (linear algebra round-off) |
| Spike-count bins and counts on the real data | **bit-identical** (apart from the dropped last value, CHANGED 5) |
| d2 (b = 2) | ≤ 2e-7 relative; the residual is MATLAB `integral`'s own error (Python matches mpmath to 1e-10) |
| sddb (b = 2) | ≤ 5e-9 relative |
| Tutorial ephys values (whole session, WAKE, NREM, REM) | identical to all printed digits with MATLAB's binning; ≤ 0.07% different with CHANGED 5 |
| b ≥ 4 | see FLAG 9 and 10 |

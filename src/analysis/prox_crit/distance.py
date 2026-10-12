"""Proximity to criticality d_b: main entry point ``calc_db``.

NOT OWN WORK. Python translation of ``calc_db`` and ``minimize_klr_b`` from ``calc_db.m``, prox_crit_toolkit, J. S. Sooter et al. (https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09), licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude). Logic unchanged except CHANGED[2] and CHANGED[3]; see FLAGS.md.

Pipeline of ``calc_db``:

1. z-score the time series (one mean / std for all segments together).
2. Fit an AR(order) model -> kern, sigma (Yule-Walker or maximum likelihood).
3. d_b = minimum over critical kernels kernc in M_b of the KL rate klr(kern -> kernc) (minimize_klr_b, started from the L2-closest point).
4. Convert nats per time step -> bits per second: d_b * log2(e) / delta_t.
5. Optional error bar (delta method): sd = sqrt(-grad(d_b)' H^-1 grad(d_b)), gradient by forward finite differences, H = log-likelihood Hessian (maximum likelihood fit only).

Abbreviations: db = distance (proximity) to criticality of type b, sddb = standard deviation (error bar) of db, kern = fitted history kernel, kernc = kernel of the closest critical model, klr = KL rate, H = Hessian, QC = quality control.
"""

import warnings
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from .ar import as_segments, mle_ar, yw_ar
from .kl_rate import get_klr_b, get_sigma_ratio
from .manifold import beta_lift, beta_project, calc_dbv1, con, explosive, half_order

# Optimizer. The original uses MATLAB's fmincon with
#   optimoptions('fmincon', 'Display','none', 'Algorithm','sqp', 'ConstraintTolerance',0.01, 'MaxFunctionEvaluations',10000, 'StepTolerance',1e-15)
# Translation: scipy's SLSQP (also an SQP method; finite-difference gradients like fmincon). maxiter = fmincon's default MaxIterations for 'sqp'. SLSQP has no step or constraint tolerance options; ftol (convergence threshold on the change of the objective) is chosen tight so that SLSQP stops at the minimizer.
CONSTRAINT_TOLERANCE = 0.01
SLSQP_OPTIONS = dict(maxiter=400, ftol=1e-12)
NATS_TO_BITS = np.log2(np.e)


class DbResult(NamedTuple):
    """Outputs of calc_db (same order as the MATLAB output list, so it unpacks the same way)."""

    db: float  # proximity to criticality of type b, bits/s
    sddb: float  # error bar for db (nan unless with_err_bars and MaxLikelihood)
    kern: np.ndarray  # history kernel of the best-fit AR model
    sigma: float  # noise standard deviation of the best-fit AR model
    H: np.ndarray  # Hessian of the log-likelihood (nan unless MaxLikelihood)
    kernc: np.ndarray  # kernel of the closest (in KL rate) AR model in M_b
    exit_status: int  # 0 if all calculations were successful, 1 otherwise


def calc_db(
    x,
    order,
    delta_t,
    b,
    *,
    fit_method="YuleWalker",
    with_err_bars=False,
    with_qc=False,
    with_parallel=False,
):
    """Proximity to criticality of type b, in bits/s.

    Parameters
    ----------
    x : array or list of arrays
        The observed time series. A segmented time series is passed as a list of segments, e.g. ``[np.array([1, 2, 3]), np.array([10, 5, 0])]``; one AR model is fitted to all segments jointly.
    order : int
        AR model order.
    delta_t : float
        Time between consecutive points of the time series, in seconds.
    b : int
        Type of criticality to calculate proximity to (b = 2, 4, 6, ..., 2*order).
    fit_method : {'YuleWalker', 'MaxLikelihood'}, default 'YuleWalker'
        AR fitting method.
    with_err_bars : bool, default False
        If True, calculate the error bar of db (MaxLikelihood only).
    with_qc : bool, default False
        If True, make quality control plots (see plotting.qc_plot).
    with_parallel : bool, default False
        Accepted for compatibility with the original (parfor over the gradient components); always computed sequentially here, results are identical.

    Returns
    -------
    DbResult
        Named tuple (db, sddb, kern, sigma, H, kernc, exit_status).
    """
    exit_status = 0
    x = as_segments(x)

    # z-score with the pooled mean and std of all segments. Centering is essential: the AR model has no intercept, and an uncentered series (e.g. counts around 20) could only be fitted with kernel weights summing to ~1, which is the criticality condition. Scaling gives unit variance, which yw_ar assumes; d_b itself is scale-invariant. ddof=1: MATLAB's std divides by N-1.
    pooled = np.concatenate(x)
    mean, std = np.mean(pooled), np.std(pooled, ddof=1)
    x = [(seg - mean) / std for seg in x]

    # Best-fit AR model.
    if fit_method == "MaxLikelihood":
        kern, sigma, H = mle_ar(x, order)
    elif fit_method == "YuleWalker":
        kern, sigma = yw_ar(x, order)
        H = np.nan
    else:
        warnings.warn(
            "Error: invalid fit_method. Change to 'MaxLikelihood' or 'YuleWalker'"
        )
        return DbResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1)

    # The KL rate needs a stationary fitted model.
    if explosive(kern):
        warnings.warn("Error: estimated model is explosive.")
        return DbResult(np.nan, np.nan, kern, sigma, H, np.nan, 1)

    db, closest_model, exit_flag = minimize_klr_b(kern, b)  # db in nats per time step
    if exit_flag == 1:
        exit_status = 1

    # CHANGED[3]: the original reads the field kernc here, which does not exist when minimize_klr_b took its "invalid model order" branch (order < b/2), so MATLAB crashes right after the intended warning. That branch clearly intends NaN output with exit status 1.
    kernc = closest_model.get("kernc", np.nan)

    db = NATS_TO_BITS / delta_t * db

    if with_qc:
        from .plotting import (
            qc_plot,
        )  # imported here so matplotlib is only needed for QC plots

        qc_plot(kern, kernc, db, b, delta_t)

    if not with_err_bars:
        return DbResult(db, np.nan, kern, sigma, H, kernc, exit_status)

    # Error bars need the Hessian H, which only the maximum likelihood fit provides.
    if fit_method == "YuleWalker":
        warnings.warn(
            "Error: error bar calculation not supported with YuleWalker fitting. Switch to MaxLikelihood if you want error bars."
        )
        return DbResult(db, np.nan, kern, sigma, H, kernc, 1)

    # Gradient of db with respect to kern by forward differences: perturb each kernel coefficient by `step` and rerun the whole minimization. If a perturbed kernel would be explosive, shrink the step by 10 and check all directions again.
    # FLAG[7]: one-sided differences (bias of order step), and stability is only checked in the + direction.
    step = 0.001
    unit_vectors = np.eye(order)
    while any(explosive(kern + step * unit_vectors[j]) for j in range(order)):
        step = step / 10

    grad_db = np.zeros(order)
    for j in range(order):
        db_perturbed, _, exit_flag = minimize_klr_b(kern + step * unit_vectors[j], b)
        if exit_flag == 1:
            exit_status = 1
        grad_db[j] = (NATS_TO_BITS / delta_t * db_perturbed - db) / step

    # Delta method: Var(db) ~ grad' Cov(kern) grad with Cov(kern) ~ -H^-1.
    # FLAG[8]: if H is not negative definite the argument is negative; np.emath.sqrt then returns a complex number, as MATLAB does.
    sddb = np.emath.sqrt(-grad_db @ np.linalg.solve(H, grad_db))
    return DbResult(db, sddb, kern, sigma, H, kernc, exit_status)


def minimize_klr_b(kern, b):
    """Minimize the KL divergence rate from ``kern`` over the critical manifold M_b.

    Within this function and the functions it calls, kernels on M_b are represented by their reduced kernel (first n - b/2 entries): ``beta_lift`` completes a reduced kernel to the full kernel, ``beta_project`` truncates a full kernel to the reduced one.

    Parameters
    ----------
    kern : (n,) array
        Kernel of an AR(n) model (n is determined by the length).
    b : int (even)
        Order of the critical manifold M_b (2, 4, 6, ...).

    Returns
    -------
    db : float
        Distance from M_b: the minimal KL rate, in nats per time step.
    closest_model : dict
        'kernc': full kernel of the closest critical model; 'sigma_ratio': its noise std relative to the fitted model's (see kl_rate.get_sigma_ratio).
    exit_flag : int
        0 if successful.
    """
    kern = np.asarray(kern, dtype=float).ravel()
    exit_flag = 0
    closest_model = {}
    half = half_order(b)

    if len(kern) < half:
        exit_flag = 1
        db = np.nan
        warnings.warn("WARNING: invalid model order")
    elif len(kern) == half:
        # M_b is a single point here (no free parameters), so there is nothing to optimize.
        db = get_klr_b(kern, [], b, "integral")
        kernc = beta_lift([], b)
        closest_model["sigma_ratio"] = get_sigma_ratio(kern, kernc)
        closest_model["kernc"] = kernc
    else:

        def objective(kernc_reduced):
            return get_klr_b(kern, kernc_reduced, b, "integral")

        # Start at the L2-closest point of M_b.
        _, kernc_l2 = calc_dbv1(kern, b)
        kernc_start = beta_project(kernc_l2, b)

        # CHANGED[2]: the original's constraint is c <= 0 with fmincon's ConstraintTolerance 0.01. SLSQP has no tolerance option and treats its inequality fun(x) >= 0 as hard. Since c ~ 0 with round-off noise of either sign on M_b by construction (FLAG[1]), a hard c <= 0 can be infeasible; the constraint is therefore passed as 0.01 - c >= 0.
        constraint = {
            "type": "ineq",
            "fun": lambda kernc_reduced: CONSTRAINT_TOLERANCE
            - con(kernc_reduced, b)[0],
        }
        result = minimize(
            objective,
            kernc_start,
            method="SLSQP",
            constraints=[constraint],
            options=SLSQP_OPTIONS,
        )
        # As in the original, the optimizer's own success flag is not checked; only the stability check below can set exit_flag.
        # FLAG[9]: for b >= 4 the ORIGINAL (fmincon) stops early, barely past this L2 starting point: in the reference cases its db is up to ~6% above the true minimum (verified with mpmath and Nelder-Mead; likely cause FLAG[1]). SLSQP reaches the minimum, so for b >= 4 this port gives LOWER db than MATLAB; for b = 2 they agree to 2e-7 (tests).
        # FLAG[10]: the gradients are finite differences (step ~1.5e-8, as in fmincon) of an objective computed by adaptive quadrature, which at b >= 4 is only ~1e-7 accurate. The noisy gradient lets the optimizer stop up to ~1e-5 (relative) above the minimum, and error bars (FLAG[7]) inherit this noise for b >= 4.
        db = result.fun

        closest_model["kernc"] = beta_lift(result.x, b)
        closest_model["sigma_ratio"] = get_sigma_ratio(kern, closest_model["kernc"])

        if explosive(closest_model["kernc"]):
            exit_flag = 1
            db = np.nan
            warnings.warn("ERROR: drifted off of critical manifold.")

    return db, closest_model, exit_flag

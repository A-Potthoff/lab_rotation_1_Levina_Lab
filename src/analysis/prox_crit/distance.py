"""Proximity to criticality d_b: main entry point ``calc_db``.

NOT OWN WORK. Python translation of ``calc_db`` and ``minimize_klr_b`` from
``calc_db.m``, prox_crit_toolkit, J. S. Sooter et al.
(https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09),
licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude).
Logic unchanged except CHANGED[2], CHANGED[3]; see FLAGS.md.

Pipeline of ``calc_db``:

1. z-score the time series (all segments pooled).
2. Fit an AR(order) model -> kern, sigma (Yule-Walker or max. likelihood).
3. d_b = min over critical kernels kernc in M_b of the KL rate
   klr(kern -> kernc)  (minimize_klr_b; started from the L2-closest point).
4. Convert nats/time step -> bits/s:  d_b * log2(e) / deltaT.
5. Optional error bar (delta method):  sd = sqrt(-grad(d_b)' H^-1 grad(d_b)),
   grad by forward finite differences, H = log-likelihood Hessian (MLE only).

Abbreviations: db = distance (proximity) to criticality of type b,
sddb = standard deviation (error bar) of db, kern = fitted history kernel,
kernc = kernel of the closest critical model, klr = KL rate, H = Hessian,
QC = quality control, cm = closest model.
"""

import warnings
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from .ar import _as_segments, mle_ar, yw_ar
from .kl_rate import get_klr_b, get_sigmac
from .manifold import beta_lift, beta_project, calc_dbv1, con, explosive, half_order

# Optimizer settings. MATLAB:
#   optimoptions('fmincon', 'Display','none', 'Algorithm','sqp', 'ConstraintTolerance',0.01, 'MaxFunctionEvaluations',10000, 'StepTolerance',1e-15)
# Translation: scipy SLSQP (an SQP method, finite-difference gradients like
# fmincon). maxiter = fmincon's default MaxIterations for 'sqp' (400).
# SLSQP has no step/constraint tolerance options; FTOL is its convergence
# threshold on the change of the objective, chosen tight so that SLSQP stops
# at the minimizer (like fmincon's first-order-optimality stop).
CONSTRAINT_TOLERANCE = 0.01
SLSQP_OPTIONS = dict(maxiter=400, ftol=1e-12)


class DbResult(NamedTuple):
    """Outputs of calc_db (same order as the MATLAB output list)."""

    db: float  # proximity to criticality of type b, bits/s
    sddb: float  # error bar for db (nan unless with_err_bars & MaxLikelihood)
    kern: np.ndarray  # history kernel of the best-fit AR model
    sigma: float  # noise standard deviation of the best-fit AR model
    H: np.ndarray  # Hessian of the log-likelihood (nan unless MaxLikelihood)
    kernc: np.ndarray  # kernel of the closest (KL rate) AR model in M_b
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
        The observed time series. If the time series is segmented, pass the segments as a list, e.g. ``[np.array([1, 2, 3]), np.array([10, 5, 0])]`` means two separate segments (MATLAB: cell array).
    order : int
        The AR model order.
    delta_t : float
        Time between consecutive points in the time series, in seconds (MATLAB: deltaT).
    b : int
        Type of criticality to calculate proximity to (b = 2, 4, 6, ..., 2*order).
    fit_method : {'YuleWalker', 'MaxLikelihood'}, default 'YuleWalker'
        AR fitting method.
    with_err_bars : bool, default False
        If True, calculate error bars for db (MaxLikelihood only).
    with_qc : bool, default False
        If True, make quality control plots (MATLAB: with_QC).
    with_parallel : bool, default False
        Accepted for API parity with MATLAB (``parfor`` over the gradient components). NOTE: always computed sequentially here; results are identical.

    Returns
    -------
    DbResult
        Named tuple (db, sddb, kern, sigma, H, kernc, exit_status); unpacks
        like the MATLAB output list.
    """
    exit_status = 0

    # If x is a single time series: cast as a list with one segment.
    x = _as_segments(x)

    # z-score with the pooled mean / std of all segments.
    # MATLAB std normalises by N-1 -> ddof=1.
    tot = np.concatenate(x)
    mu, sd = np.mean(tot), np.std(tot, ddof=1)
    x = [(seg - mu) / sd for seg in x]

    # Find the best-fit AR(n) model to x.
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

    # Check whether the best-fit model is explosive (unstable).
    if explosive(kern):
        warnings.warn("Error: estimated model is explosive.")
        return DbResult(np.nan, np.nan, kern, sigma, H, np.nan, 1)

    # Calculate db (in nats per time step).
    db, cm, exit_ = minimize_klr_b(kern, b)
    if exit_ == 1:
        exit_status = exit_

    # Closest critical AR model.
    # CHANGED[3]: MATLAB reads cm.kernc, a field that does not exist when
    # minimize_klr_b took its "invalid model order" branch (order < b/2), so
    # MATLAB crashes right after the intended warning. That branch clearly
    # intends NaN output with exit = 1, so the missing field becomes NaN here.
    kernc = cm.get("kernc", np.nan)

    # Change units: nats / time step -> bits / s.
    db = np.log2(np.e) / delta_t * db

    # Quality control plot (check optimization).
    if with_qc:
        from .plotting import qc_plot  # lazy: matplotlib only when needed

        qc_plot(kern, kernc, db, b, delta_t)

    # Calculate error bars for db.
    if with_err_bars:
        # fit_method must be 'MaxLikelihood' to calculate error bars (needs H).
        if fit_method == "YuleWalker":
            warnings.warn(
                "Error: error bar calculation not supported with YuleWalker "
                "fitting. Switch to MaxLikelihood if you want error bars."
            )
            return DbResult(db, np.nan, kern, sigma, H, kernc, 1)

        # Gradient of db with respect to kern.
        graddb = np.zeros(order)

        # Step size for the numerical gradient. If a step would push kern out
        # of the stable region, reduce STEP by 10 and check all directions again.
        # FLAG[13]: forward differences (one-sided, O(STEP) bias); only the +STEP
        # direction is checked for stability.
        STEP = 0.001
        unit = np.eye(order)  # unit[j] = [0 ... 1 ... 0] (MATLAB: 1 at position j)
        check = True
        while check:
            found_problem = False
            for j in range(order):
                if explosive(kern + STEP * unit[j]):
                    found_problem = True  # go back and check again
                    STEP = STEP / 10
                    break
            # At this point STEP was checked in all directions;
            # none of them results in an explosive model.
            if not found_problem:
                check = False

        # MATLAB has a parfor branch (with_parallel) and a for branch with the
        # same arithmetic; both set exit_status = 1 if any sub-run failed.
        for j in range(order):
            # db at the perturbed kernel kern + STEP * e_j.
            newdb, _, exit_ = minimize_klr_b(kern + STEP * unit[j], b)
            if exit_ == 1:
                exit_status = 1
            newdb = np.log2(np.e) / delta_t * newdb
            graddb[j] = (newdb - db) / STEP  # forward-difference gradient

        # Error bar (delta method): sd = sqrt(-grad' H^-1 grad).
        # FLAG[14]: negative argument (H not negative definite) gives a
        # complex number in MATLAB; np.emath.sqrt reproduces that.
        sddb = np.emath.sqrt(-graddb @ np.linalg.inv(H) @ graddb)
    else:
        sddb = np.nan

    return DbResult(db, sddb, kern, sigma, H, kernc, exit_status)


def minimize_klr_b(kern, b):
    """Minimize the KL div. rate over the desired critical manifold M_b.

    Parameters
    ----------
    kern : (n,) array
        Kernel of an AR(n) model (n is determined by the length).
    b : int (even)
        Order of the critical manifold M_b (2, 4, 6, ...).

    Returns
    -------
    db : float
        Distance from M_b (min KL rate, nats per time step).
    closest_model : dict
        'kernc': kernel of the closest model; 'sigmac': its noise std
        (relative to the fitted model's sigma, see FLAG[2]).
    exit_flag : int
        Exit flag (0 = successful). MATLAB name: exit (renamed: Python builtin).

    Notes
    -----
    Within this function and the functions it calls, kernels on M_b are
    represented by a reduced kernel (just the first n - b/2 entries); see
    manifold.beta_lift / beta_project.
    FLAG[8]: the original notes end mid-sentence ("To go from"); the omitted
    part presumably describes beta_lift / beta_project.
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
        # M_b is a single point here (zero free parameters): no optimization.
        db = get_klr_b(kern, [], b, "integral")
        temp = beta_lift([], b)
        closest_model["sigmac"] = get_sigmac(kern, temp)
        closest_model["kernc"] = temp
    else:
        # Objective: KL rate as a function of the REDUCED critical kernel.
        # (The original also has a commented-out 'trapz' variant.)
        def objective(kernc):
            return get_klr_b(kern, kernc, b, "integral")

        # Starting point: solve the linear system that minimizes the L2
        # distance to M_b, then project to the reduced kernel.
        _, cme = calc_dbv1(kern, b)
        kernc0 = beta_project(cme, b)

        # MATLAB: fmincon(objective, kernc0, [],[],[],[],[],[], nonlcon, options)
        # with nonlcon(kernc) = con(kernc, b), feasible if c <= 0.
        # CHANGED[2]: SLSQP has no constraint tolerance; its inequality is a
        # hard  fun(x) >= 0. Because c ~ 0 (with round-off noise of either
        # sign) on M_b by construction (FLAG[3]), a hard c <= 0 can be
        # infeasible. fmincon only demands c <= ConstraintTolerance (0.01),
        # so the constraint is passed as  0.01 - c >= 0.
        constraint = {
            "type": "ineq",
            "fun": lambda kernc: CONSTRAINT_TOLERANCE - con(kernc, b)[0],
        }
        res = minimize(
            objective,
            kernc0,
            method="SLSQP",
            constraints=[constraint],
            options=SLSQP_OPTIONS,
        )
        # As in the original, the optimizer's own exit flag is not checked;
        # only the final stability check below can set exit_flag = 1.
        # FLAG[15]: for b >= 4 the ORIGINAL (fmincon) stops early, barely past
        # this L2 starting point: in the reference cases its db is up to ~6%
        # above the true minimum (verified with mpmath + Nelder-Mead; likely
        # cause FLAG[3]). SLSQP reaches the minimum, so for b >= 4 this port
        # gives LOWER db than MATLAB; b = 2 agrees to 1e-7 (tests).
        # FLAG[16]: gradients are finite differences (step ~1.5e-8, as in
        # fmincon) of an objective computed by adaptive quadrature, which at
        # b >= 4 is only ~1e-7 accurate; the noisy gradient lets the optimizer
        # stop up to ~1e-5 (relative) above the minimum. Error bars (forward
        # differences of db, FLAG[13]) inherit this noise for b >= 4.
        x_opt, db = res.x, res.fun

        # Lift from the reduced kernel.
        closest_model["kernc"] = beta_lift(x_opt, b)
        closest_model["sigmac"] = get_sigmac(kern, closest_model["kernc"])

        if explosive(closest_model["kernc"]):
            exit_flag = 1
            db = np.nan
            warnings.warn("ERROR: drifted off of critical manifold.")

    return db, closest_model, exit_flag

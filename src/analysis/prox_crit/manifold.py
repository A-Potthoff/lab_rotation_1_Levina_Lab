"""Geometry of the critical manifold M_b of AR(n) models.

NOT OWN WORK. Python translation of ``calc_dbv1.m`` and of the local functions
``beta_project``, ``beta_lift``, ``con`` and ``explosive`` from ``calc_db.m``,
prox_crit_toolkit, J. S. Sooter et al.
(https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09),
licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude).
Logic unchanged; see FLAGS.md for flagged points.

Background (for recognition, no derivation):

* AR polynomial:  P(z) = 1 - sum_{k=1..n} kern_k z^k.
  Stationary (stable)  <=>  all roots of P lie outside the unit circle.
* Critical manifold of type b (b = 2, 4, 6, ...), M_b: z = 1 is a root of P of
  multiplicity b/2.  Equivalent linear constraints on the kernel:
      sum_k kern_k       = 1                       (P(1) = 0)
      sum_k k^j kern_k   = 0,  j = 1 .. b/2 - 1    (derivatives of P vanish)
  M_b is therefore an affine subspace of dimension n - b/2.
* "Reduced kernel": a point on M_b is parameterised by its first n - b/2
  entries; the last b/2 entries follow from the constraints (beta_lift).
  beta_project goes the other way (just truncates).

Abbreviations: b / beta = type (order) of criticality, kernc / phic = kernel of
a critical model, dbv1 = "version 1" of the distance: plain L2 distance,
con = constraint function for the optimizer.
"""

import warnings

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.special import comb


def half_order(b):
    """Return b/2 as an int (number of constraints defining M_b).

    FLAG[10]: the MATLAB code never checks that b is even; an odd b makes it
    error deep inside (e.g. ``nchoosek``/``zeros(b/2)``). Here the same
    situation raises a clear ValueError instead (no valid input changes).
    """
    if b % 2 != 0:
        raise ValueError(f"b must be even (b = 2, 4, 6, ...), got b = {b}")
    return int(b // 2)


def _ar_roots(kern):
    """Roots of the AR polynomial P(z) = 1 - sum_k kern_k z^k.

    MATLAB: roots([fliplr(kern(:)') -1]); coefficients are ordered from the
    highest power (z^n: kern_n) down to the constant (-1), i.e. -P(z). Both
    MATLAB ``roots`` and ``np.roots`` drop leading zeros the same way.
    """
    kern = np.asarray(kern, dtype=float).ravel()
    return np.roots(np.r_[kern[::-1], -1.0])


def explosive(kern):
    """Check whether the AR model with history kernel ``kern`` is explosive.

    i.e. whether any root of the AR polynomial lies inside the unit circle
    (min |root| < 1 - ERROR).

    FLAG[4]: tolerance ERROR = 1e-2, so models with a root in [0.99, 1) count
    as NOT explosive.
    """
    ERROR = 1e-2
    r = _ar_roots(kern)
    if r.size == 0:
        # NOTE: no roots (all-zero kernel). In MATLAB min([]) = [] and
        # `if []` is false, i.e. "not explosive"; reproduced explicitly here.
        return False
    return bool(np.min(np.abs(r)) < 1 - ERROR)


def beta_project(phic, beta):
    """Project a point on the critical manifold into the reduced space.

    Simply keeps the first n - beta/2 kernel entries.
    """
    phic = np.asarray(phic, dtype=float).ravel()
    n = len(phic)  # model order
    half = half_order(beta)
    if half > n:
        warnings.warn("ERROR (beta_project): beta is too large ")
        return np.array([])
    return phic[: n - half]  # MATLAB: phic(1:(n-beta/2))


def beta_lift(phic_reduc, beta):
    """Find the point on critical manifold M_b given a reduced kernel.

    Parameters
    ----------
    phic_reduc : (n - beta/2,) array
        First n - beta/2 kernel coefficients.
    beta : int
        Order of the critical manifold.

    Returns
    -------
    phic : (n,) array
        Full length-n kernel. Uniquely determined by the input kernel, by
        applying the linear constraints outlined in Sooter et al.:
        the last beta/2 entries solve  A @ tail = rhs  with
            row 0:   sum(tail)                = 1 - sum(phic_reduc)
            row i:   sum_k k^i tail_k          = -sum_k k^i phic_reduc_k
        (k = true lag number, 1-based), i.e. the moment constraints of M_b.
    """
    phic_reduc = np.asarray(phic_reduc, dtype=float).ravel()
    half = half_order(beta)
    n = len(phic_reduc) + half  # model order

    # Lag numbers (1-based, as in the math) of the free and of the solved entries.
    lags_reduc = np.arange(1, len(phic_reduc) + 1, dtype=float)  # 1 .. n-beta/2
    lags_tail = np.arange(n - half + 1, n + 1, dtype=float)  # n-beta/2+1 .. n

    A = np.zeros((half, half))
    # FLAG[9]: the right-hand side is called `b` in MATLAB, shadowing the
    # manifold order b used everywhere else; renamed `rhs` here for clarity.
    rhs = np.zeros(half)
    for i in range(half):  # MATLAB i = 1..beta/2 with power (i-1) -> here power i
        A[i, :] = lags_tail**i
        if i == 0:
            rhs[i] = 1 - np.sum(phic_reduc)
        else:
            rhs[i] = -(lags_reduc**i) @ phic_reduc

    return np.concatenate([phic_reduc, np.linalg.solve(A, rhs)])  # MATLAB: A\b


def con(kernc, b):
    """Constraints for the optimization problem (fmincon convention).

    Returns
    -------
    c : float
        Inequality constraint, feasible if c <= 0: the zeros of the AR
        polynomial of the lifted kernel must lie outside the unit disk,
        c = 1 - min |root|.
    ceq : empty array
        Equality constraints (ceq = 0); currently none.

    FLAG[3]: on M_b, z = 1 is ALWAYS a root (multiplicity b/2), so c >= 0 by
    construction and the constraint is permanently active, c ~ 0 up to
    root-finding round-off (~1e-16 for b=2, ~1e-8 for b=4, ~1e-5 for b=6).
    It is also non-smooth (min over roots). The original works because
    fmincon accepts c <= ConstraintTolerance = 0.01 (see CHANGED[2]).
    """
    kernc = beta_lift(kernc, b)
    c = 1 - np.min(np.abs(_ar_roots(kernc)))
    ceq = np.array([])
    return c, ceq


def calc_dbv1(kern, b):
    """Find the closest kernel on the beta manifold in L2 distance.

    This is also used as the initial condition of the KL-divergence
    minimization (minimize_klr_b).

    M_b is spanned (as an affine hull) by the n - b/2 + 1 kernels of the
    critical models (1 - L)^k x_t = eps_t, k = b/2 .. n, whose kernels are
        kern_t = C(k, t) (-1)^(t+1)   for t <= k,   0 otherwise
    (P(z) = (1 - z)^k has a k-fold root at z = 1, and k >= b/2). The closest
    point is the least-squares projection of ``kern`` onto that affine hull
    (via QR).

    Returns
    -------
    dbv1 : float
        L2 distance ||kern - kernc||.
    kernc : (n,) array
        Closest kernel on M_b.
    """
    m = half_order(b) - 1
    order = len(kern)

    # Construct points: each column is one point (one history kernel).
    # Column index (k - m) in MATLAB (1-based) -> k - m - 1 here.
    x = np.zeros((order, order - m))
    for k in range(m + 1, order + 1):  # MATLAB: k = (m+1):order
        for t in range(1, k + 1):  # lag t (1-based) -> row t-1
            x[t - 1, k - m - 1] = comb(k, t, exact=True) * (-1) ** (t + 1)
        # rows t = k+1..order stay 0 (explicitly zeroed in MATLAB)

    # Affine hull: directions relative to the first point.
    X = x - x[:, [0]]
    X = X[:, 1:]

    kern = np.asarray(kern, dtype=float).ravel()
    bhat = kern - x[:, 0]

    # QR decomposition: least-squares solution of X v = bhat.
    # MATLAB uses the full QR and R \ (Q'*bhat); with full column rank this
    # equals the economic-QR triangular solve used here.
    Q, R = qr(X, mode="economic")
    v_qr = solve_triangular(R, Q.T @ bhat)

    kernc = x[:, 0] + X @ v_qr
    dbv1 = np.sqrt(np.sum((kern - kernc) ** 2))
    return dbv1, kernc

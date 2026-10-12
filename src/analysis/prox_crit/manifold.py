"""Geometry of the critical manifold M_b of AR(n) models.

NOT OWN WORK. Python translation of ``calc_dbv1.m`` and of the local functions ``beta_project``, ``beta_lift``, ``con`` and ``explosive`` from ``calc_db.m``, prox_crit_toolkit, J. S. Sooter et al. (https://github.com/sam-sooter/prox_crit_toolkit, fetched 2026-10-09), licensed CC BY-NC 4.0. Translated by Andre Potthoff (with Claude). See FLAGS.md for flagged points.

Background (for recognition, no derivation):

* AR polynomial: P(z) = 1 - sum_{k=1..n} kern_k z^k. The model is stationary (stable) iff all roots of P lie outside the unit circle.
* Critical manifold of type b (b = 2, 4, 6, ...), M_b: z = 1 is a root of P with multiplicity b/2. Equivalent linear constraints on the kernel::

      sum_k kern_k       = 1                       (P(1) = 0)
      sum_k k^j kern_k   = 0,  j = 1 .. b/2 - 1    (derivatives of P vanish at 1)

  M_b is therefore an affine subspace of dimension n - b/2.
* Reduced kernel: a point on M_b is parameterized by its first n - b/2 entries; the last b/2 entries follow from the constraints (``beta_lift``). ``beta_project`` goes the other way (it just truncates).

Abbreviations: b / beta = type (order) of criticality, kernc / phic = kernel of a critical model, dbv1 = "version 1" of the distance (plain L2 distance), con = constraint function of the optimizer.
"""

import warnings

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.special import comb


def half_order(b):
    """Return b/2 as an int: the number of linear constraints defining M_b (b must be even)."""
    if b % 2 != 0:
        raise ValueError(f"b must be even (b = 2, 4, 6, ...), got b = {b}")
    return int(b // 2)


def ar_roots(kern):
    """Roots of the AR polynomial P(z) = 1 - sum_k kern_k z^k.

    np.roots expects coefficients from the highest power down: (kern_n, ..., kern_1, -1), i.e. the coefficients of -P(z), which has the same roots.
    """
    kern = np.asarray(kern, dtype=float).ravel()
    return np.roots(np.r_[kern[::-1], -1.0])


def explosive(kern):
    """True if the AR model with history kernel ``kern`` is explosive (a root of P inside the unit circle).

    FLAG[2]: tolerance 1e-2, so a model with a root at |z| in [0.99, 1) counts as NOT explosive.
    """
    tolerance = 1e-2
    roots = ar_roots(kern)
    if (
        roots.size == 0
    ):  # all-zero kernel: P(z) = 1 has no roots, the model is white noise
        return False
    return bool(np.min(np.abs(roots)) < 1 - tolerance)


def beta_project(phic, beta):
    """Project a kernel on M_beta to the reduced space: keep its first n - beta/2 entries."""
    phic = np.asarray(phic, dtype=float).ravel()
    n = len(phic)  # model order
    half = half_order(beta)
    if half > n:
        warnings.warn("ERROR (beta_project): beta is too large ")
        return np.array([])
    return phic[: n - half]


def beta_lift(phic_reduc, beta):
    """Complete a reduced kernel to the unique full kernel on M_beta.

    The last beta/2 entries ("tail") solve the linear constraints of M_beta (Sooter et al.), with k the true lag number::

        row 0:   sum_k tail_k        = 1 - sum_k phic_reduc_k
        row i:   sum_k k^i tail_k    =   - sum_k k^i phic_reduc_k     (i = 1 .. beta/2 - 1)

    Parameters
    ----------
    phic_reduc : (n - beta/2,) array
        First n - beta/2 kernel coefficients.
    beta : int
        Order of the critical manifold.

    Returns
    -------
    phic : (n,) array
    """
    phic_reduc = np.asarray(phic_reduc, dtype=float).ravel()
    half = half_order(beta)
    n = len(phic_reduc) + half  # model order

    lags_reduc = np.arange(1, len(phic_reduc) + 1, dtype=float)  # lags 1 .. n-beta/2
    lags_tail = np.arange(n - half + 1, n + 1, dtype=float)  # lags n-beta/2+1 .. n

    A = np.zeros((half, half))
    rhs = np.zeros(
        half
    )  # named `b` in the original, which clashes with the manifold order b
    for i in range(half):
        A[i, :] = lags_tail**i
        if i == 0:
            rhs[i] = 1 - np.sum(phic_reduc)
        else:
            rhs[i] = -(lags_reduc**i) @ phic_reduc

    return np.concatenate([phic_reduc, np.linalg.solve(A, rhs)])


def con(kernc, b):
    """Constraint of the optimization (fmincon convention: feasible if c <= 0).

    Returns
    -------
    c : float
        c = 1 - min |root| of the lifted kernel's AR polynomial: the roots must lie outside the unit disk.
    ceq : empty array
        Equality constraints: none.

    FLAG[1]: on M_b, z = 1 is ALWAYS a root (multiplicity b/2), so c >= 0 by construction: the constraint is permanently at its boundary, c ~ 0 up to root-finding round-off (~1e-16 for b=2, ~1e-8 for b=4, ~1e-5 for b=6). It is also non-smooth (minimum over roots). The original works because fmincon accepts c <= ConstraintTolerance = 0.01 (see CHANGED[2]).
    """
    kernc = beta_lift(kernc, b)
    c = 1 - np.min(np.abs(ar_roots(kernc)))
    ceq = np.array([])
    return c, ceq


def calc_dbv1(kern, b):
    """Closest kernel on M_b in L2 (Euclidean) distance; the starting point of the KL minimization.

    M_b is the affine hull of the kernels of the critical models (1 - L)^k x_t = eps_t, k = b/2 .. n (L = lag operator). Their polynomial P(z) = (1 - z)^k has a k-fold root at z = 1 (k >= b/2), and by the binomial theorem their kernels are::

        kern_t = C(k, t) (-1)^(t+1)   for t <= k,   0 otherwise

    The closest point is the least-squares projection of ``kern`` onto that affine hull, computed with a QR decomposition.

    Returns
    -------
    dbv1 : float
        L2 distance ||kern - kernc||.
    kernc : (n,) array
        Closest kernel on M_b.
    """
    m = half_order(b) - 1
    order = len(kern)

    # One column per spanning point (critical kernel of (1 - L)^k), k = m+1 .. order; t = lag.
    points = np.zeros((order, order - m))
    for k in range(m + 1, order + 1):
        for t in range(1, k + 1):
            points[t - 1, k - m - 1] = comb(k, t, exact=True) * (-1) ** (t + 1)

    # Affine hull: directions relative to the first point.
    directions = (points - points[:, [0]])[:, 1:]

    kern = np.asarray(kern, dtype=float).ravel()
    target = kern - points[:, 0]

    # Least squares  directions @ v = target  via QR (the original uses the full QR; equal for full column rank).
    Q, R = qr(directions, mode="economic")
    v = solve_triangular(R, Q.T @ target)

    kernc = points[:, 0] + directions @ v
    dbv1 = np.sqrt(np.sum((kern - kernc) ** 2))
    return dbv1, kernc

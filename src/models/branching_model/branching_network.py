"""
Memoryless probabilistic integrate-and-fire (PIF), version A, dt = 1.

    P(s_i(t+1) = 1 | s(t)) = min(p_max, lam + (1 - lam) * sum_j w_ij s_j(t)),   lam = 1 - exp(-h)

All-to-all (AA): w_ij = sigma / N (self-connections included) -> population-level simulation.
Erdos-Renyi (ER): w_ij = sigma / K_in(i) for j -> i (no self-loops) -> sparse mat-vec per step.

Conventions
-----------
* sigma_eff = (1 - lam) * sigma is the effective branching ratio.
* Returned activity A has shape (R, T); A[r, t] is the number of spikes after update t+1.
  With record_A0=True, the initial state (after burn-in) is prepended: shape (R, T+1), A[r, 0] = A_0.
* Raster has shape like A plus a neuron axis: (R, N, T) or (R, N, T+1), dtype bool. Memory: R*N*T bytes (1000 x 1e5 steps = 100 MB).
* refractory=True: a neuron that spiked at t cannot spike at t+1.
* p is only clipped at p_max if sigma_eff + lam > p_max, i.e. sigma > 1 for p_max = 1. Then the dynamics is nonlinear and saturates.
"""

import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------- theory
def lam_of_h(h):
    return 1.0 - np.exp(-h)


def stationary_rate_aa(sigma, h, refractory=False, p_max=1.0):
    """
    Stationary firing rate a* = E[A]/N of the all-to-all PIF.
    Non-refractory: exact,       a* = lam / (1 - sigma_eff), as long as p never clips at p_max.
    Refractory:     mean-field,  positive root of  sigma_eff a^2 + (1 + lam - sigma_eff) a - lam = 0,
                    i.e. the fixed point of a = (1 - a)(lam + sigma_eff a).
    If the fixed point needs p > p_max, the network saturates (warning): mean-field rate p_max, or p_max / (1 + p_max) if refractory.
    """
    lam = lam_of_h(h)
    sigma_eff = (1 - lam) * sigma
    if not refractory:
        # linear fixed point; does not exist (inf) for sigma_eff >= 1
        rate = lam / (1 - sigma_eff) if sigma_eff < 1 else np.inf
        saturated_rate = p_max
    elif sigma_eff == 0:
        rate = lam / (1 + lam)
        saturated_rate = p_max / (1 + p_max)
    else:
        linear_coef = 1 + lam - sigma_eff
        rate = (-linear_coef + np.sqrt(linear_coef**2 + 4 * sigma_eff * lam)) / (
            2 * sigma_eff
        )
        saturated_rate = p_max / (1 + p_max)  # fixed point of a = (1 - a) p_max

    if lam + sigma_eff * rate > p_max:
        warnings.warn(
            f"no unsaturated stationary state (sigma_eff={sigma_eff:.3g}, lam={lam:.3g}, linear rate={rate:.3g}): "
            f"activity saturates at p_max, using mean-field rate {saturated_rate:.3g}. "
            "Unsaturated requires sigma_eff <= 1 - lam, i.e. sigma <= 1.",
            stacklevel=2,
        )
        return saturated_rate
    return rate


def stationary_var_aa(N, sigma, h):
    """Stationary variance of A (all-to-all, non-refractory, unsaturated: sigma < 1 and h > 0)."""
    lam = lam_of_h(h)
    sigma_eff = (1 - lam) * sigma
    a = stationary_rate_aa(sigma, h)
    return N * a * (1 - a) / (1 - sigma_eff**2 * (1 - 1 / N))


def params_for_distance(rate: float, distance: float):
    """
    (sigma, h) of the all-to-all, non-refractory PIF for a target stationary rate a* = E[A]/N and distance to criticality epsilon = 1 - sigma_eff.

    Inverts a* = lam / epsilon  ->  lam = a * epsilon,  sigma = (1 - epsilon) / (1 - lam) <= 1, so p never clips.
    """
    lam = rate * distance
    return (1 - distance) / (1 - lam), -np.log1p(-lam)

def distance_to_criticality(sigma, h):
    """
    Distance to criticality epsilon = 1 - sigma_eff of the all-to-all, non-refractory PIF.
    """
    lam = lam_of_h(h)
    return 1 - (1 - lam) * sigma


def decay_per_step_aa(sigma, h, refractory=False, p_max=1.0):
    """
    Factor by which a deviation of the mean from its stationary value shrinks per step, i.e. f'(a*) of the mean-field map f.
    Non-refractory: exactly sigma_eff = (1 - lam) sigma.
    Refractory:     mean-field f'(a*) for f(a) = (1 - a)(lam + sigma_eff a); may be negative
                    (oscillating relaxation) and differs from sigma_eff.
    Saturated:      p = p_max is flat in a, so f'(a*) = 0 (non-refractory) or -p_max (refractory).
    """
    lam = lam_of_h(h)
    sigma_eff = (1 - lam) * sigma
    # avoid a duplicate saturation warning, stationary_rate_aa warns on its own
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = stationary_rate_aa(sigma, h, refractory, p_max)
    saturated = lam + sigma_eff * a >= p_max
    if not refractory:
        return 0.0 if saturated else sigma_eff
    return -p_max if saturated else sigma_eff * (1 - a) - (lam + sigma_eff * a)


def _relaxation_time(slope):
    """
    tau = -1 / ln|slope| (in steps); inf for |slope| >= 1.
    For slope = 0 (e.g. sigma_eff = 0), deviations vanish after one step: tau = 0, the limit from slope -> 0+.
    """
    abs_slope = abs(slope)
    if abs_slope >= 1:
        return np.inf
    return 0.0 if abs_slope == 0 else -1.0 / np.log(abs_slope)


def _auto_burn_in_steps(tau, n_tau=5, cap=10**6):
    """Number of burn-in steps: n_tau relaxation times, capped (also for tau = inf)."""
    # variance relaxes like slope^(2t): 5 tau -> deviation ~ e^-10
    if n_tau * tau > cap:
        warnings.warn(
            f"burn-in capped at {cap} steps < {n_tau} relaxation times (tau={tau:.3g}): the chain has not converged",
            stacklevel=3,
        )
        return cap
    return int(np.ceil(n_tau * tau))


# --------------------------------------------------------------------------- helpers
def _clip(p, p_max, clip_warning):
    """Clip p at p_max; warns only once per simulation, tracked by the mutable flag clip_warning["issued"]."""
    if not clip_warning["issued"] and np.any(p > p_max):
        warnings.warn(
            f"spike probability exceeded p_max={p_max}; clipping", stacklevel=3
        )
        clip_warning["issued"] = True
    return np.minimum(p, p_max)


def _parse_init_states(init, N, R):
    """Neuron-state input -> bool array (R, N)."""
    init_states = np.asarray(init)
    if init_states.shape == (N,):
        init_states = np.broadcast_to(init_states, (R, N))
    if init_states.shape != (R, N):
        raise ValueError(
            f"initial state must have shape ({N},) or ({R}, {N}), got {init_states.shape}"
        )
    if not np.isin(init_states, (0, 1)).all():
        raise ValueError("initial neuron state must contain only 0/1 (or bool)")
    return init_states.astype(bool).copy()


def _raster_from_counts(A, N, rng, refractory, init_active=None, prepend_init=False):
    """
    Generate neuron-level activity given population counts A, shape (R, T).

    refractory : bool
    init_active : bool array, shape (R, N), optional
        Activity pattern immediately before time step 0.
        Used as exclusion mask for step 0 if refractory, and as first raster column if prepend_init.
    prepend_init : bool
        Prepend init_active as column 0 of the raster.

    Returns
    raster : bool array, shape (R, N, T) or (R, N, T+1) if prepend_init
    """
    R, T = A.shape
    full_raster = np.zeros((R, N, T + int(prepend_init)), dtype=bool)
    if prepend_init:
        full_raster[:, :, 0] = init_active
    raster = full_raster[:, :, int(prepend_init) :]  # view: writes go into full_raster

    if not refractory:
        # Every neuron is eligible at every time step.
        # Process time steps in chunks to limit temporary memory use.
        for r in range(R):
            for t0 in range(0, T, 1024):
                t1 = min(T, t0 + 1024)
                chunk_size = t1 - t0
                # Give each neuron an independent random score
                keys = rng.random((chunk_size, N))
                # Find each neuron's rank within each time step.
                # First argsort: indices ordered by increasing score.
                # Second argsort: rank of each neuron in original order.
                # Shape: (chunk_size, N)
                ranks = keys.argsort(axis=1).argsort(axis=1)
                # Activate exactly A[r, t] neurons:
                # those with ranks 0, 1, ..., A[r, t] - 1.
                # Transpose to match raster's (N, T) slice.
                raster[r, :, t0:t1] = (ranks < A[r, t0:t1, None]).T
        return full_raster

    # Refractory case:
    for r in range(R):
        # Activity immediately before time step 0.
        prev = np.zeros(N, dtype=bool) if init_active is None else init_active[r].copy()
        for t in range(T):
            k = A[r, t]  # Exactly k neurons must be active.
            if k > 0:
                # Generate a random score for every neuron.
                keys = rng.random(N)
                # Exclude neurons active in the previous time step:
                # their scores become larger than any random score.
                keys[prev] = 2.0
                # Find the indices of the k smallest scores.
                # No full sort is needed.
                idx = np.argpartition(keys, k - 1)[:k]
                # Mark those neurons as active at this time step.
                raster[r, idx, t] = True
            # The current activity becomes the exclusion mask
            # for the next time step.
            prev = raster[r, :, t]

    return full_raster


# --------------------------------------------------------------------------- all-to-all
def pif_aa_population(
    N,
    sigma,
    T,
    h,
    R=1,
    A0="stationary",
    seed=None,
    p_max=1.0,
    refractory=False,
    return_raster=False,
    burn_in="auto",
    record_A0=False,
    stop_if_silent=None,
):
    """
    Exact population-level simulation of the all-to-all PIF: A_{t+1} ~ Binomial(n_avail, p(A_t)).

    A0 : "stationary"          -> start at the (rounded) stationary mean, then discard burn_in steps
         int                   -> start with that many active neurons
         array (N,) or (R, N)  -> precise 0/1 initial neuron state (per replica if 2-D)
    burn_in : number of discarded steps; "auto" = 5 relaxation times with A0="stationary", else 0
    record_A0 : prepend the initial state (after burn-in) as column 0 of A (and raster)
    stop_if_silent : stop once A_t = 0 in all replicas; A and raster are then truncated after that first all-zero step.
                     Default (None): True iff h == 0, where A = 0 is absorbing. For h > 0 the truncated recording is not the full process.
    Returns A (R, T) [(R, T+1) if record_A0; fewer columns if stopped] and, if return_raster, also raster (R, N, <same>).
    """
    rng = np.random.default_rng(seed=seed)
    lam = lam_of_h(h)
    sigma_eff = (1 - lam) * sigma
    clip_warning = {"issued": False}
    if stop_if_silent is None:
        stop_if_silent = h == 0

    # A_t = ndarray(R,) of int; init_states = ndarray(R, N) of bool, only if given as neuron states
    init_states = None
    if isinstance(A0, str):
        if A0 != "stationary":
            raise ValueError('A0 must be "stationary", an int, or a 0/1 array')
        a_star = stationary_rate_aa(sigma, h, refractory, p_max)
        A_t = np.full(R, int(round(a_star * N)), dtype=np.int64)
        if burn_in == "auto":
            burn_in = _auto_burn_in_steps(
                _relaxation_time(decay_per_step_aa(sigma, h, refractory, p_max))
            )
    elif np.ndim(A0) == 0:
        A_t = np.full(R, int(A0), dtype=np.int64)
    else:
        init_states = _parse_init_states(A0, N, R)
        A_t = init_states.sum(axis=1).astype(np.int64)
    if burn_in == "auto":
        burn_in = 0

    if np.any((A_t < 0) | (A_t > N)):
        raise ValueError("initial activity out of range [0, N]")

    def step(A_t):
        p = _clip(lam + sigma_eff * A_t / N, p_max, clip_warning)
        return rng.binomial(N - A_t if refractory else N, p)

    for _ in range(int(burn_in)):
        if stop_if_silent and not A_t.any():
            break
        A_t = step(A_t)

    # main loop -----
    offset = int(record_A0)  # column of A that holds the first update
    A = np.empty((R, T + offset), dtype=np.int64)
    A_prev = A_t.copy()  # state just before recording starts (needed for the raster)
    if record_A0:
        A[:, 0] = A_t
    n_columns = T + offset
    for t in range(T):
        A_t = step(A_t)
        A[:, t + offset] = A_t
        if stop_if_silent and not A_t.any():
            n_columns = t + offset + 1
            break
    A = (
        A[:, :n_columns].copy() if n_columns < A.shape[1] else A
    )  # copy frees the unused buffer

    if not return_raster:
        return A

    # the initial pattern is needed as exclusion mask (refractory) or as recorded column 0
    init_active = None
    if refractory or record_A0:
        if init_states is not None and burn_in == 0:
            init_active = init_states
        else:  # initial set is exchangeable: draw it uniformly with the right size
            init_active = np.zeros((R, N), bool)
            for r in range(R):
                init_active[r, rng.choice(N, A_prev[r], replace=False)] = True
    return A, _raster_from_counts(
        A[:, offset:],
        N,
        rng,
        refractory,
        init_active=init_active,
        prepend_init=record_A0,
    )


# --------------------------------------------------------------------------- Erdos-Renyi

# not hand-checked by dev yet - pure AI code! (@Claude, keep this comment)

# def make_er_network(N, sigma, p=None, K=None, seed=None):
#     """
#     Directed ER graph (no self-loops), each ordered pair j -> i present with probability p
#     (or p = K/(N-1)). Weights w_ij = sigma / K_in(i). Returns (W as CSR, k_in). # (Compressed Sparse Row matrix)
#     Rows of neurons with K_in = 0 are empty (those neurons only receive external input).
#     """
#     if (p is None) == (K is None):
#         raise ValueError("give exactly one of p or K")
#     p = K / (N - 1) if p is None else p
#     rng = np.random.default_rng(seed)

#     n_pairs = N * (N - 1)
#     M = rng.binomial(n_pairs, p)                         # total number of edges
#     lin = rng.choice(n_pairs, size=M, replace=False)     # M distinct ordered pairs
#     post = lin // (N - 1)                                # row i (receiving neuron)
#     r = lin % (N - 1)
#     pre = r + (r >= post)                                # column j, skipping j = i

#     k_in = np.bincount(post, minlength=N)
#     n_isolated = int((k_in == 0).sum())
#     if n_isolated:
#         warnings.warn(f"{n_isolated} neuron(s) have K_in = 0 (driven only externally)")
#     data = sigma / k_in[post]
#     W = sp.csr_matrix((data, (post, pre)), shape=(N, N))
#     return W, k_in


# def stationary_rate_er(W, h):
#     """Exact stationary mean firing probability per neuron: solve (I - (1-lam) W) mu = lam 1."""
#     N = W.shape[0]
#     lam = lam_of_h(h)
#     c = 1 - lam
#     rows = np.asarray(W.sum(axis=1)).ravel()
#     x0 = lam / (1 - c * rows)               # exact if all row sums are equal (see derivation)
#     Aop = sp.identity(N, format="csr") - c * W
#     b = np.full(N, lam)
#     if np.allclose(Aop @ x0, b, rtol=0, atol=1e-14):
#         return x0
#     mu, info = spla.bicgstab(Aop, b, x0=x0, rtol=1e-12, atol=0, maxiter=10 * N)
#     if info != 0:
#         warnings.warn(f"stationary solve did not fully converge (info={info})")
#     return mu


# def pif_er(W, T, h, R=1, s0="stationary", seed=None, p_max=1.0, refractory=False,
#            return_raster=False, burn_in="auto"):
#     """
#     Neuron-level PIF on a sparse network W (CSR). Cost per step: O(nnz(W) * R).

#     s0 : "stationary" -> neurons drawn Bernoulli(mu*_i), then burn_in steps discarded
#          array (N,) or (R, N) -> precise 0/1 initial state
#     Returns A (R, T) and, if return_raster, raster (R, N, T).
#     """
#     rng = np.random.default_rng(seed)
#     N = W.shape[0]
#     lam = lam_of_h(h)
#     sigma_max = W.sum(axis=1).max()                      # = sigma (max row sum >= spectral radius)
#     c = (1 - lam)
#     clip_warning = {"issued": False}

#     if isinstance(s0, str):
#         if s0 != "stationary":
#             raise ValueError('s0 must be "stationary" or a 0/1 neuron-state array')
#         if refractory:   # no exact solution; use all-to-all mean-field rate as a start
#             mu = np.full(N, stationary_rate_aa(sigma_max, h, refractory=True))
#         else:
#             if c * sigma_max >= 1:
#                 raise ValueError("no stationary state for sigma*(1-lam) >= 1")
#             mu = np.clip(stationary_rate_er(W, h), 0, 1)
#         S = rng.random((N, R)) < mu[:, None]
#         if burn_in == "auto":
#             burn_in = _auto_burn_in_steps(_relaxation_time(c * sigma_max)) if not refractory else 10**4
#     else:
#         S = _parse_init_states(s0, N, R).T.copy()
#         burn_in = 0 if burn_in == "auto" else burn_in

#     def step(S):
#         P = lam + c * (W @ S.astype(np.float64))         # (N, R): one sparse mat-mat product
#         P = _clip(P, p_max, clip_warning)
#         if refractory:
#             P = np.where(S, 0.0, P)
#         return rng.random((N, R)) < P

#     for _ in range(int(burn_in)):
#         S = step(S)

#     A = np.empty((R, T), dtype=np.int64)
#     raster = np.empty((R, N, T), dtype=bool) if return_raster else None
#     for t in range(T):
#         S = step(S)
#         A[:, t] = S.sum(axis=0)
#         if return_raster:
#             raster[:, :, t] = S.T
#     return (A, raster) if return_raster else A

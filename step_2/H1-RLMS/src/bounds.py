"""Bounds on pairwise Brier differences for a fixed cohort with hidden binary labels.

Notation (plan, section 2.1).  N rows, K frozen models, p[i,k] forecast, y[i] label,
w[i] >= 0 cohort weights normalised over ALL N rows.
    R_k      = sum_i w_i (p_ik - y_i)^2            (full-cohort Brier, unknown if labels hidden)
    D_kj     = R_k - R_j                           (pairwise difference)
    regret_k = R_k - min_j R_j = max_j D_kj         (j = k allowed, D_kk = 0)

Everything here is a deterministic function of the forecasts, the OBSERVED labels and the
observation mask.  Labels of hidden rows are never read (see `validate_cohort`).

Derivation of C1 (checked independently by exhaustive enumeration in tests/):
    for one row,   (p_a - y)^2 - (p_b - y)^2 = (p_a^2 - p_b^2) - 2 y (p_a - p_b) = a_i + b_i y.
    with y in {0,1} the row value is  a_i  (y=0)  or  a_i + b_i  (y=1).
    Hidden rows are unconstrained and enter D additively, so each can be chosen independently:
        min over fills = sum_all a_i + sum_obs b_i y_i + sum_hidden min(0, b_i)
        max over fills = sum_all a_i + sum_obs b_i y_i + sum_hidden max(0, b_i).
    Both extremes are attained by some admissible fill => the bounds are exact (tight).
    Width = sum_hidden |b_i| = 2 sum_hidden w_i |p_ia - p_ib|.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

TOL = 1e-10          # tolerance used when comparing bounds to ground truth
CERT_TOL = 1e-12     # numerical slack in the rule G <= epsilon (as in plan appendix A)


class H1InputError(ValueError):
    """Raised for invalid cohort data (shapes, probabilities, labels, weights)."""


# --------------------------------------------------------------------------- validation
def validate_cohort(predictions, labels, observed, weights=None):
    """Validate inputs; return (p, y, seen, w).

    * y is returned with zeros on hidden rows: hidden labels are discarded, never read.
    * w is normalised to sum to 1 over all N rows (hidden rows included).
    """
    p = np.asarray(predictions, dtype=float)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 1:
        raise H1InputError("predictions must be a non-empty N-by-K array")
    n, k = p.shape
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise H1InputError("probabilities must be finite and lie in [0,1]")
    seen = np.asarray(observed)
    if seen.shape != (n,):
        raise H1InputError("observed must have shape (N,)")
    if seen.dtype != bool:
        if not np.isin(seen, [0, 1]).all():
            raise H1InputError("observed must be boolean / 0-1")
        seen = seen.astype(bool)
    y_raw = np.asarray(labels, dtype=float)
    if y_raw.shape != (n,):
        raise H1InputError("labels must have shape (N,)")
    if not np.isin(y_raw[seen], [0.0, 1.0]).all():
        raise H1InputError("observed labels must be binary (0/1, no NaN)")
    y = np.zeros(n)
    y[seen] = y_raw[seen]
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise H1InputError("weights must be finite, non-negative, shape (N,), positive sum")
        w = w / w.sum()
    return p, y, seen, w


def _finish(lower, upper):
    return lower, upper, upper.max(axis=1)


# --------------------------------------------------------------------------- C0
def c0_bounds(predictions, labels, observed, weights=None):
    """C0: separate per-model loss intervals (plan 2.2).  Valid but loose.

    R_low[k]  = observed loss + sum_hidden w min(p^2,(1-p)^2)
    R_high[k] = observed loss + sum_hidden w max(p^2,(1-p)^2)
    U[k,j] = R_high[k] - R_low[j],  L[k,j] = R_low[k] - R_high[j]  (k != j), diagonal 0.

    What it guarantees: for every admissible fill, L[k,j] <= R_k - R_j <= U[k,j]  (finite, exact
    in the sense of a deterministic guarantee).  What it does NOT do: it is not tight, because the
    worst fill for model k and the worst fill for model j are generally different fills.
    """
    p, y, seen, w = validate_cohort(predictions, labels, observed, weights)
    hid = ~seen
    obs_loss = (w[seen, None] * (p[seen] - y[seen, None]) ** 2).sum(axis=0)
    ph = p[hid]
    wh = w[hid, None]
    r_low = obs_loss + (wh * np.minimum(ph ** 2, (1 - ph) ** 2)).sum(axis=0)
    r_high = obs_loss + (wh * np.maximum(ph ** 2, (1 - ph) ** 2)).sum(axis=0)
    upper = r_high[:, None] - r_low[None, :]
    lower = r_low[:, None] - r_high[None, :]
    np.fill_diagonal(upper, 0.0)
    np.fill_diagonal(lower, 0.0)
    return _finish(lower, upper)


# --------------------------------------------------------------------------- C1
def c1_bounds(predictions, labels, observed, weights=None):
    """C1: exact pairwise bounds (plan 2.3).  Returns (lower, upper, worst_regret).

    lower[k,j] = min over all fills of R_k - R_j,   upper[k,j] = max over all fills.
    worst_regret[k] = G_k = max_j upper[k,j]  (includes j=k, upper[k,k] = 0)
      = the largest regret model k can have over all admissible fills of the hidden labels.
    """
    p, y, seen, w = validate_cohort(predictions, labels, observed, weights)
    n, k = p.shape
    hid = ~seen
    lower = np.zeros((k, k))
    upper = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            const = w * (p[:, a] ** 2 - p[:, b] ** 2)
            coeff = -2.0 * w * (p[:, a] - p[:, b])
            center = const.sum() + (coeff[seen] * y[seen]).sum()
            ch = coeff[hid]
            lo = center + np.minimum(ch, 0.0).sum()
            hi = center + np.maximum(ch, 0.0).sum()
            lower[a, b], upper[a, b] = lo, hi
            lower[b, a], upper[b, a] = -hi, -lo
    return _finish(lower, upper)


def reveal_update(lower, upper, p_row, y_revealed, weight):
    """O(K^2) update of C1 bounds after truly revealing ONE hidden label (plan 2.3).

    p_row: forecasts of the revealed row (K,); weight: its weight normalised over the full cohort.
        U_new = U_old - max(0,b) + b*y,   L_new = L_old - min(0,b) + b*y,   b_kj = -2 w (p_k - p_j)
    Returns new (lower, upper, worst_regret).
    """
    if y_revealed not in (0, 1):
        raise H1InputError("revealed label must be 0 or 1")
    p_row = np.asarray(p_row, float)
    b = -2.0 * weight * (p_row[:, None] - p_row[None, :])
    up = upper - np.maximum(b, 0.0) + b * y_revealed
    lo = lower - np.minimum(b, 0.0) + b * y_revealed
    np.fill_diagonal(up, 0.0)
    np.fill_diagonal(lo, 0.0)
    return lo, up, up.max(axis=1)


# --------------------------------------------------------------------------- C2
def _check_counts(A, B):
    for v in (A, B):
        if not float(v).is_integer():
            raise H1InputError("count bounds A_g,B_g must be integers")
    return int(A), int(B)


def c2_bounds(predictions, labels, observed, group_ids, count_bounds, weights=None):
    """C2: C1 plus constraints  A_g <= (#ones among CURRENTLY hidden rows of group g) <= B_g.

    group_ids: (N,) ints; hidden rows of groups without an entry in `count_bounds` (and rows with
    group id -1) stay unconstrained.  Disjoint groups only (plan 2.4, first MVP).
    Returns (lower, upper, worst_regret, status) with status in {"ok", "assumption_infeasible"}.
    When infeasible the matrices are NaN (never a negative-width interval).

    For a pair (k,j) and group g with sorted hidden coefficients b: choosing exactly s ones the
    extremes are the sum of the s smallest / largest b; the extreme over s in [A,B] is the min / max
    of the prefix sums on those indices.  Group contributions are independent, so they add.
    """
    p, y, seen, w = validate_cohort(predictions, labels, observed, weights)
    n, k = p.shape
    gid = np.asarray(group_ids)
    if gid.shape != (n,):
        raise H1InputError("group_ids must have shape (N,)")
    hid = ~seen
    # feasibility of every constraint against the number of currently hidden rows
    ranges = {}
    for g, (A, B) in count_bounds.items():
        A, B = _check_counts(A, B)
        m_g = int(np.sum(hid & (gid == g)))
        A_c, B_c = max(A, 0), min(B, m_g)
        if A_c > B_c:
            nan = np.full((k, k), np.nan)
            return nan, nan.copy(), np.full(k, np.nan), "assumption_infeasible"
        ranges[g] = (A_c, B_c)
    constrained = np.zeros(n, dtype=bool)
    for g in ranges:
        constrained |= hid & (gid == g)
    free = hid & ~constrained
    lower = np.zeros((k, k))
    upper = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            const = w * (p[:, a] ** 2 - p[:, b] ** 2)
            coeff = -2.0 * w * (p[:, a] - p[:, b])
            center = const.sum() + (coeff[seen] * y[seen]).sum()
            lo = center + np.minimum(coeff[free], 0.0).sum()
            hi = center + np.maximum(coeff[free], 0.0).sum()
            for g, (A_c, B_c) in ranges.items():
                cg = coeff[hid & (gid == g)]
                asc = np.concatenate(([0.0], np.cumsum(np.sort(cg))))
                desc = np.concatenate(([0.0], np.cumsum(np.sort(cg)[::-1])))
                lo += asc[A_c:B_c + 1].min()
                hi += desc[A_c:B_c + 1].max()
            lower[a, b], upper[a, b] = lo, hi
            lower[b, a], upper[b, a] = -hi, -lo
    return lower, upper, upper.max(axis=1), "ok"


def update_count_bounds_after_reveal(count_bounds, group, y_revealed, m_remaining):
    """Plan 2.4: after revealing s=y positives from group g the residual constraint is
    [max(0, A-s), min(m_remaining, B-s)].  Original A,B are never re-estimated.
    A negative upper end (more ones revealed than allowed) is kept so that c2_bounds reports
    `assumption_infeasible`."""
    new = dict(count_bounds)
    if group in new:
        A, B = new[group]
        new[group] = (max(0, int(A) - int(y_revealed)), min(int(m_remaining), int(B) - int(y_revealed)))
    return new


def audit_count_interval(n_positive_in_sample, m_sample, M_group, n_groups, alpha=0.05):
    """Mode (b) of C2 (plan 2.4): count interval from a paid uniform audit WITHOUT replacement.

    r_g = sqrt(log(2*G/alpha) / (2 m)),  A = floor(M*max(0, mean-r)),  B = ceil(M*min(1, mean+r)).
    If the whole group was audited, the exact count is returned.  Statistical, not deterministic:
    the interval contains the true count of the group (all G groups jointly) with probability
    >= 1 - alpha by Hoeffding's inequality for sampling without replacement.
    """
    if m_sample >= M_group:
        return int(n_positive_in_sample), int(n_positive_in_sample)
    if m_sample < 1:
        return 0, int(M_group)
    mean = n_positive_in_sample / m_sample
    r = math.sqrt(math.log(2 * n_groups / alpha) / (2 * m_sample))
    A = int(math.floor(M_group * max(0.0, mean - r) + 1e-12))
    B = int(math.ceil(M_group * min(1.0, mean + r) - 1e-12))
    return A, B


def scenario_count_bounds(n_visible_positive, n_visible, M_group, gamma, widen=0.0):
    """Plan 4.5 (R3) scenario constraint from visible labels with odds-ratio parameter Gamma.

    pi_vis = (pos+0.5)/(n+1);  pi_low/high = sigmoid(logit(pi_vis) -/+ log Gamma), widened by +-`widen`
    and clipped to [0,1];  A = floor(M*pi_low), B = ceil(M*pi_high).  Gamma = inf or n_visible = 0 -> [0, M].
    This is an ASSUMPTION about the hidden rows (guarantee_type 'scenario_shift'), not an estimate.
    """
    if n_visible == 0 or math.isinf(gamma):
        return 0, int(M_group)
    pi = (n_visible_positive + 0.5) / (n_visible + 1.0)
    lg = math.log(pi / (1 - pi))
    lo = 1 / (1 + math.exp(-(lg - math.log(gamma))))
    hi = 1 / (1 + math.exp(-(lg + math.log(gamma))))
    lo, hi = max(0.0, lo - widen), min(1.0, hi + widen)
    return int(math.floor(M_group * lo + 1e-12)), int(math.ceil(M_group * hi - 1e-12))


# --------------------------------------------------------------------------- C3
def intersect_intervals(lo_s, up_s, lo_c1, up_c1):
    """Intersect statistical and C1 intervals.  If any pair has an empty intersection the
    statistical event has demonstrably failed: return the C1 intervals unchanged (plan 2.5)."""
    lo = np.maximum(lo_s, lo_c1)
    up = np.minimum(up_s, up_c1)
    if np.any(lo > up + TOL):
        return lo_c1, up_c1, True
    return lo, np.maximum(up, lo), False


def c3_bounds(predictions, labels, observed_initial, audit_idx, alpha=0.05, n_budgets=1):
    """C3: Hoeffding bounds for a random audit, intersected with C1 recomputed after the audit.

    Preconditions (plan 2.5): equal weights 1/N; `audit_idx` is a uniform random sample WITHOUT
    replacement (size m) of the initially hidden rows, drawn independently of all labels and of the
    forecasts; budgets are enumerated in advance (`n_budgets` = B, used in the union bound).
        d_i = (p_ik - y_i)^2 - (p_ij - y_i)^2  in [-1, 1]
        D_hat = sum_obs d_i / N + (M/N) * mean_{audit} d_i
        radius = (M/N) * sqrt(2 log(2 J B / alpha) / m),   J = K(K-1)
    Only labels of initially observed rows and of `audit_idx` are read.
    Returns a dict (lower/upper = final intersection, 'event_failed' if intersection is empty).
    """
    p = np.asarray(predictions, float)
    n, k = p.shape
    seen0 = np.asarray(observed_initial, bool)
    audit_idx = np.asarray(audit_idx, dtype=int)
    if np.any(seen0[audit_idx]):
        raise H1InputError("audit rows must be initially hidden")
    if len(np.unique(audit_idx)) != len(audit_idx):
        raise H1InputError("audit sample must be without replacement")
    labels = np.asarray(labels, float)
    seen1 = seen0.copy()
    seen1[audit_idx] = True                            # labels revealed by the audit
    lo_c1, up_c1, _ = c1_bounds(p, labels, seen1)      # C1 after the same reveals (uniform weights)
    M = int((~seen0).sum())
    m = len(audit_idx)
    out = {"M": M, "m": m, "N": n, "lower_c1": lo_c1, "upper_c1": up_c1,
           "lower_stat": None, "upper_stat": None, "radius": None, "event_failed": False}
    if k < 2 or m == 0 or M == 0:
        out.update(lower=lo_c1, upper=up_c1)
        return out
    if m == M:                                         # whole hidden part revealed: exact
        out.update(lower=lo_c1, upper=up_c1)
        return out
    J = k * (k - 1)
    radius = (M / n) * math.sqrt(2.0 * math.log(2.0 * J * n_budgets / alpha) / m)
    y_obs = np.where(seen0, labels, 0.0)
    y_aud = labels[audit_idx]
    dhat = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            d_obs = (p[seen0, a] - y_obs[seen0]) ** 2 - (p[seen0, b] - y_obs[seen0]) ** 2
            d_aud = (p[audit_idx, a] - y_aud) ** 2 - (p[audit_idx, b] - y_aud) ** 2
            dhat[a, b] = d_obs.sum() / n + (M / n) * d_aud.mean()
    lo_s, up_s = dhat - radius, dhat + radius
    np.fill_diagonal(lo_s, 0.0)
    np.fill_diagonal(up_s, 0.0)
    lo, up, failed = intersect_intervals(lo_s, up_s, lo_c1, up_c1)
    out.update(lower_stat=lo_s, upper_stat=up_s, radius=radius, event_failed=failed)
    out.update(lower=lo, upper=up)
    return out

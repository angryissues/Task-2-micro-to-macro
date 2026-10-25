"""Point-estimate baselines (plan 7 / 8.4).  They return RECOMMENDATIONS only
(guarantee_type 'recommendation_only', certified=False): no certificate is implied."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .bounds import validate_cohort


def _loss(p, y):
    return (p - y[:, None]) ** 2


def _pack(name, est, extra=None):
    ok = np.all(np.isfinite(est))
    rec = int(np.argmin(est)) if ok else None
    out = {"method": name, "estimates": est, "recommended_model": rec, "selected_model": rec,
           "certified": False, "guarantee_type": "recommendation_only"}
    out.update(extra or {})
    return out


def complete_case(predictions, labels, observed, weights=None):
    """L_cc[k] = sum w O l / sum w O.  Recommendation for the OBSERVED part only."""
    p, y, seen, w = validate_cohort(predictions, labels, observed, weights)
    den = w[seen].sum()
    if den <= 0:
        return _pack("complete_case", np.full(p.shape[1], np.nan))
    est = (w[seen, None] * _loss(p, y)[seen]).sum(axis=0) / den
    return _pack("complete_case", est)


def model_assisted(predictions, labels, observed, q, weights=None, n_fills=30, rng=None):
    """Expected-loss plug-in: hidden loss -> (1-q)p^2 + q(p-1)^2, known rows -> actual loss.
    Also averages `n_fills` independent Bernoulli(q) fills to check convergence to that value.
    The spread of fills is NOT a distribution-free guarantee."""
    p, y, seen, w = validate_cohort(predictions, labels, observed, weights)
    q = np.asarray(q, float)
    if q.shape != (p.shape[0],) or np.any((q < 0) | (q > 1)) or not np.isfinite(q).all():
        raise ValueError("q must be probabilities of shape (N,)")
    known = (w[seen, None] * _loss(p, y)[seen]).sum(axis=0)
    hid = ~seen
    exp_loss = ((1 - q[hid, None]) * p[hid] ** 2 + q[hid, None] * (p[hid] - 1) ** 2)
    est = known + (w[hid, None] * exp_loss).sum(axis=0)
    rng = np.random.default_rng(0) if rng is None else rng
    fills = np.empty((n_fills, p.shape[1]))
    for f in range(n_fills):
        yf = (rng.random(hid.sum()) < q[hid]).astype(float)
        fills[f] = known + (w[hid, None] * (p[hid] - yf[:, None]) ** 2).sum(axis=0)
    mean_fill = fills.mean(axis=0)
    return _pack("model_assisted", est, {
        "fill_mean_estimates": mean_fill, "fill_sd": fills.std(axis=0, ddof=1) if n_fills > 1 else None,
        "convergence_gap": float(np.abs(mean_fill - est).max()),
        "fill_agreement_with_expected": float(np.mean(fills.argmin(axis=1) == np.argmin(est)))})


def fit_aux_q(X_past, y_past):
    """Auxiliary q(x)=P(y=1|X) trained on PAST data only (never on the evaluated cohort)."""
    return LogisticRegression(max_iter=1000).fit(X_past, y_past)


def estimate_observation_prob(X, observed):
    """Logistic model of O on the available features (labels are not used)."""
    O = np.asarray(observed, int)
    if O.min() == O.max():
        return np.full(len(O), float(O[0]) if O[0] else 1.0)
    return LogisticRegression(max_iter=1000).fit(X, O).predict_proba(X)[:, 1]


def ipw(predictions, labels, observed, pi, weights=None, clip=None, name="ipw"):
    """L_ipw[k] = sum_i w_i O_i l_ik / pi_i  (Horvitz-Thompson, plan 7.3, not self-normalised).
    pi = probability of INITIAL observation; clip lower-bounds it (clipped_IPW uses 0.05).
    Not valid for adaptively requested labels; does not fix MNAR unless pi is the true one."""
    p, y, seen, w = validate_cohort(predictions, labels, observed, weights)
    pi = np.asarray(pi, float)
    if clip is not None:
        pi = np.maximum(pi, clip)
    if np.any(pi[seen] <= 0):
        raise ValueError("pi must be positive on observed rows")
    est = (w[seen, None] * _loss(p, y)[seen] / pi[seen, None]).sum(axis=0)
    return _pack(name, est)

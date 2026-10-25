"""Oracle-side metrics. Only evaluation code may use full labels (plan 8.1, evaluation/)."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

from .bounds import TOL


def full_brier(predictions, y, weights=None):
    p = np.asarray(predictions, float)
    n = p.shape[0]
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, float) / np.sum(weights)
    return (w[:, None] * (p - np.asarray(y, float)[:, None]) ** 2).sum(axis=0)


def true_regret_vector(R):
    R = np.asarray(R, float)
    return R - R.min()


def wrong_certificate(certified, true_regret, epsilon):
    return bool(certified and true_regret > epsilon + TOL)


def coverage_all_pairs(lower, upper, R):
    """All true pairwise differences R_k - R_j lie in [lower, upper] (tolerance 1e-10)."""
    D = R[:, None] - R[None, :]
    return bool(np.all(D >= lower - TOL) and np.all(D <= upper + TOL))


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def rate(k, n):
    """Rate with NA (nan) when the denominator is zero -- never 0."""
    return float(k) / n if n > 0 else float("nan")

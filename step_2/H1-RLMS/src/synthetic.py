"""Synthetic oracle generator: X=(age_z, education_z, baseline_employment), y~Bernoulli(sigmoid(beta X)),
candidate models = fixed coefficient shifts, missingness MCAR / MAR / MNAR with known pi (plan 8.3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BETA = np.array([-0.8, -0.5, 0.35, 1.6])          # intercept, age_z, education_z, employed
SPREAD = {"close": 0.05, "moderate": 0.25, "distinct": 0.8, "extreme": 3.0}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


@dataclass
class Cohort:
    X: np.ndarray            # (N,3) features (no intercept)
    y: np.ndarray            # (N,) true labels (oracle)
    p: np.ndarray            # (N,K) frozen candidate forecasts
    true_prob: np.ndarray    # (N,) generating probability


def _shift_directions(k):
    rs = np.random.RandomState(12345)
    u = rs.normal(size=(max(k, 8), 4))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u[:k]


def make_cohort(seed, n, k, spread="moderate", beta=BETA):
    """Candidate 0 has the generating coefficients; candidate j>0 is shifted by spread*u_j."""
    s = SPREAD[spread] if isinstance(spread, str) else float(spread)
    rng = np.random.default_rng(np.random.SeedSequence([1, int(seed)]))
    age = rng.normal(size=n)
    edu = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
    edu_z = (edu - 1.0) / 0.7
    emp = (rng.random(n) < 0.6).astype(float)
    X = np.column_stack([age, edu_z, emp])
    Z = np.column_stack([np.ones(n), X])
    prob = sigmoid(Z @ beta)
    y = (rng.random(n) < prob).astype(float)
    dirs = _shift_directions(k)
    betas = np.array([beta + (0.0 if j == 0 else 1.0) * s * dirs[j] for j in range(k)])
    p = sigmoid(Z @ betas.T)
    return Cohort(X=X, y=y, p=p, true_prob=prob)


def _calibrate(logit_core, target_mean):
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sigmoid(logit_core + mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def observation_probabilities(cohort, mechanism, missing_fraction, strength=1.0):
    """True pi_i = P(label initially observed).  Mean pi equals 1 - missing_fraction on the cohort."""
    n = len(cohort.y)
    if missing_fraction <= 0:
        return np.ones(n)
    if mechanism == "MCAR":
        return np.full(n, 1.0 - missing_fraction)
    if mechanism == "MAR":
        core = strength * (0.8 * cohort.X[:, 0] - 0.6 * cohort.X[:, 1] + 0.7 * cohort.X[:, 2])
    elif mechanism == "MNAR":
        core = strength * (-1.5 * cohort.y + 0.5 * cohort.X[:, 0])
    else:
        raise ValueError(f"unknown mechanism {mechanism}")
    return sigmoid(core + _calibrate(core, 1.0 - missing_fraction))


def draw_mask(pi, seed):
    rng = np.random.default_rng(np.random.SeedSequence([2, int(seed)]))
    return rng.random(len(pi)) < pi

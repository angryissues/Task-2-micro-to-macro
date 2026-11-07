"""H3 nonresponse simulator and policies.

The tested C1 implementation is reused from H2. This module adds only the
nonresponse layer: attempted rows can fail, cost is still charged, and failed
rows remain unknown to the public C1 state but cannot be attempted again.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

H2_ROOT = Path(__file__).resolve().parents[2] / "H2-RLMS"
_H2_SPEC = importlib.util.spec_from_file_location(
    "h2_active_selection", H2_ROOT / "src" / "active_selection.py"
)
_H2 = importlib.util.module_from_spec(_H2_SPEC)
assert _H2_SPEC.loader is not None
sys.modules["h2_active_selection"] = _H2
_H2_SPEC.loader.exec_module(_H2)

from h2_active_selection import (  # type: ignore[attr-defined]  # noqa: E402
    H2InputError, LabelOracle, SelectionState, q0, q2, random_policy,
)


def _sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_intercept(linear_part, target, tol=1e-10):
    """Find an intercept giving the requested expected missingness."""
    z = np.asarray(linear_part, dtype=float)
    target = float(target)
    if not 0 < target < 1:
        raise ValueError("target must lie strictly between 0 and 1")
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _sigmoid(mid + z).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def initial_missing_probabilities(age, education_low, y, mechanism="mcar",
                                   target=0.2, kappa=0.0):
    """Return oracle initial-missing probabilities for controlled simulation."""
    age = np.asarray(age, dtype=float)
    low = np.asarray(education_low, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (age.shape == low.shape == y.shape):
        raise ValueError("age, education_low and y must have equal shapes")
    if mechanism == "mcar":
        return np.full(len(y), float(target))
    age_z = (age - age.mean()) / max(age.std(), 1e-12)
    linear = 0.8 * age_z + 0.7 * low
    if mechanism == "mnar":
        linear = linear + float(kappa) * (1.0 - 2.0 * y)
    elif mechanism != "mar":
        raise ValueError("mechanism must be mcar, mar or mnar")
    a = calibrate_intercept(linear, target)
    return _sigmoid(a + linear)


def draw_initial_mask(probabilities, rng):
    p = np.asarray(probabilities, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    return rng.random(len(p)) < p


def draw_attempt_success(probabilities, rng):
    p = np.asarray(probabilities, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("answer probabilities must lie in [0,1]")
    return rng.random(len(p)) < p


def _expected_gain(state, row_id, answer_probability):
    current = state.value
    after0 = state.projected_value(row_id, 0)
    after1 = state.projected_value(row_id, 1)
    p = float(answer_probability)
    return max(0.0, current - ((1.0 - p) * current + p * ((after0 + after1) / 2.0)))


def r0_policy(state, rng=None):
    """Q2 ordering that ignores the chance of a successful answer."""
    return q2(state)


def r1_policy(state, answer_probabilities):
    """Expected C1 value reduction times answer probability, per cost."""
    ids = np.asarray(getattr(state, "_eligible_indices", state.hidden_indices), dtype=int)
    if len(ids) == 0:
        raise H2InputError("no eligible hidden rows")
    r = np.asarray(answer_probabilities, dtype=float)
    if r.shape != (state.n_rows,) or np.any((r < 0) | (r > 1)):
        raise H2InputError("answer_probabilities must have length N in [0,1]")
    scores = np.array([r[i] * _expected_gain(state, int(i), r[i]) / state.costs[i]
                       for i in ids])
    top = np.max(scores)
    tied = ids[np.isclose(scores, top, atol=1e-15, rtol=0)]
    return int(tied[np.argmin(state.costs[tied])])


def r2_policy(state, lower_answer_probabilities, rng, exploration=0.1):
    """Conservative R1 with random exploration among eligible rows."""
    ids = np.asarray(getattr(state, "_eligible_indices", state.hidden_indices), dtype=int)
    if len(ids) == 0:
        raise H2InputError("no eligible hidden rows")
    if rng.random() < float(exploration):
        return int(rng.choice(ids))
    return r1_policy(state, lower_answer_probabilities)


class AttemptOracle:
    """Private oracle for one-shot attempts with independent response draws."""

    def __init__(self, labels, answer_probabilities, rng):
        self.labels = np.asarray(labels, dtype=int).copy()
        self.answer_probabilities = np.asarray(answer_probabilities, dtype=float)
        self.rng = rng
        self.attempted = set()

    def attempt(self, row_id):
        row_id = int(row_id)
        if row_id in self.attempted:
            raise H2InputError("an outcome may be attempted only once")
        self.attempted.add(row_id)
        success = bool(self.rng.random() < self.answer_probabilities[row_id])
        return success, int(self.labels[row_id]) if success else None


def run_nonresponse(state, labels, answer_probabilities, policy_name, epsilon,
                    budget, seed=0, lower_answer_probabilities=None,
                    exploration=0.1, policy_answer_probabilities=None):
    """Run one H3 trajectory; labels are private and never passed to policy."""
    rng = np.random.default_rng(seed)
    answer_probabilities = np.asarray(answer_probabilities, dtype=float)
    if policy_answer_probabilities is None:
        policy_answer_probabilities = answer_probabilities
    policy_answer_probabilities = np.asarray(policy_answer_probabilities, dtype=float)
    if policy_answer_probabilities.shape != answer_probabilities.shape:
        raise H2InputError("policy and actual answer probabilities must have equal shapes")
    if lower_answer_probabilities is None:
        lower_answer_probabilities = answer_probabilities
    oracle = AttemptOracle(labels, answer_probabilities, np.random.default_rng(seed + 100003))
    attempted = np.zeros(state.n_rows, dtype=bool)
    trace = []
    while True:
        eligible = np.flatnonzero(~attempted & ~state.observed)
        if len(eligible) == 0:
            break
        if state.spent + float(np.min(state.costs[eligible])) > budget:
            break
        state._eligible_indices = eligible
        if policy_name == "r0":
            row_id = r0_policy(state, rng)
        elif policy_name == "random":
            row_id = random_policy(state, rng)
        elif policy_name == "r3":
            row_id = q0(state)
        elif policy_name == "r1":
            row_id = r1_policy(state, policy_answer_probabilities)
        elif policy_name == "r2":
            row_id = r2_policy(state, lower_answer_probabilities, rng, exploration)
        else:
            raise ValueError("policy_name must be r0, r1 or r2")
        attempted[row_id] = True
        cost = float(state.costs[row_id])
        v_before = state.value
        success, label = oracle.attempt(row_id)
        event = {"row_id": int(row_id), "success": success, "cost": cost,
                 "V_before": v_before, "V_after": v_before}
        if success:
            state.reveal(row_id, label)
            event["label"] = int(label)
            event["V_after"] = state.value
        else:
            state.spent += cost
        trace.append(event)
        if state.certificate(epsilon)["certified"]:
            break
    result = state.certificate(epsilon)
    result.update({"status": "certified" if result["certified"] else "budget_exhausted",
                   "total_cost": float(state.spent),
                   "queries_attempted": int(attempted.sum()),
                   "labels_obtained": int(state.observed.sum()), "trace": trace})
    return result

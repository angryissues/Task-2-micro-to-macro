"""Active label querying under the exact C1 Brier-regret certificate.

This module intentionally separates the public selection state from the oracle.
Policies only receive ``SelectionState`` and cannot receive the full label vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

import numpy as np


class H2InputError(ValueError):
    """Raised when an H2 experiment input is inconsistent."""


def _validate(predictions, labels, observed, costs):
    p = np.asarray(predictions, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise H2InputError("predictions must be a non-empty N-by-K array")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise H2InputError("predictions must lie in [0, 1]")
    n = p.shape[0]
    y = np.asarray(labels, dtype=float)
    seen = np.asarray(observed, dtype=bool)
    c = np.asarray(costs, dtype=float)
    if y.shape != (n,) or seen.shape != (n,) or c.shape != (n,):
        raise H2InputError("labels, observed and costs must have length N")
    if not np.isin(y, [0.0, 1.0]).all():
        raise H2InputError("oracle labels must be binary")
    if not np.isfinite(c).all() or np.any(c <= 0):
        raise H2InputError("costs must be finite and positive")
    return p, y, seen, c


@dataclass
class SelectionState:
    """Public state available to a query policy.

    ``labels`` contains values only where ``observed`` is true; every hidden value
    is replaced by zero and must never be interpreted as a real outcome.
    """

    predictions: np.ndarray
    labels: np.ndarray
    observed: np.ndarray
    costs: np.ndarray
    weights: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    spent: float = 0.0
    query_history: list[dict] = field(default_factory=list)

    @classmethod
    def create(cls, predictions, labels, observed, costs=None, weights=None):
        p = np.asarray(predictions, dtype=float)
        n = p.shape[0]
        if costs is None:
            costs = np.ones(n)
        if weights is None:
            weights = np.full(n, 1.0 / n)
        p, y, seen, c = _validate(p, labels, observed, costs)
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise H2InputError("weights must be non-negative with positive sum")
        w = w / w.sum()
        public_y = np.zeros(n)
        public_y[seen] = y[seen]
        lo, up = c1_bounds(p, public_y, seen, w)
        return cls(p, public_y, seen.copy(), c, w, lo, up)

    @property
    def n_rows(self):
        return self.predictions.shape[0]

    @property
    def n_models(self):
        return self.predictions.shape[1]

    @property
    def hidden_indices(self):
        return np.flatnonzero(~self.observed)

    @property
    def worst_regret(self):
        return self.upper.max(axis=1)

    @property
    def value(self):
        """V(U): the smallest available upper regret bound."""
        return float(self.worst_regret.min())

    def certificate(self, epsilon):
        g = self.worst_regret
        model = int(np.argmin(g))
        return {"recommended_model": model, "G": float(g[model]),
                "certified": bool(g[model] <= epsilon + 1e-12)}

    def projected_value(self, row_id, label):
        """Return V(U) after a hypothetical binary answer; does not mutate state."""
        _, up = reveal_bounds(self.lower, self.upper, self.predictions[row_id], label,
                              self.weights[row_id])
        return float(up.max(axis=1).min())

    def reveal(self, row_id, label):
        if row_id < 0 or row_id >= self.n_rows or self.observed[row_id]:
            raise H2InputError("only a currently hidden row may be revealed")
        if label not in (0, 1):
            raise H2InputError("revealed label must be 0 or 1")
        self.lower, self.upper = reveal_bounds(
            self.lower, self.upper, self.predictions[row_id], label, self.weights[row_id]
        )
        self.observed[row_id] = True
        self.labels[row_id] = label
        self.spent += float(self.costs[row_id])
        self.query_history.append({"row_id": int(row_id), "label": int(label),
                                   "cost": float(self.costs[row_id]), "V_after": self.value})


class LabelOracle:
    """Private label holder. Only ``query`` exposes an outcome, one row at a time."""

    def __init__(self, labels):
        y = np.asarray(labels, dtype=float)
        if y.ndim != 1 or not np.isin(y, [0.0, 1.0]).all():
            raise H2InputError("oracle labels must be a binary vector")
        self._labels = y.copy()
        self._queried: set[int] = set()

    def query(self, row_id):
        if row_id in self._queried:
            raise H2InputError("a label may be queried only once")
        self._queried.add(int(row_id))
        return int(self._labels[row_id])


def c1_bounds(p, y, seen, w):
    """Exact C1 pairwise bounds; local frozen copy of the H1 C1 arithmetic."""
    k = p.shape[1]
    hidden = ~seen
    lo, up = np.zeros((k, k)), np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            const = w * (p[:, a] ** 2 - p[:, b] ** 2)
            coeff = -2.0 * w * (p[:, a] - p[:, b])
            center = const.sum() + (coeff[seen] * y[seen]).sum()
            low = center + np.minimum(coeff[hidden], 0).sum()
            high = center + np.maximum(coeff[hidden], 0).sum()
            lo[a, b], up[a, b] = low, high
            lo[b, a], up[b, a] = -high, -low
    return lo, up


def reveal_bounds(lower, upper, p_row, label, weight):
    coeff = -2.0 * weight * (p_row[:, None] - p_row[None, :])
    lo = lower - np.minimum(coeff, 0) + coeff * label
    up = upper - np.maximum(coeff, 0) + coeff * label
    np.fill_diagonal(lo, 0.0)
    np.fill_diagonal(up, 0.0)
    return lo, up


def _tie_break(indices, scores, costs):
    """Max score; then lower cost; then lower stable row id."""
    top = np.max(scores)
    candidates = indices[np.isclose(scores, top, atol=1e-15, rtol=0)]
    cheapest = np.min(costs[candidates])
    return int(candidates[np.flatnonzero(costs[candidates] == cheapest)[0]])


def _candidate_ids(state):
    """Hidden rows available to the current policy; runner may impose a budget."""
    eligible = getattr(state, '_eligible_indices', None)
    return state.hidden_indices if eligible is None else np.asarray(eligible, dtype=int)


def q0(state: SelectionState):
    ids = _candidate_ids(state)
    disagreement = (state.predictions[ids].max(axis=1) - state.predictions[ids].min(axis=1))
    return _tie_break(ids, disagreement / state.costs[ids], state.costs)


def q1(state: SelectionState):
    ids = _candidate_ids(state)
    g = state.worst_regret
    k_star = int(np.argmin(g))
    j_star = int(np.argmax(state.upper[k_star]))
    score = 2.0 * state.weights[ids] * np.abs(
        state.predictions[ids, k_star] - state.predictions[ids, j_star]
    ) / state.costs[ids]
    return _tie_break(ids, score, state.costs)


def _lookahead(state: SelectionState, q=None):
    ids = _candidate_ids(state)
    current = state.value
    values0 = np.array([state.projected_value(i, 0) for i in ids])
    values1 = np.array([state.projected_value(i, 1) for i in ids])
    if q is None:
        post = np.maximum(values0, values1)
    else:
        q = np.asarray(q, dtype=float)
        if q.shape != (state.n_rows,) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise H2InputError("q must be a length-N probability vector")
        post = (1.0 - q[ids]) * values0 + q[ids] * values1
    score = (current - post) / state.costs[ids]
    if np.all(score <= 1e-15):
        return q1(state)
    return _tie_break(ids, score, state.costs)


def q2(state: SelectionState):
    return _lookahead(state)


def q3(state: SelectionState, q):
    return _lookahead(state, q=q)


def random_policy(state: SelectionState, rng):
    return int(rng.choice(_candidate_ids(state)))


def uncertainty_sampling(state: SelectionState, q):
    """Baseline: choose the largest auxiliary-label uncertainty per unit cost."""
    q = np.asarray(q, dtype=float)
    if q.shape != (state.n_rows,) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise H2InputError("q must be a length-N probability vector")
    ids = _candidate_ids(state)
    return _tie_break(ids, q[ids] * (1.0 - q[ids]) / state.costs[ids], state.costs)


def make_stratified_random(strata, rng):
    """Cyclic random baseline over non-empty public strata."""
    strata = np.asarray(strata)
    if strata.ndim != 1:
        raise H2InputError("strata must be a vector")
    order = list(rng.permutation(np.unique(strata)))
    cursor = 0

    def choose(state: SelectionState):
        nonlocal cursor
        if len(strata) != state.n_rows:
            raise H2InputError("strata and state have different lengths")
        hidden = _candidate_ids(state)
        for _ in range(len(order)):
            group = order[cursor % len(order)]
            cursor = (cursor + 1) % len(order)
            members = hidden[strata[hidden] == group]
            if len(members):
                return int(rng.choice(members))
        raise H2InputError("no non-empty stratum found")

    return choose


def hara_sampling(state: SelectionState, q, rng):
    """Hara et al. query-distribution adapter for binary Brier candidates.

    The score is the plan's Brier adaptation of equation (11).  This is an
    adapter for query ordering, not a claim that the paper's original objective
    and stopping rule are reproduced unchanged.  Returns the selected row and
    its *pre-query* selection probability for later LURE-style analyses.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (state.n_rows,) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise H2InputError("q must be a length-N probability vector")
    ids = _candidate_ids(state)
    p = state.predictions[ids]
    score = np.zeros(len(ids))
    for a in range(state.n_models):
        for b in range(a + 1, state.n_models):
            score += ((1.0 - q[ids]) * np.abs(p[:, a] ** 2 - p[:, b] ** 2)
                      + q[ids] * np.abs((p[:, a] - 1.0) ** 2 - (p[:, b] - 1.0) ** 2))
    if np.allclose(score.sum(), 0.0):
        pi = np.full(len(ids), 1.0 / len(ids))
    else:
        pi = .9 * score / score.sum() + .1 / len(ids)
    picked = int(rng.choice(len(ids), p=pi))
    return int(ids[picked]), float(pi[picked])


def active_testing_brier_adapter(state: SelectionState, q, rng):
    """Active Testing expected-loss query distribution for multiple Brier models.

    Kossen et al. sample proportionally to a model's expected loss.  Their
    method evaluates one model at a time; this explicit H2 adapter uses the
    mean expected binary Brier loss across all frozen candidates.  It shares
    the common C1 stopping rule rather than the paper's LURE estimator.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (state.n_rows,) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise H2InputError("q must be a length-N probability vector")
    ids = _candidate_ids(state)
    p = state.predictions[ids]
    expected_loss = ((1.0 - q[ids, None]) * p ** 2 + q[ids, None] * (1.0 - p) ** 2).mean(axis=1)
    pi = np.full(len(ids), 1.0 / len(ids)) if np.allclose(expected_loss.sum(), 0.0) else expected_loss / expected_loss.sum()
    picked = int(rng.choice(len(ids), p=pi))
    return int(ids[picked]), float(pi[picked])


def run_policy(state: SelectionState, oracle: LabelOracle, policy: Callable,
               epsilon: float, budget: float, q: Optional[np.ndarray] = None,
               rng: Optional[np.random.Generator] = None):
    """Run the prescribed H2 loop and return its decision trace."""
    if budget < 0 or epsilon < 0:
        raise H2InputError("budget and epsilon must be non-negative")
    rng = np.random.default_rng() if rng is None else rng
    while True:
        decision = state.certificate(epsilon)
        if decision["certified"]:
            status = "certified"
            break
        eligible = state.hidden_indices[state.costs[state.hidden_indices] <= budget - state.spent + 1e-12]
        if not len(eligible):
            status = "budget_exhausted"
            break
        # The policy may inspect the public state, but must return an affordable hidden row.
        state._eligible_indices = eligible
        if policy is random_policy:
            row = policy(state, rng)
        elif policy in (q3, uncertainty_sampling):
            row = policy(state, q)
        elif policy in (hara_sampling, active_testing_brier_adapter):
            row = policy(state, q, rng)
        else:
            row = policy(state)
        if isinstance(row, tuple):
            row, selection_probability = row
        else:
            selection_probability = None
        if row not in eligible:
            raise H2InputError("policy returned a non-hidden or unaffordable row")
        before = state.value
        label = oracle.query(row)
        state.reveal(row, label)
        state.query_history[-1]["V_before"] = before
        if selection_probability is not None:
            state.query_history[-1]["selection_probability"] = selection_probability
    return {**decision, "status": status, "total_cost": state.spent,
            "queries_obtained": len(state.query_history), "trace": state.query_history.copy()}


def _clone_after_reveal(state, row_id, label):
    """Copy public state and reveal a hypothetical label (used only by the exact oracle)."""
    clone = SelectionState(
        state.predictions, state.labels.copy(), state.observed.copy(), state.costs,
        state.weights, state.lower.copy(), state.upper.copy()
    )
    clone.reveal(row_id, label)
    return clone


def _state_key(state):
    """Sufficient public information for a fixed input cohort and C1 calculation."""
    return state.observed.tobytes(), state.labels.tobytes()


def exact_minimax_cost(initial_state: SelectionState, epsilon: float):
    """Minimum worst-case extra cost over all adaptive queries for M <= 8.

    This is the plan's exact tree recurrence, not an oracle using real hidden labels.
    Both binary branches are evaluated after every potential query.
    """
    if len(initial_state.hidden_indices) > 8:
        raise H2InputError("exact minimax tree is restricted to at most 8 hidden labels")
    cache = {}

    def solve(state):
        key = _state_key(state)
        if key in cache:
            return cache[key]
        if state.certificate(epsilon)['certified']:
            cache[key] = (0.0, None)
            return cache[key]
        ids = state.hidden_indices
        if not len(ids):
            cache[key] = (float('inf'), None)
            return cache[key]
        candidates = []
        for row_id in ids:
            left = solve(_clone_after_reveal(state, int(row_id), 0))[0]
            right = solve(_clone_after_reveal(state, int(row_id), 1))[0]
            candidates.append((float(state.costs[row_id]) + max(left, right), int(row_id)))
        cache[key] = min(candidates, key=lambda pair: (pair[0], pair[1]))
        return cache[key]

    cost, first_query = solve(initial_state)
    return {'worst_case_cost': cost, 'first_query': first_query, 'states_evaluated': len(cache)}


def policy_worst_case_cost(initial_state: SelectionState, epsilon: float, policy: Callable,
                           q: Optional[np.ndarray] = None):
    """Worst-case cost of a deterministic public-state policy on the same small tree."""
    if len(initial_state.hidden_indices) > 8:
        raise H2InputError("exact policy tree is restricted to at most 8 hidden labels")
    cache = {}

    def solve(state):
        key = _state_key(state)
        if key in cache:
            return cache[key]
        if state.certificate(epsilon)['certified']:
            cache[key] = 0.0
            return 0.0
        if not len(state.hidden_indices):
            cache[key] = float('inf')
            return cache[key]
        row_id = policy(state, q) if policy in (q3, uncertainty_sampling) else policy(state)
        if row_id not in state.hidden_indices:
            raise H2InputError("policy returned a non-hidden row in exact tree")
        cost = float(state.costs[row_id]) + max(
            solve(_clone_after_reveal(state, row_id, 0)),
            solve(_clone_after_reveal(state, row_id, 1)),
        )
        cache[key] = cost
        return cost

    return {'worst_case_cost': solve(initial_state), 'states_evaluated': len(cache)}

"""C3: random-audit statistical bounds.  >= 1000 repetitions on fixed fully known problems."""
import numpy as np

from src.bounds import c1_bounds, c3_bounds, intersect_intervals
from src.certification import select_with_certificate

TOL = 1e-10


def _problem(seed=0, n=300, k=3, miss=0.4, gap=0.1):
    rng = np.random.default_rng(seed)
    prob = rng.random(n) * 0.8 + 0.1
    y = (rng.random(n) < prob).astype(float)
    p = np.column_stack([np.clip(prob + gap * j * rng.normal(size=n), 0.01, 0.99) for j in range(k)])
    seen0 = rng.random(n) >= miss
    R = ((p - y[:, None]) ** 2).mean(axis=0)
    return p, y, seen0, R


def test_c3_all_pairs_all_budgets_coverage_1000_reps():
    p, y, seen0, R = _problem()
    D = R[:, None] - R[None, :]
    hidden = np.flatnonzero(~seen0)
    budgets = [10, 40, 80]
    alpha, violations, reps = 0.05, 0, 1000
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        order = rng.permutation(hidden)
        bad = False
        for m in budgets:                       # nested audits: prefixes of one permutation
            r = c3_bounds(p, y, seen0, order[:m], alpha, n_budgets=len(budgets))
            ls, us = r["lower_stat"], r["upper_stat"]
            if np.any(D < ls - TOL) or np.any(D > us + TOL):
                bad = True
            lo, up = r["lower"], r["upper"]
            if not r["event_failed"] and (np.any(D < lo - TOL) or np.any(D > up + TOL)):
                bad = True
        violations += bad
    assert violations / reps <= alpha, violations


def test_c3_limits_and_fallback():
    p, y, seen0, R = _problem(seed=1, n=120)
    hidden = np.flatnonzero(~seen0)
    r0 = c3_bounds(p, y, seen0, np.array([], int))                     # m=0: only C1
    lo1, up1, _ = c1_bounds(p, np.where(seen0, y, np.nan), seen0)
    assert np.allclose(r0["upper"], up1) and r0["upper_stat"] is None
    rM = c3_bounds(p, y, seen0, hidden)                                 # m=M: exact differences
    D = R[:, None] - R[None, :]
    assert np.abs(rM["upper"] - D).max() < TOL and np.abs(rM["lower"] - D).max() < TOL
    # an empty intersection (statistical event demonstrably failed) must fall back to C1 unchanged
    lo_c1 = np.array([[0.0, -0.1], [-0.1, 0.0]])
    up_c1 = np.array([[0.0, 0.1], [0.1, 0.0]])
    lo_s = np.array([[0.0, 0.3], [-0.5, 0.0]])
    up_s = np.array([[0.0, 0.5], [-0.3, 0.0]])             # disjoint from C1 for the pair (0,1)
    lo, up, failed = intersect_intervals(lo_s, up_s, lo_c1, up_c1)
    assert failed and np.array_equal(lo, lo_c1) and np.array_equal(up, up_c1)
    lo, up, failed = intersect_intervals(np.full((2, 2), -1.0), np.full((2, 2), 1.0), lo_c1, up_c1)
    assert not failed and np.array_equal(lo, lo_c1) and np.array_equal(up, up_c1)
    # natural failures are rare because the C1 interval after the audit is wide; just run the path
    for rep in range(200):
        idx = np.random.default_rng(rep).choice(hidden, size=5, replace=False)
        r = c3_bounds(p, y, seen0, idx)
        assert np.all(r["upper"] <= r["upper_c1"] + TOL) and np.all(r["lower"] >= r["lower_c1"] - TOL)


def test_c3_reads_only_observed_and_audited_labels():
    p, y, seen0, R = _problem(seed=2, n=100)
    hidden = np.flatnonzero(~seen0)
    idx = hidden[:15]
    a = c3_bounds(p, y, seen0, idx)
    y2 = y.copy()
    other = np.setdiff1d(hidden, idx)
    y2[other] = 1.0 - y2[other]                       # flip every NON-audited hidden label
    b = c3_bounds(p, y2, seen0, idx)
    assert np.array_equal(a["upper"], b["upper"]) and np.array_equal(a["lower"], b["lower"])
    # certified C3 decisions never rely on non-audited labels either
    r1 = select_with_certificate(p, y, seen0, 0.2, method="C3", audit_idx=idx)
    r2 = select_with_certificate(p, y2, seen0, 0.2, method="C3", audit_idx=idx)
    assert r1["certified"] == r2["certified"] and r1["recommended_model"] == r2["recommended_model"]

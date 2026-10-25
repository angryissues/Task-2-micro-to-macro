"""C2: constrained bounds vs enumeration restricted to admissible fills, and premise violations."""
import numpy as np

from helpers import brute_force, make_problem
from src.bounds import (audit_count_interval, c1_bounds, c2_bounds, scenario_count_bounds,
                        update_count_bounds_after_reveal)
from src.certification import select_with_certificate

TOL = 1e-10


def _case(rng, n=11, k=3, m=9, groups=2):
    p, y, y_in, seen, w = make_problem(rng, n, k, m, "random", "random")
    gid = rng.integers(0, groups, size=n)
    return p, y, y_in, seen, w, gid


def _filter_for(hidden_idx, gid, ranges):
    def f(bits):
        for g, (A, B) in ranges.items():
            s = bits[gid[hidden_idx] == g].sum()
            if not (A <= s <= B):
                return False
        return True
    return f


def test_c2_matches_constrained_enumeration():
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(60):
        p, y, y_in, seen, w, gid = _case(rng)
        hidx = np.flatnonzero(~seen)
        cb = {}
        for g in (0, 1):
            m_g = int(np.sum(gid[hidx] == g))
            A = int(rng.integers(0, m_g + 1))
            B = int(rng.integers(A, m_g + 1))
            cb[g] = (A, B)
        if rng.random() < 0.4:                      # leave one group unconstrained
            cb.pop(1)
        lo, up, g_, st = c2_bounds(p, y_in, seen, gid, cb, w)
        assert st == "ok"
        blo, bup, bg, cnt = brute_force(p, y_in, seen, w, _filter_for(hidx, gid, cb))
        assert cnt > 0
        assert np.abs(lo - blo).max() < TOL and np.abs(up - bup).max() < TOL
        assert np.abs(g_ - bg).max() < TOL
        checked += 1
    assert checked == 60


def test_weak_constraint_equals_c1_and_narrower_nests():
    rng = np.random.default_rng(12)
    p, y, y_in, seen, w, gid = _case(rng, n=12, m=10)
    hidx = np.flatnonzero(~seen)
    m0 = int(np.sum(gid[hidx] == 0))
    lo1, up1, g1 = c1_bounds(p, y_in, seen, w)
    lo, up, g, _ = c2_bounds(p, y_in, seen, gid, {0: (0, m0)}, w)            # too weak
    assert np.abs(lo - lo1).max() < TOL and np.abs(up - up1).max() < TOL
    # tighter constraints give nested (narrower) intervals; truth-centred ones stay feasible
    truth = int(y[hidx][gid[hidx] == 0].sum())
    prev = (lo1, up1)
    for h in (m0, max(m0 // 2, 1), 1, 0):
        A, B = max(0, truth - h), min(m0, truth + h)
        lo, up, g, st = c2_bounds(p, y_in, seen, gid, {0: (A, B)}, w)
        assert st == "ok"
        assert np.all(lo >= prev[0] - TOL) and np.all(up <= prev[1] + TOL)
        prev = (lo, up)
    # true constraint => true differences stay inside the interval
    R = ((p - y[:, None]) ** 2 * (np.asarray(w) / np.sum(w))[:, None]).sum(axis=0)
    D = R[:, None] - R[None, :]
    assert np.all(D >= prev[0] - TOL) and np.all(D <= prev[1] + TOL)


def test_incompatible_constraint_detected():
    rng = np.random.default_rng(13)
    p, y, y_in, seen, w, gid = _case(rng, n=12, m=8, groups=1)
    m0 = int(np.sum(~seen))
    for bad in [(m0 + 1, m0 + 3), (-3, -1), (3, 2)]:
        lo, up, g, st = c2_bounds(p, y_in, seen, gid, {0: bad}, w)
        assert st == "assumption_infeasible" and np.isnan(up).all()
        r = select_with_certificate(p, y_in, seen, 0.5, w, method="C2", group_ids=gid,
                                    count_bounds={0: bad})
        assert r["status"] == "assumption_infeasible" and not r["certified"]
        assert r["selected_model"] is None and r["recommended_model"] is None


def test_reveal_residual_constraint_equals_enumeration_of_remaining():
    rng = np.random.default_rng(14)
    p, y, y_in, seen, w, gid = _case(rng, n=11, m=8, groups=1)
    hidx = np.flatnonzero(~seen)
    A, B = 2, 5
    r = int(hidx[0])
    cb2 = update_count_bounds_after_reveal({0: (A, B)}, 0, y[r], len(hidx) - 1)
    seen2 = seen.copy()
    seen2[r] = True
    y2 = np.where(seen2, y, np.nan)
    lo, up, g, st = c2_bounds(p, y2, seen2, gid, cb2, w)
    # enumerate the ORIGINAL admissible fills consistent with the revealed label
    hid_all = np.flatnonzero(~seen)
    pos = list(hid_all).index(r)
    f = lambda bits: (A <= bits.sum() <= B) and bits[pos] == y[r]
    blo, bup, bg, cnt = brute_force(p, np.where(seen, y, np.nan), seen, w, f)
    assert np.abs(lo - blo).max() < TOL and np.abs(up - bup).max() < TOL
    # revealing more ones than the upper count allows -> infeasible
    cbad = update_count_bounds_after_reveal({0: (0, 0)}, 0, 1, 5)
    _, _, _, st = c2_bounds(p, y2, seen2, gid, cbad, w)
    assert st == "assumption_infeasible"


def test_premise_violation_gives_intervals_missing_the_truth():
    """Count range that excludes the true count: C2 may exclude the true difference (undetected)."""
    rng = np.random.default_rng(15)
    found = False
    for _ in range(200):
        p, y, y_in, seen, w, gid = _case(rng, n=12, m=10, groups=1)
        hidx = np.flatnonzero(~seen)
        truth = int(y[hidx].sum())
        m0 = len(hidx)
        if truth + 2 > m0:
            continue
        A, B = truth + 1, min(m0, truth + 2)               # wrong but internally feasible
        lo, up, g, st = c2_bounds(p, y_in, seen, gid, {0: (A, B)}, w)
        assert st == "ok"                                   # NOT detected by the data
        R = ((p - y[:, None]) ** 2 * (np.asarray(w) / np.sum(w))[:, None]).sum(axis=0)
        D = R[:, None] - R[None, :]
        if np.any(D < lo - TOL) or np.any(D > up + TOL):
            found = True
            break
    assert found


def test_audit_interval_and_scenario_bounds():
    assert audit_count_interval(7, 20, 20, 1) == (7, 7)                 # whole group audited
    assert audit_count_interval(0, 0, 50, 1) == (0, 50)                 # no audit
    A, B = audit_count_interval(10, 40, 200, 2, 0.05)
    assert 0 <= A <= 50 <= B <= 200                                    # contains the sample estimate
    A2, B2 = audit_count_interval(100, 400, 200, 2, 0.05)
    assert (B2 - A2) < (B - A)                                         # larger audit, narrower
    assert scenario_count_bounds(30, 100, 50, float("inf")) == (0, 50)
    assert scenario_count_bounds(0, 0, 50, 2.0) == (0, 50)
    a1, b1 = scenario_count_bounds(30, 100, 50, 1.0)
    a2, b2 = scenario_count_bounds(30, 100, 50, 2.0)
    assert a2 <= a1 <= b1 <= b2
    # the audit interval covers the true count with prob >= 1-alpha (statistical, simulated)
    rng = np.random.default_rng(16)
    M, trials, miss = 200, 2000, 0
    pop = (rng.random(M) < 0.4).astype(int)
    for _ in range(trials):
        idx = rng.choice(M, size=40, replace=False)
        A, B = audit_count_interval(int(pop[idx].sum()), 40, M, 1, 0.05)
        miss += not (A <= pop.sum() <= B)
    assert miss / trials <= 0.05

"""Invariants of C1 and of the certification rule, plus input validation."""
import numpy as np

from helpers import brute_force, make_problem
from src.bounds import (H1InputError, c0_bounds, c1_bounds, reveal_update, validate_cohort)
from src.certification import select_with_certificate

TOL = 1e-10


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    return False


def test_antisymmetry_diagonal_width():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n, k = int(rng.integers(5, 40)), int(rng.integers(2, 7))
        m = int(rng.integers(0, n + 1))
        p, y, y_in, seen, w = make_problem(rng, n, k, m, "mixed", "random")
        lo, up, g = c1_bounds(p, y_in, seen, w)
        assert np.abs(lo + up.T).max() < TOL                       # L_ab = -U_ba
        assert np.abs(np.diag(up)).max() == 0 and np.abs(np.diag(lo)).max() == 0
        ww = np.asarray(w, float) / np.sum(w)
        hid = ~seen
        width = 2 * (ww[hid, None, None] * np.abs(p[hid, :, None] - p[hid, None, :])).sum(axis=0)
        assert np.abs((up - lo) - width).max() < TOL               # width = 2 sum_M w|pa-pb|
        assert np.all(lo <= up + TOL)


def test_monotone_when_labels_are_revealed():
    rng = np.random.default_rng(2)
    for _ in range(30):
        n, k = int(rng.integers(6, 30)), int(rng.integers(2, 6))
        p, y, y_in, seen, w = make_problem(rng, n, k, n, "random", "random")   # all hidden
        seen = seen.copy()
        lo, up, g = c1_bounds(p, y, seen, w)          # y passed, but hidden entries are ignored
        for i in rng.permutation(n):
            seen[i] = True
            lo2, up2, g2 = c1_bounds(p, y, seen, w)
            assert np.all(up2 <= up + TOL) and np.all(lo2 >= lo - TOL) and np.all(g2 <= g + TOL)
            lo, up, g = lo2, up2, g2
        assert np.abs(lo - up).max() < TOL            # everything revealed -> exact


def test_reveal_update_equals_recomputation():
    rng = np.random.default_rng(3)
    for _ in range(30):
        n, k = int(rng.integers(5, 25)), int(rng.integers(2, 6))
        p, y, y_in, seen, w = make_problem(rng, n, k, n // 2 + 1, "random", "random")
        lo, up, _ = c1_bounds(p, y_in, seen, w)
        r = int(rng.choice(np.flatnonzero(~seen)))
        ww = np.asarray(w, float) / np.sum(w)
        lo_u, up_u, g_u = reveal_update(lo, up, p[r], y[r], ww[r])
        seen2 = seen.copy()
        seen2[r] = True
        lo_r, up_r, g_r = c1_bounds(p, np.where(seen2, y, np.nan), seen2, w)
        assert np.abs(lo_u - lo_r).max() < TOL and np.abs(up_u - up_r).max() < TOL
        assert np.abs(g_u - g_r).max() < TOL


def test_hidden_labels_are_never_read():
    rng = np.random.default_rng(4)
    p, y, y_in, seen, w = make_problem(rng, 30, 3, 12, "random", "random")
    a = c1_bounds(p, y_in, seen, w)
    garbage = np.where(seen, y_in, rng.integers(0, 2, 30).astype(float))
    b = c1_bounds(p, garbage, seen, w)
    weird = np.where(seen, y_in, 7.5)                 # non-binary values on hidden rows are ignored
    c = c1_bounds(p, weird, seen, w)
    for x in (b, c):
        assert all(np.array_equal(u, v) for u, v in zip(a, x))


def test_duplicate_model_gives_no_artificial_advantage():
    rng = np.random.default_rng(5)
    for _ in range(30):
        n, k = int(rng.integers(6, 30)), int(rng.integers(2, 5))
        p, y, y_in, seen, w = make_problem(rng, n, k, n // 2, "random", "random")
        _, _, g = c1_bounds(p, y_in, seen, w)
        j = int(rng.integers(k))
        p2 = np.column_stack([p, p[:, j]])
        _, _, g2 = c1_bounds(p2, y_in, seen, w)
        assert np.abs(g2[:k] - g).max() < TOL                       # originals unchanged
        assert abs(g2[k] - g[j]) < TOL                              # duplicate = its original
        assert abs(g2.min() - g.min()) < TOL
        r1 = select_with_certificate(p, y_in, seen, 0.05, w)
        r2 = select_with_certificate(p2, y_in, seen, 0.05, w)
        assert r1["recommended_model"] == r2["recommended_model"]   # tie -> lowest id (the original)


def test_weight_normalisation_is_scale_invariant():
    rng = np.random.default_rng(6)
    p, y, y_in, seen, w = make_problem(rng, 20, 3, 8, "random", "random")
    a = c1_bounds(p, y_in, seen, w)
    b = c1_bounds(p, y_in, seen, 1000.0 * np.asarray(w))
    assert all(np.abs(u - v).max() < TOL for u, v in zip(a, b))
    # normalisation over ALL rows: zero-weight hidden rows contribute nothing
    w0 = np.asarray(w).copy()
    w0[~seen] = 0.0
    lo, up, _ = c1_bounds(p, y_in, seen, w0)
    assert np.abs(up - lo).max() < TOL


def test_input_validation():
    p = np.array([[0.2, 0.8], [0.5, 0.5]])
    y = np.array([1.0, np.nan])
    seen = np.array([True, False])
    assert raises(H1InputError, c1_bounds, p[:0], y[:0], seen[:0])                 # empty cohort
    assert raises(H1InputError, c1_bounds, p, y, np.array([True, True]))           # NaN observed
    assert raises(H1InputError, c1_bounds, np.array([[1.2, 0.1], [0.1, 0.1]]), y, seen)
    assert raises(H1InputError, c1_bounds, np.array([[np.nan, 0.1], [0.1, 0.1]]), y, seen)
    assert raises(H1InputError, c1_bounds, p, np.array([2.0, np.nan]), seen)
    assert raises(H1InputError, c1_bounds, p, y, seen, weights=[1.0, -1.0])
    assert raises(H1InputError, c1_bounds, p, y, seen, weights=[0.0, 0.0])
    assert raises(H1InputError, c1_bounds, p, y, seen, weights=[1.0])
    assert raises(H1InputError, select_with_certificate, p, y, seen, -0.1)
    assert raises(H1InputError, select_with_certificate, p, y, seen, float("nan"))
    lo, up, g = c1_bounds(p[:, :1], y, seen)                                      # K=1
    assert up.shape == (1, 1) and g[0] == 0.0


def test_certification_cases():
    rng = np.random.default_rng(8)
    n = 200
    truth = (rng.random(n) < 0.5).astype(float)
    good = np.where(truth == 1, 0.95, 0.05)
    bad = 1.0 - good
    mid = np.full(n, 0.5)
    p = np.column_stack([good, mid, bad])
    seen_all = np.ones(n, bool)
    # all labels known: the truly best model gets G=0 and is certified for every eps>=0
    r = select_with_certificate(p, truth, seen_all, 0.0)
    assert r["certified"] and r["selected_model"] == 0 and r["worst_regret"] == 0.0
    assert r["guarantee_type"] == "finite_exact" and r["status"] == "certified"
    # one model certified: 90% known, models far apart
    seen = rng.random(n) < 0.9
    r = select_with_certificate(p, np.where(seen, truth, np.nan), seen, 0.1)
    assert r["certified"] and r["diagnostics"]["certifiable_models"] == [0]
    # none certified: all hidden, spread models, small eps -> refuse; recommendation != certificate
    none = np.zeros(n, bool)
    r = select_with_certificate(p, np.full(n, np.nan), none, 0.001)
    assert not r["certified"] and r["selected_model"] is None and r["status"] == "insufficient_evidence"
    assert r["recommended_model"] is not None and r["worst_regret"] > 0.001
    # several certified: very large eps -> all models certifiable (G<=1 always), lowest-G chosen
    r = select_with_certificate(p, np.full(n, np.nan), none, 1.0)
    assert r["certified"] and r["diagnostics"]["n_certifiable"] == 3
    # identical forecasts: every model has G=0; certified for eps=0, tie -> id 0
    pid = np.column_stack([good, good, good])
    r = select_with_certificate(pid, np.full(n, np.nan), none, 0.0)
    assert r["certified"] and r["selected_model"] == 0 and r["diagnostics"]["tie_for_min_g"]
    assert np.all(r["g_all"] == 0.0)
    # all hidden but exact ties between hidden-equivalent models still certify nothing extra
    r = select_with_certificate(p, np.full(n, np.nan), none, 0.0, method="C0")
    assert not r["certified"]
    # C0 is never tighter than C1
    r0 = select_with_certificate(p, np.where(seen, truth, np.nan), seen, 0.1, method="C0")
    r1 = select_with_certificate(p, np.where(seen, truth, np.nan), seen, 0.1, method="C1")
    assert np.all(r0["g_all"] >= r1["g_all"] - TOL)

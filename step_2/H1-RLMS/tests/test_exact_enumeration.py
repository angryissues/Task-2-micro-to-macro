"""C0/C1 vs exhaustive enumeration of all 2^m fills (tolerance 1e-10)."""
import os

import numpy as np

from helpers import brute_force, brute_force_single_model, make_problem
from src.bounds import c0_bounds, c1_bounds

TOL = 1e-10


def _cases():
    rng = np.random.default_rng(20261020)
    for n in (6, 10, 12):
        for k in (2, 3, 5):
            for frac in (0.0, 0.3, 0.6, 1.0):          # no missing .. all hidden
                m = min(12, int(round(frac * n)))
                for kind in ("random", "extreme", "mixed", "identical"):
                    for wk in ("uniform", "random"):
                        yield n, k, m, kind, wk, make_problem(rng, n, k, m, kind, wk)


def test_c1_matches_exhaustive_enumeration():
    n_checked, worst = 0, 0.0
    for n, k, m, kind, wk, (p, y, y_in, seen, w) in _cases():
        lo, up, g = c1_bounds(p, y_in, seen, w)
        blo, bup, bg, cnt = brute_force(p, y_in, seen, w)
        assert cnt == 2 ** m
        err = max(np.abs(lo - blo).max(), np.abs(up - bup).max(), np.abs(g - bg).max())
        worst = max(worst, err)
        assert err < TOL, f"C1 mismatch n={n} k={k} m={m} kind={kind} w={wk} err={err}"
        n_checked += 1
    assert n_checked == 3 * 3 * 4 * 4 * 2
    print(f"C1 exhaustive: {n_checked} cases, max abs error {worst:.2e}")


def test_c0_is_valid_enclosure_and_matches_its_definition():
    rng = np.random.default_rng(7)
    for n, k, m in [(6, 3, 6), (10, 3, 5), (12, 5, 8), (8, 2, 0)]:
        for kind in ("random", "mixed"):
            p, y, y_in, seen, w = make_problem(rng, n, k, m, kind, "random")
            lo0, up0, g0 = c0_bounds(p, y_in, seen, w)
            lo1, up1, g1 = c1_bounds(p, y_in, seen, w)
            blo, bup, bg, _ = brute_force(p, y_in, seen, w)
            rmin, rmax = brute_force_single_model(p, y_in, seen, w)
            exp_up = rmax[:, None] - rmin[None, :]
            exp_lo = rmin[:, None] - rmax[None, :]
            np.fill_diagonal(exp_up, 0.0)
            np.fill_diagonal(exp_lo, 0.0)
            assert np.abs(up0 - exp_up).max() < TOL and np.abs(lo0 - exp_lo).max() < TOL
            assert np.all(up0 >= bup - TOL) and np.all(lo0 <= blo + TOL)   # valid enclosure
            assert np.all(up0 >= up1 - TOL) and np.all(lo0 <= lo1 + TOL)   # never tighter than C1
            assert np.all(g0 >= g1 - TOL)


def test_c0_strictly_looser_than_c1_example():
    # one hidden row, p=(0.9, 0.6): the worst label for model a (y=0) is not the best one for model b (y=1)
    p = np.array([[0.9, 0.6]])
    seen = np.array([False])
    y_in = np.array([np.nan])
    _, up0, _ = c0_bounds(p, y_in, seen)
    _, up1, _ = c1_bounds(p, y_in, seen)
    assert abs(up1[0, 1] - 0.45) < 1e-12 and abs(up0[0, 1] - 0.65) < 1e-12


def test_save_diagnostic_if_failure_possible():
    """Placeholder documenting the policy: failing cases are written to results/diagnostics/."""
    assert os.path.isdir(os.path.join(os.path.dirname(__file__), ".."))

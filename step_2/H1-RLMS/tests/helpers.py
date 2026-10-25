"""Independent brute-force oracle + random problem generators for the tests.

IMPORTANT: nothing here imports src.bounds / C1 code.  Losses are computed directly from
(p - y)^2 for every one of the 2^m fills of the hidden labels."""
import itertools

import numpy as np


def make_problem(rng, n, k, n_hidden, kind="random", weights="uniform"):
    if kind == "random":
        p = rng.random((n, k))
    elif kind == "extreme":                      # forecasts exactly 0 or 1
        p = rng.integers(0, 2, size=(n, k)).astype(float)
    elif kind == "mixed":
        p = rng.random((n, k))
        p[rng.random((n, k)) < 0.3] = 0.0
        p[rng.random((n, k)) < 0.3] = 1.0
    elif kind == "identical":
        p = np.repeat(rng.random((n, 1)), k, axis=1)
    else:
        raise ValueError(kind)
    y = rng.integers(0, 2, size=n).astype(float)
    hidden = np.zeros(n, bool)
    hidden[rng.choice(n, size=n_hidden, replace=False)] = True
    seen = ~hidden
    y_in = np.where(seen, y, np.nan)
    w = None if weights == "uniform" else rng.random(n) + 0.05
    return p, y, y_in, seen, w


def _norm_w(n, w):
    w = np.full(n, 1.0 / n) if w is None else np.asarray(w, float)
    return w / w.sum()


def brute_force(p, y_in, seen, w=None, fill_filter=None):
    """Enumerate every fill of hidden labels (optionally filtered); return
    lower, upper (K,K) of R_k - R_j and G (K,) = max over fills of regret_k, plus #fills."""
    n, k = p.shape
    w = _norm_w(n, w)
    hid = np.flatnonzero(~seen)
    lo = np.full((k, k), np.inf)
    up = np.full((k, k), -np.inf)
    g = np.full(k, -np.inf)
    count = 0
    for bits in itertools.product([0.0, 1.0], repeat=len(hid)):
        if fill_filter is not None and not fill_filter(np.array(bits)):
            continue
        y = np.array(y_in, float)
        y[hid] = bits
        R = np.array([sum(w[i] * (p[i, m] - y[i]) ** 2 for i in range(n)) for m in range(k)])
        D = R[:, None] - R[None, :]
        lo = np.minimum(lo, D)
        up = np.maximum(up, D)
        g = np.maximum(g, R - R.min())
        count += 1
    return lo, up, g, count


def brute_force_single_model(p, y_in, seen, w=None):
    """min / max of each model's own loss over all fills (for C0)."""
    n, k = p.shape
    w = _norm_w(n, w)
    hid = np.flatnonzero(~seen)
    rmin = np.full(k, np.inf)
    rmax = np.full(k, -np.inf)
    for bits in itertools.product([0.0, 1.0], repeat=len(hid)):
        y = np.array(y_in, float)
        y[hid] = bits
        R = np.array([sum(w[i] * (p[i, m] - y[i]) ** 2 for i in range(n)) for m in range(k)])
        rmin, rmax = np.minimum(rmin, R), np.maximum(rmax, R)
    return rmin, rmax

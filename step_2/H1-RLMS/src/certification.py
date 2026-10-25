"""Certification rule (plan 2.1): G_k = max_j U_kj; certify argmin_k G_k if G <= epsilon."""
from __future__ import annotations

import numpy as np

from .bounds import (CERT_TOL, H1InputError, c0_bounds, c1_bounds, c2_bounds, c3_bounds,
                     validate_cohort)

GUARANTEE = {"C0": "finite_exact", "C1": "finite_exact", "C2": "conditional_counts",
             "C3": "statistical_random_audit"}


def select_with_certificate(predictions, labels, observed, epsilon, weights=None, method="C1",
                            group_ids=None, count_bounds=None,
                            audit_idx=None, alpha=0.05, n_budgets=1, observed_initial=None):
    """Return a certificate decision.

    method: "C0", "C1", "C2" (needs group_ids, count_bounds), "C3" (needs audit_idx; `observed`
    is then the INITIAL mask, labels of audit rows are read through `audit_idx` only).

    Result keys
      selected_model      certified model index or None (never filled by a mere recommendation)
      recommended_model   argmin_k G_k (ties -> lowest id); NOT certified unless `certified`
      worst_regret        G of the recommended model (nan if assumptions infeasible)
      epsilon, certified, status, guarantee_type, lower, upper, g_all, diagnostics
    Not certified != the model is bad: it only means the available information cannot exclude
    a regret above epsilon.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise H1InputError("epsilon must be finite and non-negative")
    flags, extra = [], {}
    status_override = None
    if method == "C0":
        lo, up, g = c0_bounds(predictions, labels, observed, weights)
    elif method == "C1":
        lo, up, g = c1_bounds(predictions, labels, observed, weights)
    elif method == "C2":
        if group_ids is None or count_bounds is None:
            raise H1InputError("C2 needs group_ids and count_bounds")
        lo, up, g, st = c2_bounds(predictions, labels, observed, group_ids, count_bounds, weights)
        if st != "ok":
            status_override = "assumption_infeasible"
    elif method == "C3":
        if weights is not None and not np.allclose(np.asarray(weights) / np.sum(weights),
                                                   1.0 / len(weights)):
            raise H1InputError("C3 is defined for equal weights only")
        if audit_idx is None:
            raise H1InputError("C3 needs audit_idx")
        r = c3_bounds(predictions, labels, observed, audit_idx, alpha, n_budgets)
        lo, up = r["lower"], r["upper"]
        g = up.max(axis=1)
        extra.update(m_audit=r["m"], M_hidden_initial=r["M"], radius=r["radius"])
        if r["event_failed"]:
            flags.append("statistical_event_failed")      # fell back to C1 after the same reveals
        guarantee = "finite_exact" if r["event_failed"] else GUARANTEE["C3"]
    else:
        raise H1InputError(f"unknown method {method!r}")
    if method != "C3":
        guarantee = GUARANTEE[method]

    seen = np.asarray(observed, bool)
    n, k = np.asarray(predictions).shape
    diag = {"n_rows": n, "n_models": k, "n_hidden": int((~seen).sum()),
            "missing_fraction": float((~seen).mean()), "flags": flags, **extra}
    return decide_from_bounds(lo, up, g, epsilon, guarantee, diag,
                              infeasible=(status_override == "assumption_infeasible"))


def decide_from_bounds(lo, up, g, epsilon, guarantee_type, diagnostics=None, infeasible=False):
    """Apply the certification rule to already computed bounds (so that one set of bounds can be
    evaluated at many epsilons without recomputation)."""
    diag = dict(diagnostics or {})
    if infeasible:
        return {"selected_model": None, "recommended_model": None, "worst_regret": float("nan"),
                "epsilon": float(epsilon), "certified": False, "status": "assumption_infeasible",
                "guarantee_type": guarantee_type, "lower": lo, "upper": up, "g_all": g,
                "diagnostics": diag}
    k = len(g)
    rec = int(np.argmin(g))                       # ties -> lowest id
    gmin = float(g[rec])
    certified = bool(gmin <= epsilon + CERT_TOL)
    ok = np.flatnonzero(g <= epsilon + CERT_TOL)
    off = ~np.eye(k, dtype=bool)
    diag.update(n_certifiable=int(len(ok)), certifiable_models=ok.tolist(),
                tie_for_min_g=int(np.sum(np.isclose(g, gmin, atol=1e-15, rtol=0))) > 1,
                mean_interval_width=float((up - lo)[off].mean()) if k > 1 else 0.0,
                max_interval_width=float((up - lo)[off].max()) if k > 1 else 0.0)
    return {"selected_model": rec if certified else None, "recommended_model": rec,
            "worst_regret": gmin, "epsilon": float(epsilon), "certified": certified,
            "status": "certified" if certified else "insufficient_evidence",
            "guarantee_type": guarantee_type, "lower": lo, "upper": up, "g_all": g,
            "diagnostics": diag}

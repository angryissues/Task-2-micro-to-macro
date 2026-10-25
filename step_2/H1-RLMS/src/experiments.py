"""Experiment runner for H1 (synthetic oracle mode and real incomplete mode).

    python -m src.experiments run --config configs/h1_default.json --out results/synthetic
    python -m src.experiments summarize --out results/synthetic
    python -m src.experiments real --input my_table.csv.gz --model-cols p_M0,p_M1,p_M2 --out results/real_run

Every unit of work (one cell of a study) is written to <out>/chunks/<study>/<cell>.csv.gz as soon as
it is finished; re-running skips finished chunks, so an interrupted run can be resumed.
Nothing here reads files other than the config and (real mode) the table you point it to; nothing
is sent anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from . import baselines as bl
from .bounds import (TOL, audit_count_interval, c0_bounds, c1_bounds, c2_bounds, c3_bounds,
                     scenario_count_bounds, update_count_bounds_after_reveal)
from .certification import decide_from_bounds
from .metrics import (clopper_pearson, coverage_all_pairs, full_brier, true_regret_vector,
                      wrong_certificate)
from .synthetic import draw_mask, make_cohort, observation_probabilities


# ----------------------------------------------------------------------------- infrastructure
def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    return cfg


def apply_quick(cfg):
    """Tiny version of the config for smoke tests (results NOT meaningful)."""
    c = json.loads(json.dumps(cfg))
    c["main"].update(n=200, n_datasets=1, n_masks=2, past_n=300, spreads=["moderate"], mechanisms=["MCAR", "MNAR"])
    c["missing_fractions"] = [0.2]
    c["size"].update(n=[50, 200], n_datasets=1, n_masks=2)
    c["kstudy"].update(k=[2, 3], n=200, n_datasets=1, n_masks=2, spreads=["moderate"])
    c["c2"].update(n=200, n_datasets=1, n_masks=2, spreads=["moderate"], oracle_half_widths=[0, 5],
                   violation_shifts=[5], audit_sizes=[10], gammas=[1.0, "inf"], widen=[0.0])
    c["c2stress"].update(n=200, n_datasets=1, n_masks=2, spreads=["moderate"], shifts=[5, 100], gammas=[1.0], widen=[0.0])
    c["c3"].update(n=[200], spreads=["moderate"], missing=[0.2], reps=30, budget_fractions=[0.25, 0.5])
    c["_sha256"] = "quick"
    return c


def chunk_path(out, study, key):
    d = os.path.join(out, "chunks", study)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.csv.gz")


def run_chunk(out, study, key, fn, log=True):
    path = chunk_path(out, study, key)
    if os.path.exists(path):
        if log:
            print(f"[skip] {study}/{key}")
        return
    t = time.perf_counter()
    df = fn()
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    if log:
        print(f"[done] {study}/{key}: {len(df)} rows, {time.perf_counter() - t:.1f}s")


def combine_chunks(out, study):
    d = os.path.join(out, "chunks", study)
    if not os.path.isdir(d):
        return None
    parts = [pd.read_csv(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".csv.gz")]
    return pd.concat(parts, ignore_index=True) if parts else None


def _eps_lists(cfg):
    plan = list(cfg["epsilon_grid"])
    return plan, plan + [e for e in cfg.get("epsilon_extra", []) if e not in plan]


def _offdiag_mean(lo, up):
    k = lo.shape[0]
    return float((up - lo)[~np.eye(k, dtype=bool)].mean()) if k > 1 else 0.0


# ----------------------------------------------------------------------------- row builders
def interval_rows(base, method, lo, up, g, guarantee, R, eps_all, eps_plan, runtime,
                  infeasible=False, extra=None):
    tr = true_regret_vector(R)
    best = int(np.argmin(R))
    cov = float("nan") if infeasible else float(coverage_all_pairs(lo, up, R))
    width = float("nan") if infeasible else _offdiag_mean(lo, up)
    rows = []
    for eps in eps_all:
        d = decide_from_bounds(lo, up, g, eps, guarantee, infeasible=infeasible)
        rec = d["recommended_model"]
        treg = float(tr[rec]) if rec is not None else float("nan")
        cert = d["certified"]
        rows.append({**base, **(extra or {}), "method": method, "epsilon": eps,
                     "in_plan_grid": eps in eps_plan, "selected_model": rec, "certified": cert,
                     "status": d["status"], "G": d["worst_regret"], "guarantee_type": guarantee,
                     "true_regret": treg,
                     "wrong_certificate": wrong_certificate(cert, treg, eps) if cert else False,
                     "recommendation_exceeds_eps": (treg > eps + TOL) if rec is not None else np.nan,
                     "picked_best": (rec == best) if rec is not None else np.nan,
                     "interval_width": width, "coverage_all_pairs": cov,
                     "all_acceptable": bool(np.all(tr <= eps + TOL)),
                     "runtime_s": runtime})
    return rows


def baseline_rows(base, name, res, R, eps_all, eps_plan, runtime):
    tr = true_regret_vector(R)
    rec = res["recommended_model"]
    best = int(np.argmin(R))
    rows = []
    for eps in eps_all:
        treg = float(tr[rec]) if rec is not None else float("nan")
        rows.append({**base, "method": name, "epsilon": eps, "in_plan_grid": eps in eps_plan,
                     "selected_model": rec, "certified": False, "status": "recommendation_only",
                     "G": float("nan"), "guarantee_type": "recommendation_only", "true_regret": treg,
                     "wrong_certificate": False,
                     "recommendation_exceeds_eps": (treg > eps + TOL) if rec is not None else np.nan,
                     "picked_best": (rec == best) if rec is not None else np.nan,
                     "interval_width": float("nan"), "coverage_all_pairs": float("nan"),
                     "all_acceptable": bool(np.all(tr <= eps + TOL)), "runtime_s": runtime})
    return rows


def _timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


# ----------------------------------------------------------------------------- study: main / size / K
def _one_run(cfg, base, cohort, observed, pi, qmodel, R, with_baselines, eps_all, eps_plan, est_rows=None, seed=0):
    p = cohort.p
    y_in = np.where(observed, cohort.y, np.nan)
    rows = []
    (lo, up, g), t = _timed(c0_bounds, p, y_in, observed)
    rows += interval_rows(base, "C0", lo, up, g, "finite_exact", R, eps_all, eps_plan, t)
    (lo, up, g), t = _timed(c1_bounds, p, y_in, observed)
    rows += interval_rows(base, "C1", lo, up, g, "finite_exact", R, eps_all, eps_plan, t)
    if not with_baselines:
        return rows
    q = qmodel.predict_proba(cohort.X)[:, 1]
    rng = np.random.default_rng(np.random.SeedSequence([5, seed]))
    res_all = {}
    res, t = _timed(bl.complete_case, p, y_in, observed)
    res_all["complete_case"] = (res, t)
    res, t = _timed(bl.model_assisted, p, y_in, observed, q, None, cfg["main"]["n_fills"], rng)
    res_all["model_assisted"] = (res, t)
    pi_hat = bl.estimate_observation_prob(cohort.X, observed)
    res, t = _timed(bl.ipw, p, y_in, observed, pi, name="ipw_oracle")
    res_all["ipw_oracle"] = (res, t)
    res, t = _timed(bl.ipw, p, y_in, observed, pi_hat, name="ipw_estimated")
    res_all["ipw_estimated"] = (res, t)
    res, t = _timed(bl.ipw, p, y_in, observed, pi_hat, clip=0.05, name="clipped_ipw")
    res_all["clipped_ipw"] = (res, t)
    for name, (res, t) in res_all.items():
        rows += baseline_rows(base, name, res, R, eps_all, eps_plan, t)
        if est_rows is not None:
            for k, v in enumerate(res["estimates"]):
                est_rows.append({**base, "method": name, "model": k, "estimate": float(v),
                                 "truth_R": float(R[k]), "truth_regret": float(R[k] - R.min())})
            if name == "model_assisted":
                est_rows.append({**base, "method": "model_assisted_fill_gap", "model": -1,
                                 "estimate": res["convergence_gap"], "truth_R": np.nan, "truth_regret": np.nan})
    return rows


def study_main(cfg, out):
    m = cfg["main"]
    plan, allp = _eps_lists(cfg)
    for mech in m["mechanisms"]:
        for spread in m["spreads"]:
            for miss in cfg["missing_fractions"]:
                key = f"{mech}_{spread}_m{int(miss * 100)}"

                def fn():
                    rows, est = [], []
                    for ds in range(m["n_datasets"]):
                        coh = make_cohort(ds, m["n"], m["k"], spread)
                        past = make_cohort(10_000 + ds, m["past_n"], m["k"], spread)
                        qmodel = bl.fit_aux_q(past.X, past.y)
                        R = full_brier(coh.p, coh.y)
                        pi = observation_probabilities(coh, mech, miss)
                        for ms in range(m["n_masks"]):
                            seed = ds * 1000 + ms
                            obs = draw_mask(pi, seed)
                            base = dict(study="main", mechanism=mech, spread=spread, n=m["n"], k=m["k"],
                                        missing_target=miss, missing_fraction=float((~obs).mean()),
                                        dataset_seed=ds, mask_seed=ms)
                            rows += _one_run(cfg, base, coh, obs, pi, qmodel, R, True, allp, plan, est, seed)
                    pd.DataFrame(est).to_csv(chunk_path(out, "main_estimates", key), index=False,
                                             compression="gzip")
                    return pd.DataFrame(rows)
                run_chunk(out, "main", key, fn)


def study_size(cfg, out):
    s = cfg["size"]
    plan, allp = _eps_lists(cfg)
    for n in s["n"]:
        for miss in cfg["missing_fractions"]:
            key = f"n{n}_m{int(miss * 100)}"

            def fn():
                rows = []
                for ds in range(s["n_datasets"]):
                    coh = make_cohort(ds, n, s["k"], s["spread"])
                    R = full_brier(coh.p, coh.y)
                    pi = observation_probabilities(coh, s["mechanism"], miss)
                    for ms in range(s["n_masks"]):
                        obs = draw_mask(pi, ds * 1000 + ms)
                        base = dict(study="size", mechanism=s["mechanism"], spread=s["spread"], n=n, k=s["k"],
                                    missing_target=miss, missing_fraction=float((~obs).mean()),
                                    dataset_seed=ds, mask_seed=ms)
                        rows += _one_run(cfg, base, coh, obs, pi, None, R, False, allp, plan)
                return pd.DataFrame(rows)
            run_chunk(out, "size", key, fn)


def study_k(cfg, out):
    s = cfg["kstudy"]
    plan, allp = _eps_lists(cfg)
    for k in s["k"]:
        for spread in s["spreads"]:
            key = f"k{k}_{spread}"

            def fn():
                rows = []
                for ds in range(s["n_datasets"]):
                    coh = make_cohort(ds, s["n"], k, spread)
                    R = full_brier(coh.p, coh.y)
                    pi = observation_probabilities(coh, s["mechanism"], s["missing"])
                    for ms in range(s["n_masks"]):
                        obs = draw_mask(pi, ds * 1000 + ms)
                        base = dict(study="kstudy", mechanism=s["mechanism"], spread=spread, n=s["n"], k=k,
                                    missing_target=s["missing"], missing_fraction=float((~obs).mean()),
                                    dataset_seed=ds, mask_seed=ms)
                        rows += _one_run(cfg, base, coh, obs, pi, None, R, False, allp, plan)
                return pd.DataFrame(rows)
            run_chunk(out, "kstudy", key, fn)


# ----------------------------------------------------------------------------- study: C2
def _c2_row_set(base, label, variant, param, p, y_in, obs, gid, cb, R, allp, plan, holds, labels_used, half):
    (lo, up, g, st), t = _timed(c2_bounds, p, y_in, obs, gid, cb)
    return interval_rows(base, label, lo, up, g, "conditional_counts", R, allp, plan, t,
                         infeasible=(st != "ok"),
                         extra=dict(variant=variant, param=param, assumption_holds=holds,
                                    labels_used=labels_used, count_half_width=half))


def study_c2(cfg, out):
    c = cfg["c2"]
    plan, allp = _eps_lists(cfg)
    for mech in c["mechanisms"]:
        for spread in c["spreads"]:
            key = f"{mech}_{spread}"

            def fn():
                rows = []
                for ds in range(c["n_datasets"]):
                    coh = make_cohort(ds, c["n"], c["k"], spread)
                    R = full_brier(coh.p, coh.y)
                    pi = observation_probabilities(coh, mech, c["missing"])
                    for ms in range(c["n_masks"]):
                        seed = ds * 1000 + ms
                        obs = draw_mask(pi, seed)
                        base = dict(study="c2", mechanism=mech, spread=spread, n=c["n"], k=c["k"],
                                    missing_target=c["missing"], missing_fraction=float((~obs).mean()),
                                    dataset_seed=ds, mask_seed=ms)
                        p = coh.p
                        y_in = np.where(obs, coh.y, np.nan)
                        gid = np.zeros(len(obs), int)
                        hid = np.flatnonzero(~obs)
                        M = len(hid)
                        T = int(coh.y[hid].sum())
                        (lo, up, g), t = _timed(c1_bounds, p, y_in, obs)
                        rows += interval_rows(base, "C1", lo, up, g, "finite_exact", R, allp, plan, t,
                                              extra=dict(variant="none", param=0, assumption_holds=True,
                                                         labels_used=0, count_half_width=M / 2))
                        # (a) oracle interval around the TRUE count (code test / price of the information)
                        for h in c["oracle_half_widths"]:
                            A, B = max(0, T - h), min(M, T + h)
                            rows += _c2_row_set(base, "C2_oracle", "oracle", h, p, y_in, obs, gid,
                                                {0: (A, B)}, R, allp, plan, A <= T <= B, 0, (B - A) / 2)
                        # (a') wrong (shifted) premises -> may be internally feasible and undetected
                        for s in c["violation_shifts"]:
                            h = c["violation_half_width"]
                            A, B = T + s - h, T + s + h
                            holds = A <= T <= B
                            rows += _c2_row_set(base, "C2_shifted", "violated_oracle", s, p, y_in, obs, gid,
                                                {0: (A, B)}, R, allp, plan, holds, 0, h)
                        # (b) paid random audit (statistical premise) + same reveals for C1
                        rng = np.random.default_rng(np.random.SeedSequence([7, seed]))
                        order = rng.permutation(hid)
                        for ma in c["audit_sizes"]:
                            if ma >= M:
                                continue
                            idx = order[:ma]
                            s_pos = int(coh.y[idx].sum())
                            A, B = audit_count_interval(s_pos, ma, M, 1, cfg["alpha"])
                            holds = A <= T <= B
                            obs2 = obs.copy()
                            obs2[idx] = True
                            y2 = np.where(obs2, coh.y, np.nan)
                            cb = update_count_bounds_after_reveal({0: (A, B)}, 0, s_pos, M - ma)
                            # residual: update subtracts the SUM of positives, call once with s_pos
                            (lo, up, g, st), t = _timed(c2_bounds, p, y2, obs2, gid, cb)
                            rows += interval_rows(base, "C2_audit", lo, up, g, "conditional_counts", R, allp,
                                                  plan, t, infeasible=(st != "ok"),
                                                  extra=dict(variant="audit", param=ma, assumption_holds=holds,
                                                             labels_used=ma, count_half_width=(B - A) / 2))
                            (lo, up, g), t = _timed(c1_bounds, p, y2, obs2)
                            rows += interval_rows(base, "C1_after_audit", lo, up, g, "finite_exact", R, allp,
                                                  plan, t, extra=dict(variant="audit", param=ma,
                                                                      assumption_holds=True, labels_used=ma,
                                                                      count_half_width=(M - ma) / 2))
                        # (c) scenario Gamma from visible labels
                        n_vis = int(obs.sum())
                        pos_vis = int(coh.y[obs].sum())
                        for gam in c["gammas"]:
                            gv = float("inf") if gam == "inf" else float(gam)
                            for wd in c["widen"]:
                                A, B = scenario_count_bounds(pos_vis, n_vis, M, gv, wd)
                                rows += _c2_row_set(base, "C2_scenario", "scenario", f"G{gam}_w{wd}", p, y_in,
                                                    obs, gid, {0: (A, B)}, R, allp, plan, A <= T <= B, 0,
                                                    (B - A) / 2)
                return pd.DataFrame(rows)
            run_chunk(out, "c2", key, fn)


# ----------------------------------------------------------------------------- study: C2 stress
def study_c2stress(cfg, out):
    """Large premise violations (shifted count intervals, scenario-Gamma under strong MNAR).
    Wrong constraints are NOT announced by the data unless they become infeasible."""
    c = cfg["c2stress"]
    plan, allp = _eps_lists(cfg)
    for mech, strength in c["cases"]:
        for spread in c["spreads"]:
            key = f"{mech}{strength:g}_{spread}"

            def fn():
                rows = []
                for ds in range(c["n_datasets"]):
                    coh = make_cohort(ds, c["n"], c["k"], spread)
                    R = full_brier(coh.p, coh.y)
                    pi = observation_probabilities(coh, mech, c["missing"], strength)
                    for ms in range(c["n_masks"]):
                        obs = draw_mask(pi, ds * 1000 + ms)
                        base = dict(study="c2stress", mechanism=f"{mech}(strength={strength:g})", spread=spread,
                                    n=c["n"], k=c["k"], missing_target=c["missing"],
                                    missing_fraction=float((~obs).mean()), dataset_seed=ds, mask_seed=ms)
                        p = coh.p
                        y_in = np.where(obs, coh.y, np.nan)
                        gid = np.zeros(len(obs), int)
                        hid = np.flatnonzero(~obs)
                        M, T = len(hid), int(coh.y[hid].sum())
                        (lo, up, g), t = _timed(c1_bounds, p, y_in, obs)
                        rows += interval_rows(base, "C1", lo, up, g, "finite_exact", R, allp, plan, t,
                                              extra=dict(variant="none", param="0", assumption_holds=True,
                                                         labels_used=0, count_half_width=M / 2))
                        for s in c["shifts"]:
                            h = c["half_width"]
                            A, B = T + s - h, T + s + h
                            rows += _c2_row_set(base, "C2_shifted", "violated_oracle", str(s), p, y_in, obs, gid,
                                                {0: (A, B)}, R, allp, plan, A <= T <= B, 0, h)
                        n_vis, pos_vis = int(obs.sum()), int(coh.y[obs].sum())
                        for gam in c["gammas"]:
                            for wd in c["widen"]:
                                A, B = scenario_count_bounds(pos_vis, n_vis, M, float(gam), wd)
                                rows += _c2_row_set(base, "C2_scenario", "scenario", f"G{gam}_w{wd}", p, y_in, obs,
                                                    gid, {0: (A, B)}, R, allp, plan, A <= T <= B, 0, (B - A) / 2)
                return pd.DataFrame(rows)
            run_chunk(out, "c2stress", key, fn)


# ----------------------------------------------------------------------------- study: C3
def study_c3(cfg, out):
    c = cfg["c3"]
    plan, _ = _eps_lists(cfg)
    eps_all = plan + [e for e in c["epsilon_extra_c3"] if e not in plan]
    for n in c["n"]:
        for spread in c["spreads"]:
            for miss in c["missing"]:
                key = f"n{n}_{spread}_m{int(miss * 100)}"

                def fn():
                    coh = make_cohort(c["dataset_seed"], n, c["k"], spread)
                    R = full_brier(coh.p, coh.y)
                    tr = true_regret_vector(R)
                    D = R[:, None] - R[None, :]
                    pi = observation_probabilities(coh, "MCAR", miss)
                    obs0 = draw_mask(pi, c["mask_seed"])
                    y_in = np.where(obs0, coh.y, np.nan)
                    hid = np.flatnonzero(~obs0)
                    M = len(hid)
                    budgets = sorted({max(1, int(round(f * M))) for f in c["budget_fractions"]})
                    B = len(budgets)
                    (_, _, g_c1), _ = _timed(c1_bounds, coh.p, y_in, obs0)
                    # accumulators
                    ev = {m_: 0 for m_ in budgets}            # stat interval missed the truth (any pair)
                    fin = {m_: 0 for m_ in budgets}           # final (intersected) interval missed the truth
                    failed = {m_: 0 for m_ in budgets}        # intersection empty -> C1 fallback
                    acc = {}
                    sim_any = 0
                    t0 = time.perf_counter()
                    for rep in range(c["reps"]):
                        rng = np.random.default_rng(np.random.SeedSequence([11, rep]))
                        order = rng.permutation(hid)
                        any_bad = False
                        for mb in budgets:
                            idx = order[:mb]
                            r = c3_bounds(coh.p, coh.y, obs0, idx, cfg["alpha"], B)
                            lo_s, up_s = r["lower_stat"], r["upper_stat"]
                            bad = bool(np.any(D < lo_s - TOL) or np.any(D > up_s + TOL))
                            ev[mb] += bad
                            any_bad |= bad
                            failed[mb] += r["event_failed"]
                            fin[mb] += not coverage_all_pairs(r["lower"], r["upper"], R)
                            obs1 = obs0.copy()
                            obs1[idx] = True
                            for meth, (lo, up) in {"C3": (r["lower"], r["upper"]),
                                                   "C1_after_audit": (r["lower_c1"], r["upper_c1"])}.items():
                                g = up.max(axis=1)
                                width = _offdiag_mean(lo, up)
                                for eps in eps_all:
                                    d = decide_from_bounds(lo, up, g, eps, "x")
                                    a = acc.setdefault((mb, meth, eps),
                                                       dict(cert=0, wrong=0, w=0.0, G=0.0, gap=0.0, n=0))
                                    a["n"] += 1
                                    a["cert"] += d["certified"]
                                    rec = d["recommended_model"]
                                    a["wrong"] += bool(d["certified"] and tr[rec] > eps + TOL)
                                    a["w"] += width
                                    a["G"] += d["worst_regret"]
                        sim_any += any_bad
                    rt = (time.perf_counter() - t0) / c["reps"]
                    rows = []
                    for (mb, meth, eps), a in acc.items():
                        rows.append(dict(study="c3", n=n, spread=spread, missing_target=miss, M=M, N=n,
                                         budget=mb, budget_fraction=mb / M, n_budgets=B, method=meth, epsilon=eps,
                                         in_plan_grid=eps in plan, reps=a["n"], n_certified=a["cert"],
                                         n_wrong_certificate=a["wrong"], mean_G=a["G"] / a["n"],
                                         mean_interval_width=a["w"] / a["n"],
                                         radius=(M / n) * math.sqrt(2 * math.log(2 * c["k"] * (c["k"] - 1) * B
                                                                            / cfg["alpha"]) / mb),
                                         stat_violation_reps=ev[mb], final_violation_reps=fin[mb],
                                         event_failed_reps=failed[mb], any_budget_violation_reps=sim_any,
                                         G_c1_no_audit=float(g_c1.min()),
                                         true_regret_min_G_model=float(tr[int(np.argmin(g_c1))]),
                                         best_gap=float(np.sort(R)[1] - np.sort(R)[0]),
                                         runtime_per_rep_s=rt))
                    return pd.DataFrame(rows)
                run_chunk(out, "c3", key, fn)


# ----------------------------------------------------------------------------- orchestration
STUDIES = {"main": study_main, "size": study_size, "kstudy": study_k, "c2": study_c2,
           "c2stress": study_c2stress, "c3": study_c3}


def write_manifest(out, cfg, studies, quick):
    import scipy
    import sklearn
    man = {"config": cfg, "studies": studies, "quick": quick, "python": platform.python_version(),
           "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)


def cmd_run(args):
    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick(cfg)
    os.makedirs(args.out, exist_ok=True)
    studies = args.studies or list(STUDIES)
    write_manifest(args.out, cfg, studies, args.quick)
    for s in studies:
        STUDIES[s](cfg, args.out)
    for s in studies + ["main_estimates"]:
        df = combine_chunks(args.out, s)
        if df is not None:
            df.to_csv(os.path.join(args.out, f"{s}_rows.csv.gz"), index=False, compression="gzip")
    print("finished", studies)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="synthetic oracle studies")
    r.add_argument("--config", default="configs/h1_default.json")
    r.add_argument("--out", default="results/synthetic")
    r.add_argument("--studies", nargs="*", choices=list(STUDIES))
    r.add_argument("--quick", action="store_true", help="tiny smoke run; results are NOT meaningful")
    r.set_defaults(fn=cmd_run)
    from . import real_mode
    real_mode.add_parser(sub)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

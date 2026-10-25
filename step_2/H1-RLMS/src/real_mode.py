"""Real incomplete mode: runs LOCALLY on a prepared table; nothing is uploaded anywhere.

Input table (CSV / CSV.GZ; parquet if pyarrow is installed), one row per cohort member of ONE
temporal split (the split is yours: this code never re-splits or retrains):
    y          0/1 label, empty/NaN where genuinely unknown
    observed   optional 0/1 (default: y is not NaN)
    weight     optional non-negative cohort weight (default uniform; normalised over ALL rows)
    group      optional disjoint group id used for C2 count scenarios (default: one group)
    p_*        forecasts of the frozen candidate models (or pass --model-cols)
    q          optional auxiliary P(y=1|X) from a model trained on the PAST (model-assisted baseline)
    pi         optional probability of initial observation (IPW with supplied probabilities)
    --feature-cols  columns used to ESTIMATE pi by logistic regression (IPW_estimated)

Outputs (never a true_regret: unknown labels are not imputed):
    real_decisions.csv       method x epsilon: recommended model, G, certified, status, guarantee type
    real_pairwise_bounds.csv C0/C1 lower & upper pairwise differences
    real_c2_sensitivity.csv  Gamma x widen -> G, recommended model / refusal  (scenario_shift; NOT a guarantee
                             about the real hidden labels, only conditional on the stated assumption)
    real_baselines.csv       recommendation-only estimates
    real_artificial_masks.csv (optional, --artificial) hide MORE of the KNOWN labels (MCAR 10/20/40 %),
                             regret is computed on the observed subcohort only (column name says so)
Re-running resumes: finished artificial-mask chunks are skipped.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import baselines as bl
from .bounds import (TOL, c0_bounds, c1_bounds, c2_bounds, scenario_count_bounds)
from .certification import decide_from_bounds
from .metrics import full_brier, true_regret_vector


def read_table(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def add_parser(sub):
    r = sub.add_parser("real", help="real incomplete mode (run locally on your prepared table)")
    r.add_argument("--input", required=True)
    r.add_argument("--out", default="results/real_run")
    r.add_argument("--config", default="configs/h1_default.json")
    r.add_argument("--model-cols", default=None, help="comma separated forecast columns (default: p_*)")
    r.add_argument("--feature-cols", default=None, help="comma separated features to estimate pi (IPW)")
    r.add_argument("--artificial", action="store_true", help="also run the artificial-mask grid")
    r.add_argument("--n-seeds", type=int, default=30, help="mask seeds 0..n-1 (plan: 30 pilot, 100 confirm)")
    r.set_defaults(fn=cmd_real)


def _prepare(df, model_cols):
    cols = model_cols.split(",") if model_cols else [c for c in df.columns if c.startswith("p_")]
    if len(cols) < 2:
        raise SystemExit("need at least two forecast columns (p_*)")
    p = df[cols].to_numpy(float)
    y = df["y"].to_numpy(float)
    obs = df["observed"].to_numpy(bool) if "observed" in df.columns else ~np.isnan(y)
    w = df["weight"].to_numpy(float) if "weight" in df.columns else None
    return cols, p, y, obs, w


def cmd_real(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    df = read_table(args.input)
    cols, p, y, obs, w = _prepare(df, args.model_cols)
    n, k = p.shape
    eps_all = list(cfg["epsilon_grid"]) + list(cfg.get("epsilon_extra", []))
    y_in = np.where(obs, y, np.nan)
    print(f"N={n} K={k} hidden={int((~obs).sum())} ({(~obs).mean():.1%}); models={cols}")

    dec_rows, pair_rows = [], []
    for name, fn in (("C0", c0_bounds), ("C1", c1_bounds)):
        lo, up, g = fn(p, y_in, obs, w)
        for a in range(k):
            for b in range(k):
                pair_rows.append(dict(method=name, model_a=cols[a], model_b=cols[b],
                                      lower=lo[a, b], upper=up[a, b]))
        for eps in eps_all:
            d = decide_from_bounds(lo, up, g, eps, "finite_exact")
            dec_rows.append(dict(method=name, epsilon=eps, recommended=cols[d["recommended_model"]],
                                 G=d["worst_regret"], certified=d["certified"], status=d["status"],
                                 guarantee_type="finite_exact", true_regret=np.nan,
                                 missing_fraction=float((~obs).mean())))
    pd.DataFrame(pair_rows).to_csv(os.path.join(args.out, "real_pairwise_bounds.csv"), index=False)

    # C2 scenario sensitivity (single group or the supplied disjoint groups; NOT a guarantee)
    gid = df["group"].to_numpy() if "group" in df.columns else np.zeros(n, int)
    c2_rows = []
    for gam in cfg["c2"]["gammas"]:
        gv = float("inf") if gam == "inf" else float(gam)
        for wd in cfg["c2"]["widen"]:
            cb = {}
            for g_ in np.unique(gid):
                hid_g = (~obs) & (gid == g_)
                vis_g = obs & (gid == g_)
                cb[g_] = scenario_count_bounds(int(np.nansum(y[vis_g])), int(vis_g.sum()), int(hid_g.sum()), gv, wd)
            lo, up, g, st = c2_bounds(p, y_in, obs, gid, cb, w)
            for eps in cfg["epsilon_grid"]:
                d = decide_from_bounds(lo, up, g, eps, "scenario_shift", infeasible=(st != "ok"))
                rec = d["recommended_model"]
                c2_rows.append(dict(gamma=gam, widen=wd, epsilon=eps, status=d["status"],
                                    recommended=cols[rec] if rec is not None else None,
                                    G=d["worst_regret"], certified=d["certified"],
                                    guarantee_type="scenario_shift"))
    pd.DataFrame(c2_rows).to_csv(os.path.join(args.out, "real_c2_sensitivity.csv"), index=False)

    base_rows = []
    res = bl.complete_case(p, y_in, obs, w)
    base_rows.append(dict(method="complete_case", **{f"est_{c}": v for c, v in zip(cols, res["estimates"])},
                          recommended=cols[res["recommended_model"]]))
    if "q" in df.columns:
        res = bl.model_assisted(p, y_in, obs, df["q"].to_numpy(float), w, cfg["main"]["n_fills"],
                                np.random.default_rng(0))
        base_rows.append(dict(method="model_assisted", **{f"est_{c}": v for c, v in zip(cols, res["estimates"])},
                              recommended=cols[res["recommended_model"]], fill_gap=res["convergence_gap"]))
    pis = {}
    if "pi" in df.columns:
        pis["ipw_supplied_pi"] = (df["pi"].to_numpy(float), None)
    if args.feature_cols:
        X = df[args.feature_cols.split(",")].to_numpy(float)
        ph = bl.estimate_observation_prob(X, obs)
        pis["ipw_estimated"] = (ph, None)
        pis["clipped_ipw"] = (ph, 0.05)
    for name, (pi, clip) in pis.items():
        res = bl.ipw(p, y_in, obs, pi, w, clip=clip, name=name)
        base_rows.append(dict(method=name, **{f"est_{c}": v for c, v in zip(cols, res["estimates"])},
                              recommended=cols[res["recommended_model"]]))
    pd.DataFrame(base_rows).to_csv(os.path.join(args.out, "real_baselines.csv"), index=False)
    pd.DataFrame(dec_rows).to_csv(os.path.join(args.out, "real_decisions.csv"), index=False)

    if args.artificial:
        run_artificial(args, cfg, cols, p, y, obs, w, eps_all)
    print("written to", args.out)


def run_artificial(args, cfg, cols, p, y, obs, w, eps_all):
    """Hide extra KNOWN labels (MCAR).  Reference = full Brier on the observed subcohort."""
    idx = np.flatnonzero(obs)
    ps, ys = p[idx], y[idx]
    ws = None if w is None else w[idx]
    R = full_brier(ps, ys, ws)
    tr = true_regret_vector(R)
    for m in cfg["missing_fractions"]:
        path = os.path.join(args.out, "chunks", f"artificial_m{int(m * 100)}.csv")
        if os.path.exists(path):
            print("[skip]", path)
            continue
        rows = []
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(np.random.SeedSequence([3, seed, int(m * 100)]))
            o2 = rng.random(len(idx)) >= m
            yi = np.where(o2, ys, np.nan)
            for name, fn in (("C0", c0_bounds), ("C1", c1_bounds)):
                lo, up, g = fn(ps, yi, o2, ws)
                for eps in eps_all:
                    d = decide_from_bounds(lo, up, g, eps, "finite_exact")
                    rec = d["recommended_model"]
                    rows.append(dict(method=name, missing_fraction=m, mask_seed=seed, epsilon=eps,
                                     recommended=cols[rec], G=d["worst_regret"], certified=d["certified"],
                                     regret_on_observed_subcohort=float(tr[rec]),
                                     wrong_certificate=bool(d["certified"] and tr[rec] > eps + TOL)))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    parts = [pd.read_csv(os.path.join(args.out, "chunks", f)) for f in sorted(os.listdir(os.path.join(args.out, "chunks")))
             if f.startswith("artificial_")]
    pd.concat(parts).to_csv(os.path.join(args.out, "real_artificial_masks.csv"), index=False)

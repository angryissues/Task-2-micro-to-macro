"""Generate report/h1_tables.md from results/synthetic/*.csv (so no number is transcribed by hand)."""
import argparse
import os

import numpy as np
import pandas as pd

from .metrics import clopper_pearson


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("NA" if (isinstance(v, float) and np.isnan(v)) else (floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/synthetic")
    ap.add_argument("--out", default="report/h1_tables.md")
    a = ap.parse_args()
    R = a.res
    T = lambda n: pd.read_csv(os.path.join(R, f"table_{n}.csv"))
    parts = []

    def add(title, df, fmt="{:.3f}", note=None):
        parts.append(f"### {title}\n")
        if note:
            parts.append(note + "\n")
        parts.append(md(df, fmt) + "\n")

    ms = T("main_summary")
    # T1: certification by cell, MCAR, plan eps grid, C0 vs C1
    d = ms[ms.method.isin(["C0", "C1"]) & ms.in_plan_grid]
    for mech in ("MCAR", "MAR", "MNAR"):
        p = d[d.mechanism == mech].pivot_table(index=["spread", "missing_target"], columns=["method", "epsilon"], values="certification_rate")
        p.columns = [f"{m} ε={e:g}" for m, e in p.columns]
        add(f"T1-{mech}. Certification rate, {mech}, N=1000, K=3 (100 runs per cell)", p.reset_index())

    # T2: wrong certificates by method / guarantee
    main = pd.read_csv(os.path.join(R, "main_rows.csv.gz"))
    rows = []
    for m, g in main.groupby("method"):
        k, n = int(g.wrong_certificate.sum()), int(g.certified.sum())
        rows.append(dict(study="main", method=m, guarantee=g.guarantee_type.iloc[0], decisions=len(g), certificates=n,
                         wrong=k, wrong_over_certificates=("NA" if n == 0 else f"{k / n:.4f}"),
                         cp95_upper_of_wrong_rate=(clopper_pearson(k, n)[1] if n else np.nan)))
    for study, f in (("c2", "c2_rows.csv.gz"), ("c2stress", "c2stress_rows.csv.gz")):
        c = pd.read_csv(os.path.join(R, f))
        for (m, h), g in c.groupby(["method", "assumption_holds"]):
            k, n = int(g.wrong_certificate.sum()), int(g.certified.sum())
            rows.append(dict(study=f"{study} (premise {'holds' if h else 'FALSE'})", method=m,
                             guarantee=g.guarantee_type.iloc[0], decisions=len(g), certificates=n, wrong=k,
                             wrong_over_certificates=("NA" if n == 0 else f"{k / n:.4f}"),
                             cp95_upper_of_wrong_rate=(clopper_pearson(k, n)[1] if n else np.nan)))
    c3 = T("c3_summary")
    for m, g in c3.groupby("method"):
        k, n = int(g.n_wrong_certificate.sum()), int(g.n_certified.sum())
        rows.append(dict(study="c3 (random audit)", method=m, guarantee="statistical_random_audit" if m == "C3" else "finite_exact",
                         decisions=int(g.reps.sum()), certificates=n, wrong=k, wrong_over_certificates="NA" if n == 0 else f"{k / n:.5f}",
                         cp95_upper_of_wrong_rate=clopper_pearson(k, n)[1] if n else np.nan))
    add("T2. Wrong certificates by method and guarantee type (all studies)", pd.DataFrame(rows), "{:.5f}",
        "A 'decision' = one run x one epsilon. Baselines never certify, so they cannot issue a wrong certificate.")

    # T3: C1/C0 pooled by eps (+ non trivial)
    dd = main[main.method.isin(["C0", "C1"]) & main.in_plan_grid]
    rows = []
    for (m, e), g in dd.groupby(["method", "epsilon"]):
        nt = g[~g.all_acceptable]
        rows.append(dict(method=m, epsilon=e, cert_rate_all=g.certified.mean(), share_all_candidates_acceptable=g.all_acceptable.mean(),
                         cert_rate_when_some_candidate_unacceptable=nt.certified.mean(), runs=len(g)))
    add("T3. Pooled certification rate vs epsilon (main grid, 27 cells)", pd.DataFrame(rows), "{:.4f}")

    # T4: C2 oracle vs C1 / audit
    c2 = T("c2_summary")
    x = c2[(c2.mechanism == "MCAR") & (c2.epsilon == 0.005) & (c2.method.isin(["C1", "C2_oracle"]))]
    x = x[["spread", "method", "param", "count_half_width", "certification_rate", "mean_width", "n_wrong"]].copy()
    x["_o"] = pd.to_numeric(x.param, errors="coerce")
    add("T4. C2 with an ORACLE count interval vs C1 (MCAR 20% hidden, eps=0.005)", x.sort_values(["spread", "method", "_o"]).drop(columns="_o"))
    y = c2[(c2.mechanism == "MCAR") & (c2.epsilon == 0.005) & (c2.method.isin(["C1_after_audit", "C2_audit"]))]
    y = y[["spread", "method", "labels_used", "certification_rate", "mean_width", "assumption_holds_rate", "n_wrong"]]
    add("T5. Paid random audit: C2 (statistical count interval) vs C1 with the same labels revealed (MCAR 20%, eps=0.005)",
        y.sort_values(["spread", "labels_used", "method"]))

    # T6: C2 stress
    w = T("c2stress_summary")
    w = w[w.in_plan_grid & w.method.isin(["C2_shifted", "C2_scenario"])]
    w = w.groupby(["mechanism", "spread", "method", "param"]).agg(runs=("n_runs", "max"), certificates=("n_certified", "sum"),
                                                                 wrong=("n_wrong", "sum"), premise_holds_rate=("assumption_holds_rate", "mean"),
                                                                 infeasible_rate=("infeasible_rate", "mean")).reset_index()
    w["wrong_over_certificates"] = np.where(w.certificates > 0, w.wrong / w.certificates.clip(lower=1), np.nan)
    add("T6. C2 under large premise violations, plan eps grid pooled (40% hidden)", w, note="`param`: shift of the count interval (people) or Gamma_widen scenario.")

    # T7: C3
    cv = T("c3_coverage")
    g = cv.groupby("spread").agg(configs=("n", "size"), reps_per_budget=("reps", "first"), min_radius=("radius", "min"), max_radius=("radius", "max"),
                                 best_gap=("best_gap", "mean"), stat_violations=("stat_violation_reps", "sum"),
                                 any_budget_violations=("any_budget_violation_reps", "sum"), final_violations=("final_violation_reps", "sum"),
                                 event_failed_detected=("event_failed_reps", "sum")).reset_index()
    add("T7. C3 statistical bounds: violations of the nominal 95% (2000 audit repetitions per configuration)", g, "{:.4f}")
    s = T("c3_summary")
    a_ = s[s.method == "C3"].set_index(["n", "spread", "missing_target", "budget", "epsilon"])
    b_ = s[s.method == "C1_after_audit"].set_index(["n", "spread", "missing_target", "budget", "epsilon"])
    j = a_[["budget_fraction", "radius", "mean_interval_width", "certification_rate"]].join(b_[["mean_interval_width", "certification_rate"]], rsuffix="_c1after")
    j = j.reset_index()
    t = j[(j.certification_rate - j.certification_rate_c1after).abs() > 1e-9][["n", "spread", "missing_target", "budget_fraction", "epsilon", "certification_rate", "certification_rate_c1after"]]
    add("T8. The only (configuration, epsilon) cells where C3 certifies more often than C1 with the same revealed labels", t)
    nw = j[(j.epsilon == 0.0025)]
    add("T9. C3 vs C1-after-audit: share of (config, budget) cells where C3 intervals are strictly narrower",
        pd.DataFrame([dict(cells=len(nw), c3_narrower=int((nw.mean_interval_width < nw.mean_interval_width_c1after - 1e-9).sum()),
                           max_relative_narrowing=float(((nw.mean_interval_width_c1after - nw.mean_interval_width) / nw.mean_interval_width_c1after).max()))]))

    # T10 baselines
    base = ms[(ms.epsilon == 0.0025) & (ms.missing_target == 0.2) & (ms.spread == "moderate")][["mechanism", "method", "picked_best", "mean_true_regret_rec", "rec_exceeds_eps"]]
    add("T10. Recommendations (not certificates), moderate spread, 20% hidden, eps=0.0025", base.sort_values(["mechanism", "method"]), "{:.4f}")
    bb = T("baseline_bias")
    bb = bb[bb.missing_target == 0.2].groupby(["mechanism", "method"])[["mean_bias_loss", "sd_loss_estimate", "mean_abs_err", "mean_bias_regret"]].mean().reset_index()
    add("T11. Bias / spread of baseline LOSS estimates (20% hidden, averaged over 3 spreads x 3 models)", bb, "{:.5f}",
        "bias = mean(estimate - true full-cohort Brier); sd across datasets/masks.")
    add("T12. Model-assisted: 30 Bernoulli fills vs the expected-loss estimate", T("model_assisted_fill_convergence"), "{:.5f}")

    # T13 size, K
    sz = T("size_summary")
    sz = sz[(sz.epsilon == 0.0025) & (sz.method.isin(["C0", "C1"]))].pivot_table(index=["n"], columns=["method", "missing_target"], values="mean_width")
    sz.columns = [f"{m} width, {int(e*100)}% hidden" for m, e in sz.columns]
    add("T13. Mean pairwise interval width vs cohort size N (MCAR, moderate spread)", sz.reset_index(), "{:.4f}")
    szr = T("size_summary")
    szr = szr[(szr.epsilon == 0.0025) & (szr.method.isin(["C1"]))].pivot_table(index=["n"], columns=["missing_target"], values="certification_rate")
    szr.columns = [f"C1 cert rate, {int(e*100)}% hidden" for e in szr.columns]
    add("T14. C1 certification rate vs N (eps=0.0025, MCAR, moderate)", szr.reset_index())
    ks = T("kstudy_summary")
    ks = ks[(ks.epsilon == 0.0025) & ks.method.isin(["C0", "C1"])].pivot_table(index=["k", "spread"], columns="method", values=["certification_rate", "mean_width"])
    ks.columns = [f"{a} {b}" for a, b in ks.columns]
    add("T15. Number of models K (MCAR 20% hidden, N=1000, eps=0.0025)", ks.reset_index(), "{:.4f}")
    rt = sz  # placeholder to keep names
    rts = T("size_summary")
    rts = rts[(rts.epsilon == 0.0025) & (rts.missing_target == 0.2)].pivot_table(index="n", columns="method", values="mean_runtime_s") * 1000
    add("T16. Runtime of the bounds computation, ms per call (K=3)", rts.reset_index(), "{:.3f}")
    mrt = main[(main.epsilon == 0.0025)].groupby("method").runtime_s.mean().reset_index()
    mrt["runtime_ms"] = mrt.runtime_s * 1000
    add("T17. Mean runtime per run incl. all baselines (N=1000, K=3), ms", mrt[["method", "runtime_ms"]], "{:.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# Generated tables (do not edit; produced by src/report_tables.py)\n\n" + "\n".join(parts))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

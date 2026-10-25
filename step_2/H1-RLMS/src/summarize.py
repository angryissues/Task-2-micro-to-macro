"""Aggregate raw rows -> summary tables + figures.   python -m src.summarize --out results/synthetic"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .metrics import clopper_pearson

COL = {"C0": "#2a78d6", "C1": "#eb6834", "C2": "#1baf7a", "C3": "#e87ba4", "ink": "#0b0b0b",
       "muted": "#52514e", "grid": "#e4e3df", "base": "#8a8985"}
BASE_METHODS = ["complete_case", "model_assisted", "ipw_oracle", "ipw_estimated", "clipped_ipw"]


# ----------------------------------------------------------------------------- tables
def rate_ci(k, n):
    lo, hi = clopper_pearson(int(k), int(n))
    return lo, hi


def summarize_rows(df, keys):
    g = df.groupby(keys, dropna=False)
    out = g.agg(n_runs=("certified", "size"), n_certified=("certified", "sum"),
                n_wrong=("wrong_certificate", "sum"), mean_G=("G", "mean"),
                mean_width=("interval_width", "mean"), mean_true_regret_rec=("true_regret", "mean"),
                picked_best=("picked_best", "mean"),
                rec_exceeds_eps=("recommendation_exceeds_eps", "mean"),
                coverage_all_pairs=("coverage_all_pairs", "mean"),
                share_all_acceptable=("all_acceptable", "mean"),
                mean_runtime_s=("runtime_s", "mean"), mean_missing=("missing_fraction", "mean")).reset_index()
    out["certification_rate"] = out.n_certified / out.n_runs
    out["abstention_rate"] = 1 - out.certification_rate
    out["wrong_rate_all_runs"] = out.n_wrong / out.n_runs
    out["wrong_rate_given_certificate"] = np.where(out.n_certified > 0, out.n_wrong / out.n_certified.clip(lower=1), np.nan)
    ci = [rate_ci(a, b) for a, b in zip(out.n_wrong, out.n_runs)]
    out["wrong_rate_all_cp95_high"] = [c[1] for c in ci]
    return out


def informative_split(df, keys):
    """Certification rate separately where NOT all candidates are already acceptable (non-trivial cases)."""
    d = df[~df.all_acceptable]
    s = d.groupby(keys).agg(n_nontrivial=("certified", "size"), cert_nontrivial=("certified", "mean")).reset_index()
    return s


def build_tables(out):
    t = {}
    main = pd.read_csv(os.path.join(out, "main_rows.csv.gz"))
    keys = ["mechanism", "spread", "missing_target", "method", "epsilon", "in_plan_grid"]
    t["main_summary"] = summarize_rows(main, keys).merge(informative_split(main, keys), on=keys, how="left")
    est = pd.read_csv(os.path.join(out, "main_estimates_rows.csv.gz"))
    e = est[est.model >= 0].copy()
    e["err"] = e.estimate - e.truth_R
    e["regret_hat"] = e.estimate - e.groupby(["mechanism", "spread", "missing_target", "dataset_seed", "mask_seed", "method"]).estimate.transform("min")
    e["regret_err"] = e.regret_hat - e.truth_regret
    t["baseline_bias"] = e.groupby(["mechanism", "spread", "missing_target", "method"]).agg(
        mean_bias_loss=("err", "mean"), sd_loss_estimate=("err", "std"),
        mean_abs_err=("err", lambda x: np.abs(x).mean()), mean_bias_regret=("regret_err", "mean")).reset_index()
    gap = est[est.method == "model_assisted_fill_gap"]
    t["model_assisted_fill_convergence"] = pd.DataFrame({"mean_gap": [gap.estimate.mean()], "max_gap": [gap.estimate.max()],
                                                         "n": [len(gap)]})
    for study in ("size", "kstudy"):
        d = pd.read_csv(os.path.join(out, f"{study}_rows.csv.gz"))
        ks = ["n", "k", "mechanism", "spread", "missing_target", "method", "epsilon", "in_plan_grid"]
        t[f"{study}_summary"] = summarize_rows(d, ks)
    c2 = pd.read_csv(os.path.join(out, "c2_rows.csv.gz"))
    c2["variant"] = c2["variant"].astype(str)
    c2["param"] = c2["param"].astype(str)
    c2["infeasible"] = c2.status == "assumption_infeasible"
    k2 = ["mechanism", "spread", "method", "variant", "param", "epsilon", "in_plan_grid"]
    s2 = summarize_rows(c2, k2)
    extra = c2.groupby(k2).agg(assumption_holds_rate=("assumption_holds", "mean"), infeasible_rate=("infeasible", "mean"),
                               labels_used=("labels_used", "mean"), count_half_width=("count_half_width", "mean")).reset_index()
    t["c2_summary"] = s2.merge(extra, on=k2)
    sp = c2[c2.certified].groupby(["mechanism", "method", "variant", "assumption_holds", "epsilon"]).agg(
        n_certified=("certified", "size"), n_wrong=("wrong_certificate", "sum")).reset_index()
    sp["wrong_rate_given_certificate"] = sp.n_wrong / sp.n_certified
    t["c2_wrong_by_premise"] = sp
    cs = pd.read_csv(os.path.join(out, "c2stress_rows.csv.gz"))
    cs["param"] = cs["param"].astype(str)
    cs["infeasible"] = cs.status == "assumption_infeasible"
    ks = ["mechanism", "spread", "method", "variant", "param", "epsilon", "in_plan_grid"]
    s3 = summarize_rows(cs, ks).merge(
        cs.groupby(ks).agg(assumption_holds_rate=("assumption_holds", "mean"), infeasible_rate=("infeasible", "mean")).reset_index(), on=ks)
    t["c2stress_summary"] = s3
    sp2 = cs[cs.certified].groupby(["mechanism", "spread", "method", "variant", "assumption_holds"]).agg(
        n_certified=("certified", "size"), n_wrong=("wrong_certificate", "sum")).reset_index()
    sp2["wrong_rate_given_certificate"] = sp2.n_wrong / sp2.n_certified
    t["c2stress_wrong_by_premise"] = sp2
    c3 = pd.read_csv(os.path.join(out, "c3_rows.csv.gz"))
    c3["certification_rate"] = c3.n_certified / c3.reps
    c3["wrong_rate_all_runs"] = c3.n_wrong_certificate / c3.reps
    c3["wrong_cp95_high"] = [rate_ci(a, b)[1] for a, b in zip(c3.n_wrong_certificate, c3.reps)]
    t["c3_summary"] = c3
    cov = c3[(c3.method == "C3") & (c3.epsilon == c3.epsilon.min())].copy()
    cov["stat_violation_rate"] = cov.stat_violation_reps / cov.reps
    cov["stat_violation_cp95_high"] = [rate_ci(a, b)[1] for a, b in zip(cov.stat_violation_reps, cov.reps)]
    cov["final_violation_rate"] = cov.final_violation_reps / cov.reps
    cov["any_budget_violation_rate"] = cov.any_budget_violation_reps / cov.reps
    cov["any_budget_cp95_high"] = [rate_ci(a, b)[1] for a, b in zip(cov.any_budget_violation_reps, cov.reps)]
    t["c3_coverage"] = cov[["n", "spread", "missing_target", "M", "budget", "budget_fraction", "n_budgets", "reps", "radius",
                            "best_gap", "stat_violation_reps", "stat_violation_rate", "stat_violation_cp95_high",
                            "any_budget_violation_reps", "any_budget_violation_rate", "any_budget_cp95_high",
                            "final_violation_reps", "final_violation_rate", "event_failed_reps"]]
    for k, v in t.items():
        v.to_csv(os.path.join(out, f"table_{k}.csv"), index=False)
    return t


# ----------------------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 9, "axes.edgecolor": COL["muted"], "axes.labelcolor": COL["ink"],
                         "xtick.color": COL["muted"], "ytick.color": COL["muted"], "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": COL["grid"],
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 130,
                         "axes.titlesize": 9.5, "axes.titleweight": "normal", "legend.frameon": False})


def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name), bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


def build_figures(out, figdir, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    ms = t["main_summary"]
    spreads = ["close", "moderate", "distinct"]
    mk = {"C0": "o", "C1": "s"}

    # 1: abstention vs missing fraction, eps = 0.0025 (MCAR) + C1 under MAR/MNAR dashed
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, sp in zip(axs, spreads):
        for m_ in ("C0", "C1"):
            d = ms[(ms.method == m_) & (ms.spread == sp) & (ms.epsilon == 0.0025)]
            for mech, ls in (("MCAR", "-"), ("MNAR", ":")):
                dd = d[d.mechanism == mech].sort_values("missing_target")
                ax.plot(dd.missing_target * 100, dd.abstention_rate, ls, marker=mk[m_], color=COL[m_], lw=1.6,
                        label=f"{m_} {mech}")
        ax.set_title(f"spread = {sp}")
        ax.set_xlabel("hidden labels, %")
        ax.set_xticks([10, 20, 40])
    axs[0].set_ylabel("abstention rate (ε = 0.0025)")
    axs[0].legend(fontsize=7, loc="lower left")
    _save(fig, figdir, "fig1_abstention_vs_missing.png")

    # 2: certification rate vs epsilon (MCAR, m=20%)
    eps_all = sorted(ms.epsilon.unique())
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, sp in zip(axs, spreads):
        for m_ in ("C0", "C1"):
            d = ms[(ms.method == m_) & (ms.spread == sp) & (ms.mechanism == "MCAR") & (ms.missing_target == 0.2)].sort_values("epsilon")
            ax.plot(range(len(d)), d.certification_rate, marker=mk[m_], color=COL[m_], lw=1.6, label=m_)
            if m_ == "C1":
                ax.plot(range(len(d)), d.share_all_acceptable, "--", color=COL["base"], lw=1.2,
                        label="all candidates already ≤ ε")
        ax.set_xticks(range(len(eps_all)))
        ax.set_xticklabels([f"{e:g}" for e in eps_all], rotation=60, fontsize=7)
        ax.set_title(f"spread = {sp}")
        ax.set_xlabel("ε (last 3 are outside the plan grid)")
    axs[0].set_ylabel("certification rate (MCAR, 20% hidden)")
    axs[0].legend(fontsize=7)
    _save(fig, figdir, "fig2_certification_vs_epsilon.png")

    # 3: C1 width vs N and missing (size study)
    sz = t["size_summary"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for m_, c in zip((0.1, 0.2, 0.4), (COL["C0"], COL["C1"], COL["C2"])):
        d = sz[(sz.method == "C1") & (sz.missing_target == m_) & (sz.epsilon == 0.0025)].sort_values("n")
        ax.plot(d.n, d.mean_width, marker="o", color=c, lw=1.6)
        ax.annotate(f"{int(m_*100)}% hidden", (d.n.iloc[-1], d.mean_width.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=COL["ink"], va="center")
    ax.axhline(0.0025, color=COL["base"], ls="--", lw=1)
    ax.text(sz.n.min(), 0.0025, " ε = 0.0025", fontsize=7, va="bottom", color=COL["muted"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(right=sz.n.max() * 4)
    ax.set_xlabel("cohort size N"); ax.set_ylabel("mean width of C1 pairwise intervals")
    ax.set_title("C1 width does not shrink with N: it scales with the hidden share")
    _save(fig, figdir, "fig3_c1_width_vs_n.png")

    # 4: C0 vs C1 same information
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.1))
    a = ms[(ms.method == "C0")].set_index(["mechanism", "spread", "missing_target", "epsilon"])
    b = ms[(ms.method == "C1")].set_index(["mechanism", "spread", "missing_target", "epsilon"])
    j = a.join(b, lsuffix="_c0", rsuffix="_c1")
    j1 = j.xs(0.0025, level="epsilon")
    axs[0].scatter(j1.mean_width_c0, j1.mean_width_c1, color=COL["C1"], s=18)
    lim = [0, float(max(j1.mean_width_c0.max(), j1.mean_width_c1.max())) * 1.05]
    axs[0].plot(lim, lim, color=COL["base"], lw=1)
    axs[0].set_xlabel("C0 mean interval width"); axs[0].set_ylabel("C1 mean interval width")
    axs[0].set_title("same information, C1 never wider than C0")
    j2 = j.xs(0.01, level="epsilon")
    axs[1].scatter(j2.certification_rate_c0, j2.certification_rate_c1, color=COL["C1"], s=18)
    axs[1].plot([0, 1], [0, 1], color=COL["base"], lw=1)
    axs[1].set_xlabel("C0 certification rate (ε = 0.01)"); axs[1].set_ylabel("C1 certification rate")
    axs[1].set_title("one point = one scenario cell")
    _save(fig, figdir, "fig4_c0_vs_c1.png")

    # 5: C1 vs C2
    c2 = t["c2_summary"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.1))
    ax = axs[0]
    e = 0.005
    for sp, ls, mkr in (("moderate", "-", "o"), ("distinct", "--", "s")):
        q = c2[(c2.spread == sp) & (c2.mechanism == "MCAR") & (c2.epsilon == e)]
        c1r = q[q.method == "C1"].certification_rate.iloc[0]
        d = q[q.method == "C2_oracle"].copy()
        d["hw"] = d.param.astype(float)
        d = d.sort_values("hw")
        ax.plot(d.hw, d.certification_rate, ls, marker=mkr, color=COL["C2"], lw=1.6, label=f"C2 oracle count, {sp}")
        ax.axhline(c1r, color=COL["C1"], ls=ls, lw=1.2, label=f"C1 (no count info), {sp}")
    ax.set_xlabel("half-width of TRUE-centred count interval, people")
    ax.set_ylabel("certification rate (ε = 0.005, 20% hidden)")
    ax.set_title("C2 ≥ C1; the gain fades as the count interval widens")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax = axs[1]
    st = t["c2stress_summary"]
    st = st[(st.method == "C2_shifted") & st.in_plan_grid]
    for (mech, sp), c, ls in (((("MCAR(strength=1)", "moderate")), COL["C2"], "-"), ((("MCAR(strength=1)", "distinct")), COL["C2"], "--"),
                              ((("MNAR(strength=3)", "moderate")), COL["C3"], "-"), ((("MNAR(strength=3)", "distinct")), COL["C3"], "--")):
        d = st[(st.mechanism == mech) & (st.spread == sp)].groupby("param").agg(c=("n_certified", "sum"), w=("n_wrong", "sum")).reset_index()
        d["shift"] = d.param.astype(float)
        d = d.sort_values("shift")
        d["rate"] = np.where(d.c > 0, d.w / d.c.clip(lower=1), np.nan)
        ax.plot(d["shift"], d.rate, ls, marker=("o" if sp == "moderate" else "s"), color=c, lw=1.4, label=f"{mech.split('(')[0]} {sp}")
    ax.set_xlabel("shift of assumed count interval (people; M≈400)")
    ax.set_ylabel("wrong certificates / certificates (plan ε grid)")
    ax.set_title("C2 under false premises (gap = no certificate issued)")
    ax.legend(fontsize=7)
    _save(fig, figdir, "fig5_c1_vs_c2.png")

    # 6: C3 vs budget
    c3 = t["c3_summary"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.1))
    sel = (c3.n == 1000) & (c3.spread == "distinct") & (c3.missing_target == 0.2)
    for meth, c, mkr in (("C3", COL["C3"], "o"), ("C1_after_audit", COL["C1"], "s")):
        for eps, ls in ((0.0025, "-"), (0.005, "--")):
            dd = c3[sel & (c3.method == meth) & (c3.epsilon == eps)].sort_values("budget_fraction")
            axs[0].plot(dd.budget_fraction, dd.certification_rate, ls, marker=mkr, color=c, lw=1.6, label=f"{meth}, ε={eps}")
    axs[0].set_xlabel("audited share of the hidden rows"); axs[0].set_ylabel("certification rate")
    axs[0].set_title("n=1000, spread=distinct, 20% hidden (C3 and C1 curves coincide)"); axs[0].legend(fontsize=7)
    r = c3[sel & (c3.epsilon == 0.0025)]
    for meth, c, mkr in (("C1_after_audit", COL["C1"], "s"), ("C3", COL["C3"], "o")):
        dd = r[r.method == meth].sort_values("budget_fraction")
        axs[1].plot(dd.budget_fraction, dd.mean_interval_width, marker=mkr, color=c, lw=1.6, label=f"{meth}: mean pair width")
    dd = r[r.method == "C3"].sort_values("budget_fraction")
    axs[1].plot(dd.budget_fraction, 2 * dd.radius, ":", color=COL["C3"], lw=1.4, label="statistical width 2·radius")
    axs[1].axhline(0.0025, color=COL["base"], ls="--", lw=1); axs[1].text(0.1, 0.0025, " ε=0.0025", fontsize=7, va="bottom")
    axs[1].set_yscale("log"); axs[1].set_xlabel("audited share of the hidden rows"); axs[1].set_ylabel("interval width")
    axs[1].set_title("Hoeffding width ≫ C1 width, so C3 = C1 here"); axs[1].legend(fontsize=7)
    _save(fig, figdir, "fig6_c3_budget.png")

    # 7: certificates vs true regret (per run, eps=0.0025 rows)
    main = pd.read_csv(os.path.join(out, "main_rows.csv.gz"))
    d = main[(main.epsilon == 0.0025) & main.method.isin(["C0", "C1"])]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m_ in ("C0", "C1"):
        dd = d[d.method == m_]
        ax.scatter(dd.true_regret, dd.G, s=4, alpha=0.25, color=COL[m_], label=m_, linewidths=0)
    mx = float(d[["true_regret", "G"]].max().max())
    ax.plot([0, mx], [0, mx], color=COL["ink"], lw=1)
    ax.axhline(0.0025, color=COL["base"], ls="--", lw=1)
    ax.set_xlabel("true regret of the recommended model (oracle)"); ax.set_ylabel("certified bound G")
    ax.set_title("G ≥ true regret for every run (no point below the line)")
    ax.legend(markerscale=3, fontsize=8)
    _save(fig, figdir, "fig7_G_vs_true_regret.png")

    # 8: baselines vs C1 recommendation
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    d = ms[(ms.epsilon == 0.0025) & (ms.missing_target == 0.2) & (ms.spread == "moderate")]
    methods = BASE_METHODS + ["C1"]
    xs = np.arange(len(methods))
    for i, mech in enumerate(["MCAR", "MAR", "MNAR"]):
        dd = d[d.mechanism == mech].set_index("method").loc[methods]
        axs[0].bar(xs + (i - 1) * 0.27, dd.mean_true_regret_rec, width=0.25,
                   color=[COL["C1"] if m_ == "C1" else COL["base"] for m_ in methods],
                   alpha=[1.0, 0.65, 0.4][i], label=mech)
        axs[1].bar(xs + (i - 1) * 0.27, dd.picked_best, width=0.25,
                   color=[COL["C1"] if m_ == "C1" else COL["base"] for m_ in methods], alpha=[1.0, 0.65, 0.4][i])
    for ax in axs:
        ax.set_xticks(xs); ax.set_xticklabels([m_.replace("_", "\n") for m_ in methods], fontsize=7)
    axs[0].set_ylabel("mean true regret of the recommendation"); axs[1].set_ylabel("share picking the truly best model")
    axs[0].set_title("moderate spread, 20% hidden; bars: MCAR (dark) / MAR / MNAR (light)")
    axs[1].set_title("C1 bar = argmin G (a recommendation unless certified)")
    _save(fig, figdir, "fig8_baselines.png")

    # 9: runtime
    sz = t["size_summary"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for m_ in ("C0", "C1"):
        d = sz[(sz.method == m_) & (sz.epsilon == 0.0025) & (sz.missing_target == 0.2)].sort_values("n")
        ax.plot(d.n, d.mean_runtime_s * 1000, marker=mk[m_], color=COL[m_], lw=1.6)
        ax.annotate(m_, (d.n.iloc[-1], d.mean_runtime_s.iloc[-1] * 1000), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(right=sz.n.max() * 3)
    ax.set_xlabel("cohort size N (K = 3)"); ax.set_ylabel("ms per bounds computation")
    ax.set_title("C0/C1 runtime vs N")
    _save(fig, figdir, "fig9_runtime.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/synthetic")
    ap.add_argument("--figdir", default="report/figures")
    a = ap.parse_args()
    t = build_tables(a.out)
    build_figures(a.out, a.figdir, t)
    print("tables:", sorted(t), "-> figures in", a.figdir)


if __name__ == "__main__":
    main()

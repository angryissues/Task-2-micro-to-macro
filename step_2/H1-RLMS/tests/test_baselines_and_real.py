"""Baselines, synthetic generator and the real-mode CLI (on a SYNTHETIC table)."""
import os
import tempfile

import numpy as np
import pandas as pd

from src import baselines as bl
from src.bounds import c1_bounds
from src.metrics import full_brier
from src.synthetic import draw_mask, make_cohort, observation_probabilities


def test_generator_missing_fraction_and_mechanisms():
    coh = make_cohort(0, 4000, 3, "moderate")
    assert coh.p.shape == (4000, 3) and set(np.unique(coh.y)) <= {0.0, 1.0}
    for mech in ("MCAR", "MAR", "MNAR"):
        for m in (0.1, 0.2, 0.4):
            pi = observation_probabilities(coh, mech, m)
            assert abs((1 - pi).mean() - m) < 1e-6
    pi = observation_probabilities(coh, "MNAR", 0.2)
    assert pi[coh.y == 1].mean() < pi[coh.y == 0].mean()           # depends on the label
    assert np.array_equal(draw_mask(pi, 3), draw_mask(pi, 3))       # reproducible


def test_complete_case_and_model_assisted_limits():
    coh = make_cohort(1, 500, 3, "moderate")
    R = full_brier(coh.p, coh.y)
    seen = np.ones(500, bool)
    res = bl.complete_case(coh.p, coh.y, seen)
    assert np.allclose(res["estimates"], R) and res["certified"] is False
    assert res["guarantee_type"] == "recommendation_only"
    # nothing observed -> no recommendation
    assert bl.complete_case(coh.p, np.full(500, np.nan), np.zeros(500, bool))["recommended_model"] is None
    # model-assisted with perfect q on hidden rows: expected loss equals the true expected loss
    seen = np.random.default_rng(0).random(500) < 0.6
    res = bl.model_assisted(coh.p, np.where(seen, coh.y, np.nan), seen, coh.true_prob,
                            n_fills=400, rng=np.random.default_rng(1))
    assert res["convergence_gap"] < 5e-3             # Bernoulli fills converge to the expected value
    assert res["guarantee_type"] == "recommendation_only"


def test_ipw_unbiased_under_known_pi_mcar_and_clipping():
    coh = make_cohort(2, 600, 3, "moderate")
    R = full_brier(coh.p, coh.y)
    pi = observation_probabilities(coh, "MCAR", 0.3)
    est = []
    for s in range(400):
        seen = draw_mask(pi, s)
        est.append(bl.ipw(coh.p, np.where(seen, coh.y, np.nan), seen, pi)["estimates"])
    assert np.abs(np.mean(est, axis=0) - R).max() < 2e-3
    # clipping at 0.05 changes small probabilities only
    seen = draw_mask(pi, 0)
    pi_small = pi.copy()
    pi_small[:50] = 0.01
    a = bl.ipw(coh.p, np.where(seen, coh.y, np.nan), seen, pi_small)["estimates"]
    b = bl.ipw(coh.p, np.where(seen, coh.y, np.nan), seen, pi_small, clip=0.05)["estimates"]
    assert not np.allclose(a, b)


def test_c1_never_issues_a_wrong_certificate_on_synthetic_masks():
    from src.certification import select_with_certificate
    from src.metrics import true_regret_vector, wrong_certificate
    for mech in ("MCAR", "MAR", "MNAR"):
        for ds in range(3):
            coh = make_cohort(ds, 300, 3, "close")
            R = full_brier(coh.p, coh.y)
            pi = observation_probabilities(coh, mech, 0.2)
            for s in range(30):
                seen = draw_mask(pi, s)
                for eps in (0.0, 0.0025, 0.01, 0.05):
                    r = select_with_certificate(coh.p, np.where(seen, coh.y, np.nan), seen, eps)
                    tr = true_regret_vector(R)[r["recommended_model"]]
                    assert not wrong_certificate(r["certified"], tr, eps)


def test_real_mode_cli_on_synthetic_table():
    from src import experiments
    coh = make_cohort(5, 400, 3, "moderate")
    pi = observation_probabilities(coh, "MCAR", 0.3)
    seen = draw_mask(pi, 0)
    df = pd.DataFrame({"y": np.where(seen, coh.y, np.nan), "p_M0": coh.p[:, 0], "p_M1": coh.p[:, 1],
                       "p_M2": coh.p[:, 2], "q": coh.true_prob, "f1": coh.X[:, 0], "f2": coh.X[:, 1]})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "table.csv")
        df.to_csv(path, index=False)
        out = os.path.join(d, "out")
        experiments.main(["real", "--input", path, "--out", out, "--feature-cols", "f1,f2",
                          "--artificial", "--n-seeds", "3",
                          "--config", os.path.join(os.path.dirname(__file__), "..", "configs", "h1_default.json")])
        dec = pd.read_csv(os.path.join(out, "real_decisions.csv"))
        assert dec.true_regret.isna().all()                         # unknown truth is never filled
        c1 = dec[(dec.method == "C1") & (dec.epsilon == 0.1)]
        _, _, g = c1_bounds(coh.p, np.where(seen, coh.y, np.nan), seen)
        assert abs(c1.G.iloc[0] - g.min()) < 1e-12
        for f in ("real_pairwise_bounds.csv", "real_c2_sensitivity.csv", "real_baselines.csv",
                  "real_artificial_masks.csv"):
            assert os.path.exists(os.path.join(out, f))
        art = pd.read_csv(os.path.join(out, "real_artificial_masks.csv"))
        assert not art.wrong_certificate.any()
        # resume: second call skips finished artificial chunks and does not fail
        experiments.main(["real", "--input", path, "--out", out, "--artificial", "--n-seeds", "3",
                          "--config", os.path.join(os.path.dirname(__file__), "..", "configs", "h1_default.json")])


def test_quick_smoke_and_resume():
    from src import experiments
    with tempfile.TemporaryDirectory() as d:
        cfgp = os.path.join(os.path.dirname(__file__), "..", "configs", "h1_default.json")
        experiments.main(["run", "--config", cfgp, "--out", d, "--quick", "--studies", "main", "c3"])
        n1 = len(pd.read_csv(os.path.join(d, "main_rows.csv.gz")))
        experiments.main(["run", "--config", cfgp, "--out", d, "--quick", "--studies", "main", "c3"])
        assert n1 == len(pd.read_csv(os.path.join(d, "main_rows.csv.gz")))   # resumed, nothing duplicated

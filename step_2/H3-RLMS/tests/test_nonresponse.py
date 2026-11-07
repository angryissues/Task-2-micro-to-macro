import numpy as np

from src.nonresponse import (AttemptOracle, calibrate_intercept,
                             draw_initial_mask, initial_missing_probabilities,
                             run_nonresponse)
from src.nonresponse import SelectionState


def test_calibrated_mar_hits_target():
    age = np.linspace(18, 70, 1000)
    low = np.tile([0.0, 1.0], 500)
    y = np.zeros(1000)
    p = initial_missing_probabilities(age, low, y, "mar", target=.2)
    assert abs(p.mean() - .2) < 1e-9


def test_attempt_failure_costs_without_revealing_label():
    labels = np.array([1, 0])
    oracle = AttemptOracle(labels, np.zeros(2), np.random.default_rng(1))
    ok, label = oracle.attempt(0)
    assert not ok and label is None


def test_h3_run_never_exposes_failed_label_and_charges_attempt():
    p = np.array([[.1, .2], [.8, .7], [.4, .5], [.2, .3]])
    y = np.array([0, 1, 0, 1])
    observed = np.array([True, False, False, True])
    state = SelectionState.create(p, y, observed)
    result = run_nonresponse(state, y, np.full(4, .5), "r1", .001, 2, seed=4)
    assert result["queries_attempted"] >= result["labels_obtained"] - observed.sum()
    assert result["total_cost"] <= 2
    assert all("label" in e for e in result["trace"] if e["success"])
    assert all("label" not in e for e in result["trace"] if not e["success"])

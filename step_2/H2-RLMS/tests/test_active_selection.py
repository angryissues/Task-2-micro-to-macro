import numpy as np

from src.active_selection import (LabelOracle, SelectionState, active_testing_brier_adapter, c1_bounds, make_stratified_random,
                                  exact_minimax_cost, hara_sampling, policy_worst_case_cost, q0, q1, q2, q3,
                                  random_policy, run_policy, uncertainty_sampling)


def fixture_state():
    p = np.array([[.1, .8, .4], [.8, .3, .5], [.2, .6, .4], [.7, .4, .2]])
    y = np.array([0, 1, 1, 0])
    observed = np.array([True, False, False, True])
    return p, y, observed, SelectionState.create(p, y, observed)


def test_public_state_discards_hidden_labels():
    p, y, seen, state = fixture_state()
    changed = y.copy()
    changed[~seen] = 1 - changed[~seen]
    alternative = SelectionState.create(p, changed, seen)
    assert np.array_equal(state.labels, alternative.labels)
    assert np.allclose(state.upper, alternative.upper)


def test_reveal_matches_complete_c1_recomputation():
    p, y, seen, state = fixture_state()
    state.reveal(1, int(y[1]))
    lo, up = c1_bounds(p, y, np.array([True, True, False, True]), np.full(4, .25))
    assert np.allclose(state.lower, lo)
    assert np.allclose(state.upper, up)


def test_every_policy_returns_a_hidden_row():
    _, _, _, state = fixture_state()
    q = np.array([.5, .2, .8, .5])
    for policy in (q0, q1, q2):
        assert policy(state) in {1, 2}
    assert q3(state, q) in {1, 2}
    assert uncertainty_sampling(state, q) in {1, 2}
    assert random_policy(state, np.random.default_rng(3)) in {1, 2}


def test_hara_sampling_returns_visible_probability_and_no_hidden_label():
    _, _, _, state = fixture_state()
    row, probability = hara_sampling(state, np.array([.5, .2, .8, .5]), np.random.default_rng(2))
    assert row in {1, 2}
    assert 0 < probability <= 1


def test_active_testing_adapter_returns_visible_probability_and_no_hidden_label():
    _, _, _, state = fixture_state()
    row, probability = active_testing_brier_adapter(state, np.array([.5, .2, .8, .5]), np.random.default_rng(2))
    assert row in {1, 2}
    assert 0 < probability <= 1


def test_stratified_random_cycles_over_available_groups():
    _, _, _, state = fixture_state()
    choose = make_stratified_random(np.array(['a', 'a', 'b', 'b']), np.random.default_rng(4))
    first, second = choose(state), choose(state)
    assert {first, second} == {1, 2}


def test_runner_never_spends_past_budget():
    p, y, seen, state = fixture_state()
    result = run_policy(state, LabelOracle(y), q2, epsilon=0.0, budget=1.0)
    assert result["total_cost"] <= 1.0
    assert result["status"] in {"certified", "budget_exhausted"}


def test_exact_minimax_is_no_worse_than_greedy_policies():
    p = np.array([[.1, .9], [.3, .6], [.8, .2], [.4, .5]])
    y = np.zeros(4)
    state = SelectionState.create(p, y, np.zeros(4, dtype=bool), costs=np.array([1., 2., 1., 5.]))
    optimal = exact_minimax_cost(state, epsilon=.02)
    assert optimal['states_evaluated'] > 1
    for policy in (q0, q1, q2):
        greedy = policy_worst_case_cost(state, epsilon=.02, policy=policy)
        assert optimal['worst_case_cost'] <= greedy['worst_case_cost']


def test_exact_minimax_rejects_large_tree():
    p = np.tile(np.array([[.2, .8]]), (9, 1))
    state = SelectionState.create(p, np.zeros(9), np.zeros(9, dtype=bool))
    import pytest
    with pytest.raises(ValueError, match='at most 8'):
        exact_minimax_cost(state, epsilon=0.)

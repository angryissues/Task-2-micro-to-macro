import numpy as np

from src.active_selection import SelectionState, exact_minimax_cost, policy_worst_case_cost, q0, q1, q2


def make_state(predictions, costs):
    n = len(costs)
    return SelectionState.create(np.array(predictions), np.zeros(n), np.zeros(n, dtype=bool), costs=np.array(costs))


def test_q0_q1_have_a_costly_greedy_counterexample():
    state = make_state([
        [.30148, .37893], [.64095, .82997], [.41887, .62640],
        [.51897, .24150], [.21097, .45420], [.78259, .58955],
    ], [2, 5, 1, 5, 1, 2])
    optimal = exact_minimax_cost(state, .02)['worst_case_cost']
    assert optimal == 14
    assert policy_worst_case_cost(state, .02, q0)['worst_case_cost'] == 16
    assert policy_worst_case_cost(state, .02, q1)['worst_case_cost'] == 16


def test_q2_is_not_a_general_minimax_solver():
    state = make_state([
        [.55602, .88061, .13733, .87469], [.37347, .41604, .82559, .72266],
        [.06163, .25767, .91366, .11295], [.71811, .29428, .39541, .82543],
        [.80214, .05472, .88222, .04320], [.62040, .55041, .10465, .96649],
    ], [5, 2, 2, 5, 5, 5])
    assert exact_minimax_cost(state, .05)['worst_case_cost'] == 19
    assert policy_worst_case_cost(state, .05, q2)['worst_case_cost'] == 24

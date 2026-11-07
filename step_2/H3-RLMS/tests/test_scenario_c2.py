import numpy as np
from src.scenario_c2 import ScenarioState, scenario_count_bounds

def test_gamma_infinite_is_unrestricted():
    assert scenario_count_bounds(2, 5, 7, float('inf')) == (0, 7)

def test_scenario_state_reduces_hidden_pool_after_reveal():
    p = np.array([[.1, .2], [.8, .7], [.4, .5], [.2, .3]])
    y = np.array([0, 1, 0, 1]); seen = np.array([True, False, False, True])
    gid = np.array([0, 0, 1, 1])
    state = ScenarioState.create(p, y, seen, gid, 2.0)
    before = state.value
    state.reveal(1, 1)
    assert state.observed[1] and state.spent == 1.0
    assert state.value <= before + 1e-12

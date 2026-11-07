import numpy as np
from src.trajectory_bounds import admissible_paths, bounds

def test_endpoint_p1_matches_independent_completion():
    obs=np.array([[1,np.nan,0],[np.nan,1,np.nan]],float)
    p=np.array([[[.8,.2],[.5,.5],[.2,.8]],[[.6,.4],[.7,.3],[.3,.7]]])
    _,u1,_=bounds(p,obs,target='endpoint',mode='p1')
    _,u0,_=bounds(p,obs,target='endpoint',mode='p0')
    np.testing.assert_allclose(u1,u0)

def test_first_exit_path_constraints_are_monotone():
    paths=admissible_paths([1,np.nan,np.nan])
    assert all(np.all(np.diff(np.maximum.accumulate(path==0))>=0) for path in paths)
    paths_zero=admissible_paths([0,np.nan,np.nan])
    assert all(np.all(np.maximum.accumulate(path==0)==1) for path in paths_zero)

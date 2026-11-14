"""Meaningful pre-confirmation checks against reference implementations and enumeration."""
import argparse,json
from pathlib import Path
import numpy as np
from replicate import module,lookahead
from paths_groups import cache_bounds,BITS,POW
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();repo=Path(a.repo);h2=module('nls_check_h2',repo/'step_2/H2-RLMS/src/active_selection.py');rng=np.random.default_rng(1007);tests={}
    for rep in range(20):
        p=rng.random((8,3));y=rng.integers(2,size=8);seen=rng.random(8)>.5;state=h2.SelectionState.create(p,y,seen);state._eligible_indices=state.hidden_indices;q=rng.random(8)
        if not len(state.hidden_indices):continue
        assert lookahead(state,q,'Q2',h2)==h2.q2(state)
        assert lookahead(state,q,'Q3',h2)==h2.q3(state,q)
        altered=y.copy();altered[~seen]=1-altered[~seen];other=h2.SelectionState.create(p,altered,seen);np.testing.assert_allclose(state.upper,other.upper)
    tests['vectorized_Q2_Q3_match_H2_reference_20_cases']=True;tests['hidden_label_invariance']=True
    pred=rng.random((4,3,3));p0=cache_bounds(pred,'first_exit','P0');p1=cache_bounds(pred,'first_exit','P1');assert np.all(p1<=p0+1e-12);tests['P1_tighter_than_independent_exit_marginals']=True
    for target in ['endpoint','multi_horizon']:
        np.testing.assert_allclose(cache_bounds(pred,target,'P0'),cache_bounds(pred,target,'P1'),atol=1e-14)
    tests['endpoint_and_additive_equal_without_extra_constraints']=True
    # All known outcomes must have exact pair differences in both P0/P1.
    for path in BITS:
        code=int((path+1)@POW);F=np.maximum.accumulate(path==0).astype(float);loss=((pred-F[None,:,None])**2).mean(1);diff=loss[:,:,None]-loss[:,None,:];np.testing.assert_allclose(p0[:,code],diff,atol=1e-14);np.testing.assert_allclose(p1[:,code],diff,atol=1e-14)
    tests['fully_known_paths_exact_in_both_modes']=True
    from build_panel import state
    s=state(np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,9701,199999,202301]));assert np.isnan(s[:7]).all() and s[7]==0 and np.isnan(s[8]) and s[9]==0 and s[10]==0 and np.isnan(s[11:13]).all() and np.all(s[13:]==1);tests['employment_codes_and_ambiguity']=True
    h5=module('nls_check_h5',repo/'step_2/H5-RLMS/scripts/analysis.py');tests['subgroup_reference_checks']=h5.controlled_checks();Path(a.out).write_text(json.dumps(tests,indent=2));print(json.dumps(tests,indent=2))
if __name__=='__main__':main()

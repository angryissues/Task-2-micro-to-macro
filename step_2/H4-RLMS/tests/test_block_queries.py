from itertools import product
import numpy as np
import pytest
from src.block_queries import PublicTrajectoryState,TrajectoryOracle,select_query,run_queries
from src.trajectory_bounds import bounds,first_exit

def test_all_ternary_states_match_independent_path_enumeration():
    pred=np.random.default_rng(52).random((1,3,3))
    for digits in product([0,1,2],repeat=3):
        obs=np.array([[np.nan if d==0 else d-1 for d in digits]])
        state=PublicTrajectoryState(pred,obs);_,expected,_=bounds(pred,obs,target='first_exit',mode='p1')
        np.testing.assert_allclose(state.upper,expected,atol=1e-14)

def test_fully_known_p0_and_p1_are_both_exact():
    pred=np.random.default_rng(42).random((1,3,3))
    for path in product([0,1],repeat=3):
        obs=np.array([path],float);ls=((pred[0]-first_exit(path)[:,None])**2).mean(0);exact=ls[:,None]-ls[None,:]
        for mode in ['p0','p1']:
            _,U,_=bounds(pred,obs,target='first_exit',mode=mode);np.testing.assert_allclose(U,exact,atol=1e-14)

def test_p1_never_wider_than_correct_p0():
    pred=np.random.default_rng(69).random((1,3,3))
    for digits in product([0,1,2],repeat=3):
        obs=np.array([[np.nan if d==0 else d-1 for d in digits]])
        _,p0,_=bounds(pred,obs,target='first_exit',mode='p0');_,p1,_=bounds(pred,obs,target='first_exit',mode='p1');assert np.all(p1<=p0+1e-12)

@pytest.mark.parametrize('mode',['point_only','block_only','mixed'])
def test_query_maximizes_public_minimax_gain_against_brute_force(mode):
    pred=np.random.default_rng(53).random((3,3,3));obs=np.array([[1,np.nan,np.nan],[np.nan,0,1],[np.nan,np.nan,1]],float);s=PublicTrajectoryState(pred,obs);choice=select_query(s,mode,1.5,2)
    def gain(i,ids,cost):
        worst=-np.inf
        for fill in product([0,1],repeat=len(ids)):
            z=obs.copy();z[i,list(ids)]=fill;_,_,g=bounds(pred,z,target='first_exit',mode='p1');worst=max(worst,float(g.min()))
        return (s.value-worst)/cost
    options=[]
    for i in range(3):
        hidden=tuple(np.flatnonzero(np.isnan(obs[i])))
        if mode in ['point_only','mixed']:options.extend([gain(i,(h,),1.) for h in hidden])
        if mode in ['block_only','mixed'] and hidden:options.append(gain(i,hidden,1.5))
    assert choice is not None;assert abs(gain(choice.row,choice.horizons,choice.cost)-max(options))<1e-12

def test_changing_never_queried_truth_does_not_change_policy_trace():
    rng=np.random.default_rng(7);pred=rng.random((8,3,3));truth=rng.integers(2,size=(8,3));public=np.full((8,3),np.nan);oracle=TrajectoryOracle(truth)
    a=run_queries(pred,public,oracle,'point_only',1.5,2,epsilon=0)
    changed=1-truth
    for i,ids in oracle.calls:changed[i,list(ids)]=truth[i,list(ids)]
    other=TrajectoryOracle(changed);b=run_queries(pred,public,other,'point_only',1.5,2,epsilon=0)
    assert len(oracle.calls)==2;assert oracle.calls==other.calls;assert a['trace']==b['trace']

def test_already_certified_does_not_query_or_charge():
    pred=np.full((4,3,3),.5);public=np.full((4,3),np.nan);oracle=TrajectoryOracle(np.zeros((4,3)))
    r=run_queries(pred,public,oracle,'mixed',1.5,20);assert r['certified'] and r['initially_certified'];assert r['total_cost']==0 and not oracle.calls

@pytest.mark.parametrize('mode',['point_only','block_only','mixed'])
def test_complete_reveal_certifies_and_never_charges_observed_answers(mode):
    rng=np.random.default_rng(18);pred=rng.random((5,3,3));truth=rng.integers(2,size=(5,3));public=truth.astype(float);public[rng.random((5,3))<.6]=np.nan;oracle=TrajectoryOracle(truth);r=run_queries(pred,public,oracle,mode,1.5,100)
    losses=((pred-np.maximum.accumulate(truth==0,axis=1)[:,:,None])**2).mean((0,1));assert r['certified'];assert losses[r['model']]-losses.min()<=.0025+1e-10
    paid=[(i,h) for i,ids in oracle.calls for h in ids];assert len(paid)==len(set(paid));assert all(np.isnan(public[i,h]) for i,h in paid)
    assert all(t['G_after']<=t['G_before']+1e-10 for t in r['trace'])

def test_insufficient_budget_and_invalid_inputs():
    pred=np.random.default_rng(1).random((3,3,3));public=np.full((3,3),np.nan);s=PublicTrajectoryState(pred,public)
    assert select_query(s,'mixed',1.5,.4) is None
    for price in [0,-1,np.nan]:
        with pytest.raises(ValueError):select_query(s,'mixed',price,20)

def test_information_cache_matches_scalar_reference_in_all_states_and_targets():
    from scripts.recalculate_p2 import cache_upper
    pred=np.random.default_rng(32).random((1,3,3));power=3**np.arange(3)
    for target in ['endpoint','multi_horizon','first_exit']:
      for mode in ['p0','p1']:
        cache=cache_upper(pred,target,mode)
        for digits in product([0,1,2],repeat=3):
            code=int(np.asarray(digits)@power);obs=np.array([[np.nan if d==0 else d-1 for d in digits]]);_,u,_=bounds(pred,obs,target=target,mode=mode);np.testing.assert_allclose(cache[:,code].mean(0),u,atol=1e-14)

"""H6 T2/T3 and T4 with exact finite-cohort bounds and paid nonresponse."""
import argparse,hashlib,importlib.util,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog

MODELS=['baseline','logistic','hgb'];EPS=.0025;TOL=1e-12

def pair_rows(p,y,seen):
    d0=p[:,:,None]**2-p[:,None,:]**2;d1=d0-2*(p[:,:,None]-p[:,None,:]);known=np.where(y[:,None,None]==1,d1,d0)
    return np.where(seen[:,None,None],known,np.minimum(d0,d1)),np.where(seen[:,None,None],known,np.maximum(d0,d1))

def robust_max(d,w,eta):
    if not 0<=eta<=1:raise ValueError('eta must be in [0,1]')
    w=np.asarray(w,float);w=w/w.sum();v=(1-eta)*w;capacity=2*eta*w;remaining=1-v.sum()
    for i in np.argsort(-d,kind='stable'):
        add=min(remaining,capacity[i]);v[i]+=add;remaining-=add
        if remaining<=1e-14:break
    assert abs(v.sum()-1)<1e-10
    return float(v@d)

def robust_bounds(p,y,seen,eta,w=None):
    if w is None:w=np.ones(len(p))/len(p)
    _,hi=pair_rows(p,y,seen);k=p.shape[1];U=np.zeros((k,k))
    for a in range(k):
        for b in range(k):
            if a!=b:U[a,b]=robust_max(hi[:,a,b],w,eta)
    return U

def decision(U,eps=EPS):
    G=U.max(1);k=int(G.argmin());return k,float(G[k]),bool(G[k]<=eps+TOL)

def q2_vectorized(state,h2):
    """Exact H2 Q2 minimax lookahead, vectorized with the original tie-breaking."""
    ids=state._eligible_indices;p=state.predictions[ids];w=state.weights[ids]
    coeff=-2*w[:,None,None]*(p[:,:,None]-p[:,None,:]);delta0=-np.maximum(coeff,0);delta1=delta0+coeff
    v0=(state.upper+delta0).max(2).min(1);v1=(state.upper+delta1).max(2).min(1)
    scores=state.value-np.maximum(v0,v1)
    if np.all(scores<=1e-15):return int(h2.q1(state))
    return int(h2._tie_break(ids,scores,state.costs))

def checks_and_synthetic():
    rng=np.random.default_rng(606);checks={};worst=0.;tvrows=[]
    for rep in range(25):
        n=2+rep%11;w=rng.random(n);w/=w.sum();d=rng.uniform(-1,1,n)
        vals=[]
        for eta in [0,.25,.5,1]:
            exact=linprog(-d,A_eq=np.ones((1,n)),b_eq=[1],bounds=list(zip((1-eta)*w,(1+eta)*w)),method='highs');assert exact.success
            val=robust_max(d,w,eta);worst=max(worst,abs(val+exact.fun));assert abs(val+exact.fun)<1e-9;vals.append(val)
        assert np.all(np.diff(vals)>=-1e-12)
    checks['T2_greedy_matches_100_linear_programs']=worst;checks['T2_monotone_in_eta']=True
    p=rng.random((7,3));y=rng.integers(2,size=7);seen=rng.random(7)>.4;_,hi=pair_rows(p,y,seen);np.testing.assert_allclose(robust_bounds(p,y,seen,0),hi.mean(0),atol=1e-14);checks['T2_eta0_equals_C1']=True
    # Finite distributions on identical records, allowed weights, and hidden labels.
    w=np.ones(7)/7;v=w.copy();v[0]+=.4/7;v[1]-=.4/7
    actual=((p-y[:,None])**2);actualdiff=np.einsum('n,nij->ij',v,actual[:,:,None]-actual[:,None,:]);assert np.all(actualdiff<=robust_bounds(p,y,seen,.5,w)+1e-12);checks['T2_permitted_reweighting_covered']=True
    # A fixed joint distribution on (X,Y); forecast is a function of X.
    P=np.array([.45,.05,.05,.45]);X=np.array([0,0,1,1]);Y=np.array([0,1,0,1]);pred=np.array([[.1,.9,.5],[.9,.1,.5]])[X];ls=(pred-Y[:,None])**2;D=ls[:,:,None]-ls[:,None,:];base=np.einsum('n,nij->ij',P,D)
    for rho in [0,.01,.03,.1]:
        Q=P.copy();Q[0]-=rho;Q[1]+=rho;actualrho=.5*abs(P-Q).sum();assert abs(actualrho-rho)<1e-12;upper=base+2*rho;np.fill_diagonal(upper,0);actual=np.einsum('n,nij->ij',Q,D);assert np.all(actual<=upper+TOL);k,G,cert=decision(upper);qloss=Q@ls;tvrows.append(dict(rho=rho,G=G,certified=cert,selected_model=k,true_Q_regret=float(qloss[k]-qloss.min())))
    checks['T3_known_joint_TV_bound']=True;assert np.all(np.diff([r['G'] for r in tvrows])>=-TOL);checks['T3_bound_monotone_in_rho']=True
    # IID correction is tested only on synthetic independent draws from finite P.
    iid=[];alpha=.05;n=400;J=6;radius=np.sqrt(2*np.log(2*J/alpha)/n)
    for rep in range(100):
        sample=rng.choice(4,size=n,p=P);u=D[sample].mean(0)+radius+2*.03;np.fill_diagonal(u,0);Q=P.copy();Q[0]-=.03;Q[1]+=.03;iid.append(bool(np.all(np.einsum('n,nij->ij',Q,D)<=u+TOL)))
    checks['T3_iid_synthetic_coverage_100']=float(np.mean(iid))
    # Same X, opposite conditional Y: exact population expectations, not sampled luck.
    forecast=np.array([.1,.9]);old=.9*forecast**2+.1*(forecast-1)**2;new=.1*forecast**2+.9*(forecast-1)**2
    assert old.argmin()==0 and new.argmin()==1;counter={'old_winner':0,'new_winner':1,'old_choice_new_regret':float(new[0]-new.min()),'X_distribution_change':0.,'joint_TV':.8};checks['conditional_shift_counterexample']=True
    # Pure covariate shift: fixed P(Y|X), known density ratios; clipping is recorded.
    P=np.array([.81,.09,.01,.09]);Q=np.array([.09,.01,.09,.81]);pr=np.tile(forecast,(4,1));ys=np.array([0,1,0,1]);loss=(pr-ys[:,None])**2;ratio=np.array([1/9,1/9,9,9]);weights=P*np.clip(ratio,.2,5);weights/=weights.sum();a=int((P@loss).argmin());b=int((weights@loss).argmin());truth=Q@loss;assert a==0 and b==1 and b==truth.argmin();checks['covariate_shift_positive_control']=True
    positive={'T0_regret':float(truth[a]-truth.min()),'T1_known_clipped_weights_regret':float(truth[b]-truth.min()),'scope':'mechanism check with known ratios; no proof of estimated RLMS weights'}
    return checks,tvrows,counter,positive

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);cfg=json.loads(Path(args.config).read_text());x=pd.read_csv(args.input)
    spec=importlib.util.spec_from_file_location('h2active',str(Path(args.repo)/'step_2/H2-RLMS/src/active_selection.py'));h2=importlib.util.module_from_spec(spec);sys.modules['h2active']=h2;spec.loader.exec_module(h2)
    conditional=[];rho_rows=[];rows=[];coverage_failures=0;union_paths=len(cfg['temporal_pairs'])*args.reps*3;B=len(cfg['t4_query_budgets']);alpha_path=.05/union_paths
    for oldyear,newyear in cfg['temporal_pairs']:
        old=x[x.target_year==oldyear];new=x[x.target_year==newyear].reset_index(drop=True);pold=old[MODELS].to_numpy();yold=old.target.astype(int).to_numpy();p=new[MODELS].to_numpy();y=new.target.astype(int).to_numpy();n=len(new);assert not new.idind.duplicated().any()
        for missing in (0.,.2):
            seen=np.random.default_rng(700+oldyear).random(len(old))>=missing
            for eta in cfg['t2_eta_grid']:
                U=robust_bounds(pold,np.where(seen,yold,np.nan),seen,eta);k,G,cert=decision(U);conditional.append(dict(old_year=oldyear,new_year=newyear,missing_fraction=missing,eta=eta,model=MODELS[k],G=G,certified=cert,scope='same_record_weight_scenario',future_certificate=False))
        base=robust_bounds(pold,yold,np.ones(len(old),bool),0)
        for rho in cfg['t3_joint_tv_rho_grid']:
            U=base+2*rho;np.fill_diagonal(U,0);k,G,cert=decision(U);rho_rows.append(dict(old_year=oldyear,new_year=newyear,rho=rho,model=MODELS[k],G=G,conditional_certificate=cert,rho_known_on_RLMS=False,unconditional_future_certificate=False))
        true_loss=((p-y[:,None])**2).mean(0);true_D=true_loss[:,None]-true_loss[None,:];regret=true_loss-true_loss.min()
        for scenario in ['complete_response','MAR','MNAR']:
          for rep in range(args.reps):
            rng=np.random.default_rng(800000+newyear*1000+rep);potential=rng.random(n)
            prob=np.ones(n) if scenario=='complete_response' else (np.where(new.age<35,.55,.85) if scenario=='MAR' else np.where(y==1,.85,.45));responds=potential<prob
            order=np.random.default_rng(900000+newyear*1000+rep).permutation(n)
            for policy in ['uniform_C3','active_Q2_C1','active_Q1_C1']:
                state=h2.SelectionState.create(p,np.zeros(n),np.zeros(n,bool));attempted=np.zeros(n,bool);history=[];success=0
                for t in range(1,max(cfg['t4_query_budgets'])+1):
                    if policy=='uniform_C3':i=int(order[t-1])
                    else:
                        state._eligible_indices=np.flatnonzero(~attempted);i=q2_vectorized(state,h2) if policy=='active_Q2_C1' else int(h2.q1(state))
                    attempted[i]=True;history.append(i)
                    if responds[i]:state.reveal(i,int(y[i]));success+=1
                    if t not in cfg['t4_query_budgets']:continue
                    upper=state.upper.copy();lower=state.lower.copy();event_failed=False;radius=np.nan
                    if policy=='uniform_C3':
                        ids=np.asarray(history);seen=responds[ids];visible=np.where(seen,y[ids],np.nan);low_sample,high_sample=pair_rows(p[ids],visible,seen)
                        radius=np.sqrt(2*np.log(2*6*B/alpha_path)/t)
                        # Uniform ATTEMPTS remain a sample without replacement even under MNAR.
                        # Missing sample answers are bounded over 0/1, not dropped or imputed.
                        stat_hi=high_sample.mean(0)+radius;stat_lo=low_sample.mean(0)-radius;np.fill_diagonal(stat_hi,0);np.fill_diagonal(stat_lo,0)
                        upper=np.minimum(upper,stat_hi);lower=np.maximum(lower,stat_lo);event_failed=bool(np.any(lower>upper+1e-10));coverage_failures+=int(np.any(true_D>upper+1e-10) or np.any(true_D<lower-1e-10))
                    k,G,cert=decision(upper);cert=cert and not event_failed
                    if cert:assert regret[k]<=EPS+1e-10
                    rows.append(dict(old_year=oldyear,new_year=newyear,scenario=scenario,rep=rep,method=policy,budget=t,cost=t,successful=success,failed=t-success,model=MODELS[k],G=G,certified=cert,true_regret=regret[k],interval_event_failed=event_failed,stat_radius=radius,alpha_total=.05,alpha_path=alpha_path,scope='finite_new_cohort'))
        print('completed pair',oldyear,newyear,flush=True)
    pd.DataFrame(conditional).to_csv(out/'T2_weight_scenarios.csv',index=False);pd.DataFrame(rho_rows).to_csv(out/'T3_RLMS_conditional_sensitivity.csv',index=False);runs=pd.DataFrame(rows);runs.to_csv(out/'T4_runs.csv',index=False)
    runs.groupby(['new_year','scenario','method','budget']).agg(runs=('rep','count'),certificate_rate=('certified','mean'),mean_G=('G','mean'),mean_regret=('true_regret','mean'),mean_success=('successful','mean'),mean_failed=('failed','mean')).reset_index().to_csv(out/'T4_summary.csv',index=False)
    checks,tv,counter,positive=checks_and_synthetic();pd.DataFrame(tv).to_csv(out/'T3_synthetic_TV.csv',index=False)
    # Verify imported exact C1 against direct pairwise arithmetic and hidden-data erasure.
    pp=np.array([[.1,.7,.4],[.8,.2,.3],[.5,.4,.1]]);yy=np.array([1,0,1]);ss=np.array([True,False,False]);pub=np.where(ss,yy,0);s=h2.SelectionState.create(pp,pub,ss);_,hi=pair_rows(pp,np.where(ss,yy,np.nan),ss);np.testing.assert_allclose(s.upper,hi.mean(0),atol=1e-14);checks['T4_C1_direct_formula']=True
    altered=yy.copy();altered[~ss]=1-altered[~ss];s2=h2.SelectionState.create(pp,altered,ss);np.testing.assert_allclose(s2.upper,s.upper,atol=1e-14);assert h2.q1(s)==h2.q1(s2);checks['T4_hidden_labels_do_not_change_state_or_policy']=True
    rg=np.random.default_rng(818)
    for it in range(12):
        probs=rg.random((6,3));labs=rg.integers(2,size=6);obs=rg.random(6)>.6;st=h2.SelectionState.create(probs,labs,obs);st._eligible_indices=np.flatnonzero(~obs)
        if len(st._eligible_indices):assert q2_vectorized(st,h2)==h2.q2(st)
    checks['T4_vectorized_Q2_matches_H2_reference']=True
    verification={'checks':checks,'counterexample':counter,'positive_control':positive,'C3_coverage_failure_checkpoints':coverage_failures,'C3_coverage_scope':'simultaneous all budgets, 3 cohorts, scenarios, repetitions by union bound; uniform sampling of attempts without replacement, no iid panel assumption','runs':len(runs),'repetitions':args.reps,'new_labels_initially_observed':0,'primary_active_policy':'H2 Q2, selected in H2 for unit query costs','additional_active_policy':'H2 Q1, previous full-RLMS comparator','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(out/'verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--config',required=True);ap.add_argument('--repo',required=True);ap.add_argument('--out',required=True);ap.add_argument('--reps',type=int,default=100);run(ap.parse_args())

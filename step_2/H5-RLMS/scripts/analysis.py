"""Exact subgroup certification and paid single-label queries; no oracle in policies."""
import argparse,itertools,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd

NAMES=['baseline','logistic','hgb']; TOL=1e-12

def arrays(d):
    n=len(d);p=np.stack([d[[f'p_{m}_{h}' for m in NAMES]].to_numpy() for h in (1,2,3)],axis=1)
    y=d[['e1','e2','e3']].to_numpy();age=np.where(d.age<35,0,np.where(d.age<50,1,2))
    edu=np.where(d.education.isna(),2,np.where(d.education.isin([21,22,23]),1,0))
    group=np.stack([np.ones(n,bool),age==0,age==1,age==2,d.sex==1,d.sex==2,edu==0,edu==1,edu==2],axis=1)
    labels=['all','age18_34','age35_49','age50_64','male','female','no_higher','higher','education_unknown']
    keep=group.sum(0)>0;return p,y,group[:,keep],np.array(labels)[keep],age

class State:
    def __init__(self,p,visible,groups):
        self.p=p;self.n,self.h,self.k=p.shape;self.groups=groups;self.sizes=groups.sum(0);self.seen=np.isfinite(visible);self.attempted=self.seen.copy();self.visible=visible.copy()
        self.d0=p[:,:,:,None]**2-p[:,:,None,:]**2;self.d1=self.d0-2*(p[:,:,:,None]-p[:,:,None,:]);self.hi=np.maximum(self.d0,self.d1)
        row=np.where(self.seen[:,:,None,None],np.where(visible[:,:,None,None]==1,self.d1,self.d0),self.hi)
        self.upper=np.einsum('ng,nhij->ghij',groups.astype(float),row)/self.sizes[:,None,None,None]
    def g(self):return np.maximum(0,self.upper.max(axis=(1,3))) # group x model, worst horizon and competitor
    def decision(self,eps=.005,globaleps=.0025):
        gs=self.g();g0=int(gs[0].argmin());g1=int(gs[1:].max(0).argmin());ok=(gs[1:]<=eps+TOL).all(0);g2=int(np.where(ok,gs[0],np.inf).argmin()) if ok.any() else -1
        return [(g0,gs[0,g0]<=globaleps+TOL),(g1,gs[1:,g1].max()<=eps+TOL),(g2,g2>=0)]
    def reveal(self,i,h,value):
        assert not self.attempted[i,h];self.attempted[i,h]=True
        if value is None:return
        assert value in (0,1);v=self.d1[i,h] if value else self.d0[i,h];delta=v-self.hi[i,h]
        for g in np.flatnonzero(self.groups[i]):self.upper[g,h]+=delta/self.sizes[g]
        self.seen[i,h]=True;self.visible[i,h]=value

def bottleneck(s,global_only=False):
    if global_only:gi=0;k=int(s.g()[0].argmin())
    else:
        gs=s.g();k=int(gs[1:].max(0).argmin());gi=1+int(gs[1:,k].argmax())
    h,j=np.unravel_index(np.argmax(s.upper[gi,:,k,:]),s.upper[gi,:,k,:].shape)
    eligible=(~s.attempted[:,h]) & s.groups[:,gi]
    if not eligible.any():
        ij=np.argwhere(~s.attempted);return tuple(ij[0]) if len(ij) else None
    scores=np.where(eligible,2*np.abs(s.p[:,h,k]-s.p[:,h,j])/s.sizes[gi],-np.inf);return int(scores.argmax()),h

def q3(s,q):
    """Exact H2 expected-lookahead Q3 on global max-horizon C1; no subgroup objective."""
    U=s.upper[0];current=s.g()[0].min();best=(-np.inf,None)
    for h in range(s.h):
        ix=np.flatnonzero(~s.attempted[:,h]);
        if not len(ix):continue
        other=np.max(np.delete(U,h,axis=0),axis=0) if s.h>1 else np.full((s.k,s.k),-np.inf)
        v=[]
        for d in [s.d0,s.d1]:
            new=U[h]+(d[ix,h]-s.hi[ix,h])/s.n
            v.append(np.maximum(new,other).max(2).min(1))
        scores=current-((1-q[ix,h])*v[0]+q[ix,h]*v[1]);a=int(scores.argmax())
        if scores[a]>best[0]:best=(scores[a],(int(ix[a]),h))
    return best[1] if best[0]>1e-15 else bottleneck(s,True)

def mask(d,age,rng,skew):
    # One random value per person and horizon; repeated windows are not independent respondents.
    ids,inv=np.unique(d.idind,return_inverse=True);u=rng.random((len(ids),3))[inv];prob=np.where(age==0,.4 if skew else .2,.2)
    return u>=prob[:,None]

def truth(p,y,groups):
    row=(p-y[:,:,None])**2;loss=np.einsum('ng,nhk->ghk',groups.astype(float),row)/groups.sum(0)[:,None,None]
    regret=loss-loss.min(2)[:,:,None];return loss,regret

def controlled_checks():
    rng=np.random.default_rng(771);p=rng.random((4,2,3));y=rng.integers(2,size=(4,2)).astype(float);gr=np.stack([np.ones(4,bool),np.array([1,1,0,0],bool),np.array([0,0,1,1],bool)],axis=1);visible=y.copy();visible.flat[[0,3,5]]=np.nan;s=State(p,visible,gr)
    upp=[]
    for fill in itertools.product([0,1],repeat=3):
        yy=visible.copy();yy.flat[[0,3,5]]=fill;ls,_=truth(p,yy,gr);upp.append(ls[:,:,:,None]-ls[:,:,None,:])
    np.testing.assert_allclose(s.upper,np.max(upp,axis=0),atol=1e-14)
    altered=y.copy();altered[np.isnan(visible)]=1-altered[np.isnan(visible)];s2=State(p,np.where(np.isfinite(visible),altered,np.nan),gr);np.testing.assert_allclose(s.upper,s2.upper)
    before=s.upper.copy();s.reveal(0,0,int(y[0,0]));assert np.all(s.upper<=before+TOL)
    ref=State(p,s.visible,gr);np.testing.assert_allclose(s.upper,ref.upper,atol=1e-14)
    # Opposing groups: no shared adequate candidate, even with all labels.
    pp=np.tile(np.array([.1,.9]),(4,1))[:,None,:];yy=np.array([0,0,1,1])[:,None];im=State(pp,yy,gr);assert im.decision()[0][1] and not im.decision()[1][1] and not im.decision()[2][1]
    third=np.array([.1,.1,.9,.9])[:,None,None];fixed=State(np.concatenate([pp,third],axis=2),yy,gr);assert fixed.decision()[1][1] and fixed.decision()[2][1]
    sf=State(p,visible,gr);before=sf.upper.copy();sf.reveal(0,0,None);np.testing.assert_array_equal(sf.upper,before);assert sf.attempted[0,0] and not sf.seen[0,0]
    # Compare Q3 with literal two-outcome enumeration on a small multi-horizon cohort.
    qq=np.full((4,2),.3);ss=State(p,visible,gr);scores={};current=ss.g()[0].min()
    for i,h in np.argwhere(~ss.attempted):
        values=[]
        for v in (0,1):
            rr=State(p,visible,gr);rr.reveal(int(i),int(h),v);values.append(rr.g()[0].min())
        scores[(int(i),int(h))]=current-(.7*values[0]+.3*values[1])
    pick=q3(ss,qq)
    if max(scores.values())>1e-15:assert abs(scores[pick]-max(scores.values()))<1e-12
    pa=np.sqrt(np.array([[.02,0.]]*9+[[.01,.04]]))[:,None,:];ya=np.zeros((10,1));ga=np.stack([np.ones(10,bool),np.arange(10)<9,np.arange(10)==9],axis=1);dd=State(pa,ya,ga).decision(eps=.035);assert dd[1][0]==0 and dd[2][0]==1
    return {'exhaustive_exact_bounds':'pass','hidden_label_invariance':'pass','incremental_update':'pass','impossible_requirement':'pass','global_certificate_can_fail_groups':'pass','added_candidate_restores_feasibility':'pass','failed_query_preserves_interval':'pass','q3_matches_two_outcome_enumeration':'pass','g1_and_g2_have_distinct_objectives':'pass'}

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);d=pd.read_csv(args.input);p,y,groups,labels,age=arrays(d);loss,reg=truth(p,y,groups);rows=[];audit=[];raw=[]
    for g,l in enumerate(labels):
        for h in range(3):
            for k,m in enumerate(NAMES):raw.append(dict(group=l,horizon=h+1,model=m,n=int(groups[:,g].sum()),small_group=groups[:,g].sum()<100,brier=loss[g,h,k],true_regret=reg[g,h,k]))
    pd.DataFrame(raw).to_csv(out/'full_information.csv',index=False)
    feasible_rows=[]
    for eps in (.0025,.005,.01):
        for k,m in enumerate(NAMES):feasible_rows.append(dict(epsilon=eps,model=m,true_max_group_regret=float(reg[1:,:,k].max()),feasible=bool(reg[1:,:,k].max()<=eps+TOL)))
    pd.DataFrame(feasible_rows).to_csv(out/'oracle_feasibility.csv',index=False)
    for skew in (False,True):
      for rep in range(args.masks):
        seen=mask(d,age,np.random.default_rng(10000+rep),skew);s=State(p,np.where(seen,y,np.nan),groups)
        for g,l in enumerate(labels):
            audit.append(dict(scenario='young_double_missing' if skew else 'uniform',mask=rep,group=l,n=int(groups[:,g].sum()),visible=int(seen[groups[:,g]].sum()),unknown_fraction=float(1-seen[groups[:,g]].mean())))
        for eps in (.0025,.005,.01):
          for method,(k,cert) in zip(('G0','G1','G2'),s.decision(eps)):
            oracle_ok=(reg[1:].max((0,1))<=eps+TOL).any()
            tr=float(reg[1:,:,k].max()) if k>=0 else np.nan
            if cert and method!='G0':assert tr<=eps+1e-10
            gg=s.g();partial=int((gg[1:,k]<=eps+TOL).sum()) if k>=0 else 0
            rows.append(dict(scenario='young_double_missing' if skew else 'uniform',mask=rep,epsilon=eps,method=method,model=NAMES[k] if k>=0 else '',certified=bool(cert),certified_groups=partial,total_groups=len(labels)-1,true_global_regret=float(reg[0,:,k].max()) if k>=0 else np.nan,true_max_group_regret=tr,oracle_feasible=bool(oracle_ok)))
    pd.DataFrame(rows).to_csv(out/'audit_decisions.csv',index=False);pd.DataFrame(audit).to_csv(out/'group_visibility.csv',index=False)
    pd.DataFrame(rows).groupby(['scenario','epsilon','method']).agg(certificate_rate=('certified','mean'),oracle_feasible=('oracle_feasible','mean'),max_true_group_regret=('true_max_group_regret','max')).reset_index().to_csv(out/'audit_summary.csv',index=False)
    print('full audit complete',len(d),flush=True)
    # Resource-bounded active experiment: deterministic person subsample chosen without labels.
    ids=sorted(d.idind.unique(),key=lambda v:hashlib.sha256(('H5-active:'+str(int(v))).encode()).hexdigest())[:args.people]
    z=d[d.idind.isin(ids)].sort_values(['idind','current_year']).groupby('idind',as_index=False).tail(1).reset_index(drop=True);p,y,groups,labels,age=arrays(z);loss,reg=truth(p,y,groups);runs=[];curves=[];q=np.stack([z[f'p_logistic_{h}'].to_numpy() for h in (1,2,3)],axis=1)
    for scenario in ['uniform','young_double_missing','MAR','MNAR']:
      for rep in range(args.active_masks):
        seen=mask(z,age,np.random.default_rng(20000+rep),scenario=='young_double_missing')
        # Fixed potential response of each query, shared by policies. An unsuccessful query costs 1.
        rr=np.random.default_rng(30000+rep).random(y.shape)
        propensity=np.ones(y.shape) if scenario in ('uniform','young_double_missing') else (np.where(age[:,None]==0,.55,.85)*np.ones(y.shape) if scenario=='MAR' else np.where(y==1,.85,.45))
        success=rr<propensity
        for policy in ['global_Q3','equal_age_random','group_bottleneck']:
          s=State(p,np.where(seen,y,np.nan),groups);rng=np.random.default_rng(40000+rep);count=np.zeros(3,int);hits=0;cost=0;first=None;initial=int((~seen).sum());snapshots={0,50,100,200,args.budget}
          while True:
            dec=s.decision();cert=dec[1][1]
            if cert and first is None:first=cost
            if cost in snapshots or cert or cost==args.budget or s.attempted.all():
              for method,(k,ok) in zip(('G0','G1','G2'),dec):
                tr=float(reg[1:,:,k].max()) if k>=0 else np.nan
                if ok and method!='G0':assert tr<=.005+1e-10
                curves.append(dict(scenario=scenario,mask=rep,policy=policy,cost=cost,method=method,certified=bool(ok),model=NAMES[k] if k>=0 else '',true_max_group_regret=tr))
            if cert or cost>=args.budget or s.attempted.all():break
            if policy=='global_Q3':item=q3(s,q)
            elif policy=='group_bottleneck':item=bottleneck(s)
            else:
                avail=[g for g in range(3) if ((~s.attempted)&(age[:,None]==g)).any()]
                g=min(avail,key=lambda g:(count[g],g));ij=np.argwhere((~s.attempted)&(age[:,None]==g));item=tuple(ij[rng.integers(len(ij))])
            if item is None:break
            i,h=item;value=int(y[i,h]) if success[i,h] else None;s.reveal(i,h,value);cost+=1;count[age[i]]+=1;hits+=int(value is not None)
          k,cert=s.decision()[1];oracle_g=reg[1:].max((0,1));g0=s.decision()[0]
          status='certified' if cert else ('no_acceptable_candidate' if s.seen.all() else 'insufficient_information')
          runs.append(dict(scenario=scenario,mask=rep,policy=policy,n_rows=len(z),people=len(ids),initial_hidden=initial,cost=cost,successful=hits,failed=cost-hits,certified=bool(cert),status=status,first_certificate_cost=first,oracle_feasible=bool((oracle_g<=.005+TOL).any()),true_max_group_regret=float(reg[1:,:,k].max()),global_certificate_group_failure=bool(g0[1] and reg[1:,:,g0[0]].max()>.005+TOL),queries_age18_34=int(count[0]),queries_age35_49=int(count[1]),queries_age50_64=int(count[2])))
        print('active',scenario,rep+1,flush=True)
    pd.DataFrame(runs).to_csv(out/'active_runs.csv',index=False);pd.DataFrame(curves).to_csv(out/'active_curves.csv',index=False)
    pd.DataFrame(runs).groupby(['scenario','policy']).agg(runs=('mask','count'),certificate_rate=('certified','mean'),mean_cost=('cost','mean'),mean_failed=('failed','mean'),oracle_feasible_rate=('oracle_feasible','mean')).reset_index().to_csv(out/'active_summary.csv',index=False)
    manifest={'full_rows':len(d),'full_people':int(d.idind.nunique()),'active_rows':len(z),'active_people':len(ids),'audit_masks':args.masks,'active_masks':args.active_masks,'active_budget':args.budget,'group_epsilon':.005,'global_epsilon':.0025,'query_unit':'one person-window-horizon binary label; cost=1 including failures','mask_unit':'shared person-horizon uniforms across repeated windows','groups':labels.tolist(),'checks':controlled_checks(),'scope':'retrospective corrected held-out experiment; earlier test results were already seen; no claim of pristine preregistration'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);ap.add_argument('--masks',type=int,default=100);ap.add_argument('--active-masks',type=int,default=10);ap.add_argument('--people',type=int,default=300);ap.add_argument('--budget',type=int,default=500);run(ap.parse_args())

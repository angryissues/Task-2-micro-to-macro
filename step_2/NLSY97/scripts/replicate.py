"""Frozen primary H1-H5 replication on NLSY97. H6 uses the separate transfer runner."""
import argparse,datetime,hashlib,importlib.util,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd

NAMES=['baseline','logistic','hgb'];EPS=.0025
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def latest(d):return d.sort_values(['idind','origin_week']).groupby('idind',as_index=False).tail(1).reset_index(drop=True)
def predictions(d,h):return d[[f'p_{m}_{h}' for m in NAMES]].to_numpy(float)
def value(U):
    g=U.max(1);k=int(g.argmin());return k,float(g[k]),bool(g[k]<=EPS+1e-12)
def rates(age,edu,y,kind,mean):
    if kind=='MCAR':return np.full(len(y),mean)
    z=.7*(age-age.mean())/max(age.std(),1e-10)+.5*edu
    if kind=='MNAR':z=z+1.5*(1-2*y)
    lo,hi=-30.,30.
    for _ in range(60):
        mid=(lo+hi)/2;p=1/(1+np.exp(-(z+mid)))
        if p.mean()<mean:lo=mid
        else:hi=mid
    return 1/(1+np.exp(-(z+(lo+hi)/2)))
def lookahead(state,q,kind,h2):
    ids=state._eligible_indices;p=state.predictions[ids];c=-2*state.weights[ids,None,None]*(p[:,:,None]-p[:,None,:]);v0=(state.upper-np.maximum(c,0)).max(2).min(1);v1=(state.upper-np.maximum(c,0)+c).max(2).min(1)
    post=np.maximum(v0,v1) if kind=='Q2' else (1-q[ids])*v0+q[ids]*v1;scores=state.value-post
    return int(h2.q1(state)) if np.all(scores<=1e-15) else int(h2._tie_break(ids,scores,state.costs))
def choose(state,name,q,rng,h2,strata,counts,answer=None):
    ids=state._eligible_indices
    if name=='random':return int(rng.choice(ids))
    if name=='stratified':
        available=np.unique(strata[ids]);g=min(available,key=lambda g:(counts.get(int(g),0),int(g)));c=ids[strata[ids]==g];i=int(rng.choice(c));counts[int(g)]=counts.get(int(g),0)+1;return i
    if name=='uncertainty':return int(h2.uncertainty_sampling(state,q))
    if name=='Q0':return int(h2.q0(state))
    if name=='Q1':return int(h2.q1(state))
    if name in ('Q2','Q3'):return lookahead(state,q,name,h2)
    if name=='Hara_adapter':return int(h2.hara_sampling(state,q,rng)[0])
    if name=='ActiveTesting_adapter':return int(h2.active_testing_brier_adapter(state,q,rng)[0])
    if name in ('response_weighted_Q1','biased_response_Q1'):
        k=state.worst_regret.argmin();j=state.upper[k].argmax();prob=answer[ids];prob=np.clip(1-prob,.05,.95) if name=='biased_response_Q1' else prob
        scores=np.abs(state.predictions[ids,k]-state.predictions[ids,j])*prob;return int(ids[np.argmax(scores)])
    raise ValueError(name)
def h1(d,h1,out,config):
    records=[]
    for h in (1,2,3):
        z=d[d[f'y{h}'].notna()].reset_index(drop=True);p=predictions(z,h);y=z[f'y{h}'].to_numpy();loss=((p-y[:,None])**2).mean(0);reg=loss-loss.min()
        for kind in config['H1']['mechanisms']:
          for frac in config['H1']['missing_fractions']:
            prob=rates(z.age.to_numpy(),z.education.fillna(-1).to_numpy()<4,y,kind,frac)
            for rep in range(config['H1']['masks']):
                rng=np.random.default_rng(41000+rep+h*1000);seen=rng.random(len(y))>=prob;public=np.where(seen,y,np.nan)
                audit_ids=rng.permutation(np.flatnonzero(~seen))[:50];labels=public.copy();labels[audit_ids]=y[audit_ids];audited=seen.copy();audited[audit_ids]=True
                for method in ('C0','C1','C1_same_audit','uniform_C3'):
                    if method=='C0':_,up,g=h1.c0_bounds(p,public,seen)
                    elif method=='C1':_,up,g=h1.c1_bounds(p,public,seen)
                    elif method=='C1_same_audit':_,up,g=h1.c1_bounds(p,labels,audited)
                    else:
                        res=h1.c3_bounds(p,labels,seen,audit_ids,alpha=.05/(3*3*3*config['H1']['masks']));up=res['upper'];g=up.max(1)
                    k=int(g.argmin());cert=bool(g[k]<=EPS+1e-12);wrong=bool(cert and reg[k]>EPS+1e-10)
                    if method!='uniform_C3':assert not wrong
                    records.append(dict(horizon=h,mechanism=kind,missing_fraction=frac,rep=rep,method=method,n=len(z),actual_hidden=int((~seen).sum()),paid_audit_labels=len(audit_ids) if method in ['C1_same_audit','uniform_C3'] else 0,model=NAMES[k],G=g[k],certified=cert,true_regret=reg[k],wrong_certificate=wrong))
    pd.DataFrame(records).to_csv(out/'H1_runs.csv',index=False)
    pd.DataFrame(records).groupby(['horizon','mechanism','missing_fraction','method']).agg(runs=('rep','count'),certificate_rate=('certified','mean'),wrong=('wrong_certificate','sum'),mean_G=('G','mean')).reset_index().to_csv(out/'H1_summary.csv',index=False)

def query_experiment(d,dev,h2,out,config,hypothesis):
    z=d[d[['y1','y2','y3']].notna().all(1)].reset_index(drop=True);p=predictions(z,3);y=z.y3.to_numpy();q=z.p_logistic_3.to_numpy();loss=((p-y[:,None])**2).mean(0);reg=loss-loss.min();strata=(z.age.to_numpy()>=35).astype(int);rows=[]
    cfg=config[hypothesis];scenarios=['complete_response'] if hypothesis=='H2' else cfg['response_scenarios'];seed=42000 if hypothesis=='H2' else 43000
    for scenario in scenarios:
      # Response probabilities are learned on simulated development responses using X only.
      from sklearn.linear_model import LogisticRegression
      dv=dev[dev.y3.notna()].copy();Xdev=np.column_stack([dv.age,dv.sex.fillna(0),dv.current_employment]);Xnew=np.column_stack([z.age,z.sex.fillna(0),z.current_employment]);rr=np.random.default_rng(4001)
      pp=np.full(len(dv),.8) if scenario in ['complete_response','MCAR'] else (np.where(dv.age<35,.55,.85) if scenario=='MAR' else np.where(dv.y3==1,.85,.45));respmodel=LogisticRegression(max_iter=2000).fit(Xdev,(rr.random(len(dv))<pp).astype(int));estimated=respmodel.predict_proba(Xnew)[:,1]
      for rep in range(cfg['masks']):
        rng=np.random.default_rng(seed+rep);seen=rng.random(len(y))>=.2;potential=rng.random(len(y));prop=np.ones(len(y)) if scenario=='complete_response' else (np.full(len(y),.8) if scenario=='MCAR' else (np.where(z.age<35,.55,.85) if scenario=='MAR' else np.where(y==1,.85,.45)));success=potential<prop
        for policy in cfg['policies']:
            s=h2.SelectionState.create(p,np.where(seen,y,0),seen);attempted=seen.copy();count={};cost=0;answers=0;prng=np.random.default_rng(seed+10000+rep);first=None
            while True:
                cert=s.certificate(EPS)
                if cert['certified'] or cost>=cfg['max_budget'] or attempted.all():break
                s._eligible_indices=np.flatnonzero(~attempted);i=choose(s,policy,q,prng,h2,strata,count,estimated);attempted[i]=True;cost+=1
                if success[i]:s.reveal(i,int(y[i]));answers+=1
            cert=s.certificate(EPS);k=cert['recommended_model'];assert not cert['certified'] or reg[k]<=EPS+1e-10
            rows.append(dict(scenario=scenario,rep=rep,policy=policy,n=len(y),initial_hidden=int((~seen).sum()),cost=cost,successful=answers,failed=cost-answers,certified=cert['certified'],G=cert['G'],model=NAMES[k],true_regret=reg[k],response_model='development_X_only'))
    pd.DataFrame(rows).to_csv(out/f'{hypothesis}_runs.csv',index=False);pd.DataFrame(rows).groupby(['scenario','policy']).agg(runs=('rep','count'),certificate_rate=('certified','mean'),mean_cost=('cost','mean'),mean_failed=('failed','mean'),mean_G=('G','mean')).reset_index().to_csv(out/f'{hypothesis}_summary.csv',index=False)

def natural_and_gamma(d,h1,out,config):
    p=predictions(d,3);y=d.y3.to_numpy();seen=np.isfinite(y);_,U,G=h1.c1_bounds(p,y,seen);k=int(G.argmin());natural={'n':len(d),'unknown':int((~seen).sum()),'model':NAMES[k],'G':float(G[k]),'certified':bool(G[k]<=EPS),'true_regret':None,'scope':'archival unresolved statuses include nonresponse/ambiguity/structural lack of followup; not all are item refusals'}
    (out/'H3_natural_bounds.json').write_text(json.dumps(natural,indent=2));records=[];z=d[d.y3.notna()].reset_index(drop=True);p=predictions(z,3);truth=z.y3.to_numpy();groups=(z.age.to_numpy()>=35).astype(int)
    for rep in range(50):
        rng=np.random.default_rng(43100+rep);prob=rates(z.age.to_numpy(),z.education.fillna(-1).to_numpy()<4,truth,'MNAR',.2);visible=rng.random(len(z))>=prob
        for gamma in (1,2,4):
            counts={};premise=True
            for group in np.unique(groups):
                m=(groups==group)&~visible;ob=(groups==group)&visible;mean=float(truth[ob].mean()) if ob.any() else .5;mean=np.clip(mean,1/(len(z)+1),1-1/(len(z)+1));odds=mean/(1-mean);lower=odds/gamma/(1+odds/gamma);upper=odds*gamma/(1+odds*gamma);A=int(np.floor(lower*m.sum()));B=int(np.ceil(upper*m.sum()));counts[int(group)]=(A,B);premise &= A<=int(truth[m].sum())<=B
            lo,up,g,status=h1.c2_bounds(p,np.where(visible,truth,np.nan),visible,groups,counts)
            if status=='ok':k=int(g.argmin());cert=bool(g[k]<=EPS);ls=((p-truth[:,None])**2).mean(0);wrong=bool(cert and ls[k]-ls.min()>EPS+1e-10)
            else:k=-1;cert=False;wrong=False
            if premise:assert not wrong
            records.append(dict(rep=rep,Gamma=gamma,premise_holds=bool(premise),status=status,certified=cert,wrong_certificate_unconditional=wrong,scope='conditional_count_sensitivity'))
    pd.DataFrame(records).to_csv(out/'H3_Gamma_sensitivity.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);ap.add_argument('--split',choices=['development','confirmation'],required=True);ap.add_argument('--hypotheses',nargs='+',default=['H1','H2','H3']);a=ap.parse_args();repo=Path(a.repo);root=Path(a.predictions);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);config=json.loads(Path(a.config).read_text());d=latest(pd.read_csv(root/f'{a.split}_predictions.csv.gz'));dev=latest(pd.read_csv(root/'development_predictions.csv.gz'));h1m=module('nls_h1',repo/'step_2/H1-RLMS/src/bounds.py');h2=module('nls_h2',repo/'step_2/H2-RLMS/src/active_selection.py')
    for hyp in a.hypotheses:
        print('START',a.split,hyp,flush=True)
        if hyp=='H1':h1(d,h1m,out,config)
        elif hyp in ('H2','H3'):query_experiment(d,dev,h2,out,config,hyp)
        if hyp=='H3':natural_and_gamma(d,h1m,out,config)
        print('DONE',a.split,hyp,flush=True)
    (out/'access_manifest.json').write_text(json.dumps({'split':a.split,'hypotheses':a.hypotheses,'config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'fixed predictions; no refitting or parameter selection on confirmation'},indent=2))
if __name__=='__main__':main()

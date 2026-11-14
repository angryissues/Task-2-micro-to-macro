"""H4/H5 public-information-only trajectory and subgroup experiments."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from replicate import module,latest,predictions,NAMES,EPS

BITS=np.array(list(itertools.product([0.,1.],repeat=3)));POW=np.array([1,3,9])
def cache_bounds(pred,target,mode):
    n,h,k=pred.shape;cache=np.zeros((n,27,k,k))
    for code in range(27):
        digits=(code//POW)%3;obs=np.where(digits==0,np.nan,digits-1);admissible=BITS[np.all(np.isnan(obs)|(BITS==obs),axis=1)]
        outcomes=np.maximum.accumulate(admissible==0,axis=1).astype(float) if target=='first_exit' else admissible
        if mode=='P0' and target=='first_exit':outcomes=np.array(list(itertools.product(*[np.unique(outcomes[:,j]) for j in range(3)])))
        if target=='endpoint':ls=(pred[:,None,-1,:]-outcomes[None,:,-1,None])**2
        else:ls=((pred[:,None,:,:]-outcomes[None,:,:,None])**2).mean(2)
        pair=ls[:,:,:,None]-ls[:,:,None,:];cache[:,code]=pair.max(1)
    return cache
def best_action(cur,U,cache,policy,price,remaining):
    """Only public states/predictions are accepted; true answers cannot enter scoring."""
    n=len(cur);candidates=[]
    for i,code in enumerate(cur):
        hidden_idx=np.flatnonzero((code//POW)%3==0)
        if not len(hidden_idx):continue
        actions=[]
        if policy in ['point','mixed'] and remaining>=1-1e-12:actions.extend([(np.array([h]),1.,'point') for h in hidden_idx])
        if policy in ['block','mixed'] and remaining>=price-1e-12:actions.append((hidden_idx,float(price),'block'))
        for ids,fee,kind in actions:
            postcodes=[code+sum((bit+1)*POW[h] for bit,h in zip(fill,ids)) for fill in itertools.product([0,1],repeat=len(ids))]
            posts=U+(cache[i,postcodes]-cache[i,code])/n;worst=float(posts.max(2).min(1).max());width=float(posts.sum((1,2)).max());gain=(float(U.max(1).min())-worst)/fee;secondary=(U.sum()-width)/fee;candidates.append((gain,secondary,-fee,-i,-int(ids[0]),i,ids,fee))
    return max(candidates,key=lambda x:x[:5])[5:] if candidates else None

def h4(d,out,cfg):
    c=d[(d.current_employment==1)&d[['y1','y2','y3']].notna().all(1)].copy();c['hash']=c.idind.map(lambda x:hashlib.sha256(('NLSY97-H4:'+str(int(x))).encode()).hexdigest());c=c.sort_values('hash').head(cfg['people']).reset_index(drop=True);n=len(c);y=c[['y1','y2','y3']].to_numpy();records=[];query=[]
    ann=np.stack([predictions(c,h) for h in (1,2,3)],axis=1);ex=np.stack([c[[f'p_{m}_exit{h}' for m in NAMES]].to_numpy() for h in (1,2,3)],axis=1);cache=cache_bounds(ex,'first_exit','P1');allknown=((y+1)*POW).sum(1).astype(int);trueU=cache[np.arange(n),allknown].mean(0);trueG=trueU.max(1)
    other={(t,m):cache_bounds(ex if t=='first_exit' else ann,t,m) for t in ['endpoint','multi_horizon','first_exit'] for m in ['P0','P1']}
    for rep in range(cfg['masks']):
      hidden=np.random.default_rng(44000+rep).random(y.shape)<.2;codes=np.where(hidden,0,y+1).astype(int)@POW
      for target in ['endpoint','multi_horizon','first_exit']:
        vals={m:other[target,m][np.arange(n),codes].mean(0) for m in ['P0','P1']};assert np.all(vals['P1']<=vals['P0']+1e-12)
        for mode,U in vals.items():G=U.max(1);records.append(dict(rep=rep,target=target,mode=mode,n=n,hidden_points=int(hidden.sum()),G=float(G.min()),certified=bool(G.min()<=EPS),max_pair_width=float(np.max(U+U.T))))
      for price in cfg['block_prices']:
       for policy in ['point','block','mixed']:
        cur=codes.copy();U=cache[np.arange(n),cur].mean(0);cost=0.;ops=0
        while U.max(1).min()>EPS+1e-12:
            item=best_action(cur,U,cache,policy,price,cfg['P2_budget']-cost)
            if item is None:break
            i,ids,fee=item;old=cur[i];cur[i]+=int(((y[i,ids]+1)*POW[ids]).sum());U+=(cache[i,cur[i]]-cache[i,old])/n;cost+=fee;ops+=1
        k=int(U.max(1).argmin());G=float(U.max(1)[k]);cert=G<=EPS+1e-12;assert not cert or trueG[k]<=EPS+1e-10
        query.append(dict(rep=rep,block_price=price,policy=policy,n=n,cost=cost,operations=ops,G=G,certified=cert,true_regret=trueG[k],scope='first_exit_on_three_future_control_points'))
    pd.DataFrame(records).to_csv(out/'H4_P0_P1.csv',index=False);pd.DataFrame(query).to_csv(out/'H4_P2_runs.csv',index=False);pd.DataFrame(query).groupby(['block_price','policy']).agg(runs=('rep','count'),certificate_rate=('certified','mean'),mean_cost=('cost','mean'),mean_G=('G','mean')).reset_index().to_csv(out/'H4_P2_summary.csv',index=False)

def groups_for(d):
    age=d.age.to_numpy();edu=d.education.to_numpy();vals=[np.ones(len(d),bool),(age>=18)&(age<30),(age>=30)&(age<35),(age>=35)&(age<40),(age>=40)&(age<45),d.sex.to_numpy()==1,d.sex.to_numpy()==2,np.isfinite(edu)&(edu<4),np.isfinite(edu)&(edu>=4),~np.isfinite(edu)];labels=np.array(['all','age18_29','age30_34','age35_39','age40_44','male','female','no_bachelor','bachelor_plus','education_unknown']);mat=np.stack(vals,axis=1);keep=mat.sum(0)>0
    return mat[:,keep],labels[keep],np.where(age<30,0,np.where(age<35,1,np.where(age<40,2,3)))
def h5(d,out,cfg,h5module):
    d=d[d[['y1','y2','y3']].notna().all(1)].reset_index(drop=True);p=np.stack([predictions(d,h) for h in (1,2,3)],axis=1);y=d[['y1','y2','y3']].to_numpy();groups,labels,age=groups_for(d);ls=np.einsum('ng,nhk->ghk',groups.astype(float),(p-y[:,:,None])**2)/groups.sum(0)[:,None,None];reg=ls-ls.min(2)[:,:,None];full=[]
    for g,label in enumerate(labels):
      for h in range(3):
       for k,m in enumerate(NAMES):full.append(dict(group=label,n=int(groups[:,g].sum()),small_group=groups[:,g].sum()<100,horizon=h+1,model=m,brier=ls[g,h,k],true_regret=reg[g,h,k]))
    pd.DataFrame(full).to_csv(out/'H5_full_information.csv',index=False);rows=[];decisions=[]
    youngest=int(np.min(age))
    for scenario in ['uniform','young_double_missing','MAR','MNAR']:
      for rep in range(cfg['masks']):
        rng=np.random.default_rng(45000+rep);prob=np.where(age[:,None]==youngest,.4,.2) if scenario=='young_double_missing' else np.full(y.shape,.2);seen=rng.random(y.shape)>=prob;response=np.ones(y.shape) if scenario in ['uniform','young_double_missing'] else (np.where(age[:,None]==youngest,.55,.85)*np.ones(y.shape) if scenario=='MAR' else np.where(y==1,.85,.45));success=rng.random(y.shape)<response
        for policy in cfg['policies']:
            s=h5module.State(p,np.where(seen,y,np.nan),groups);cost=0;answers=0;counts=np.zeros(4,int);prng=np.random.default_rng(45100+rep)
            while True:
                dec=s.decision();k,cert=dec[1]
                if cert or cost>=cfg['max_budget'] or s.attempted.all():break
                if policy=='global_Q3':item=h5module.q3(s,np.stack([d[f'p_logistic_{h}'] for h in (1,2,3)],axis=1))
                elif policy=='group_bottleneck':item=h5module.bottleneck(s)
                else:
                    avail=[g for g in range(4) if ((~s.attempted)&(age[:,None]==g)).any()];g=min(avail,key=lambda g:(counts[g],g));items=np.argwhere((~s.attempted)&(age[:,None]==g));item=tuple(items[prng.integers(len(items))])
                if item is None:break
                i,h=item;ans=int(y[i,h]) if success[i,h] else None;s.reveal(i,h,ans);cost+=1;answers+=int(ans is not None);counts[age[i]]+=1
            for method,(k,cert) in zip(['G0','G1','G2'],s.decision()):
                tr=float(reg[1:,:,k].max()) if k>=0 else np.nan
                if cert and method!='G0':assert tr<=.005+1e-10
                decisions.append(dict(scenario=scenario,rep=rep,policy=policy,method=method,model=NAMES[k] if k>=0 else '',certified=bool(cert),true_max_group_regret=tr))
            k,cert=s.decision()[1];rows.append(dict(scenario=scenario,rep=rep,policy=policy,n=len(d),cost=cost,successful=answers,failed=cost-answers,certified=bool(cert),G=float(s.g()[1:,k].max()),true_max_group_regret=float(reg[1:,:,k].max()),oracle_feasible=bool((reg[1:].max((0,1))<=.005+1e-12).any()),status='certified' if cert else ('no_acceptable_candidate' if s.seen.all() else 'insufficient_information'),queries_age18_29=int(counts[0]),queries_age30_34=int(counts[1]),queries_age35_39=int(counts[2]),queries_age40_44=int(counts[3])))
    pd.DataFrame(rows).to_csv(out/'H5_runs.csv',index=False);pd.DataFrame(decisions).to_csv(out/'H5_G0_G1_G2.csv',index=False);pd.DataFrame(rows).groupby(['scenario','policy']).agg(runs=('rep','count'),certificate_rate=('certified','mean'),mean_cost=('cost','mean'),mean_failed=('failed','mean'),oracle_feasible=('oracle_feasible','mean')).reset_index().to_csv(out/'H5_summary.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);ap.add_argument('--split',choices=['development','confirmation'],required=True);ap.add_argument('--hypotheses',nargs='+',default=['H4','H5']);a=ap.parse_args();repo=Path(a.repo);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=latest(pd.read_csv(Path(a.predictions)/f'{a.split}_predictions.csv.gz'));config=json.loads(Path(a.config).read_text());h5module=module('nls_h5state',repo/'step_2/H5-RLMS/scripts/analysis.py')
    for h in a.hypotheses:
        print('START',a.split,h,flush=True)
        if h=='H4':h4(d,out,config['H4'])
        elif h=='H5':h5(d,out,config['H5'],h5module)
        print('DONE',a.split,h,flush=True)
if __name__=='__main__':main()

"""Recalculate oracle-free P2 on old frozen forecasts, plus unique-person future cohorts."""
import argparse,hashlib,itertools,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.block_queries import TrajectoryOracle,run_queries

MODELS=['baseline','logistic','hgb']
def predictions(x,target):
    suffix='exit' if target=='first_exit' else 'h'
    return np.stack([x[[f'p_{m}_{suffix}{h}' for m in MODELS]].to_numpy() for h in (1,2,3)],axis=1)
def cache_upper(pred,target,mode):
    n,h,k=pred.shape;paths=np.array(list(itertools.product([0.,1.],repeat=h)));power=3**np.arange(h);cache=np.zeros((n,3**h,k,k))
    for code in range(3**h):
        digits=(code//power)%3;admissible=paths[np.all((digits==0)|(paths==digits-1),axis=1)];outcome=np.maximum.accumulate(admissible==0,axis=1).astype(float) if target=='first_exit' else admissible
        if mode=='p0' and target=='first_exit':outcome=np.array(list(itertools.product(*[np.unique(outcome[:,j]) for j in range(h)])))
        ls=(pred[:,None,-1,:]-outcome[None,:,-1,None])**2 if target=='endpoint' else ((pred[:,None,:,:]-outcome[None,:,:,None])**2).mean(2)
        cache[:,code]=(ls[:,:,:,None]-ls[:,:,None,:]).max(1)
    return cache
def information(x,masks,seed):
    truth=x[['e1','e2','e3']].to_numpy();n=len(x);power=3**np.arange(3);rows=[]
    for target in ['endpoint','multi_horizon','first_exit']:
        pred=predictions(x,target);cache={mode:cache_upper(pred,target,mode) for mode in ['p0','p1']}
        for rep in range(masks):
            hidden=np.random.default_rng(seed+rep).random(truth.shape)<.2;codes=np.where(hidden,0,truth+1).astype(int)@power;U={m:cache[m][np.arange(n),codes].mean(0) for m in cache};assert np.all(U['p1']<=U['p0']+1e-12)
            for mode,u in U.items():rows.append(dict(mask_seed=rep,target=target,mode=mode,n=n,G=float(u.max(1).min()),max_pair_width=float(np.max(u+u.T)),certified=bool(u.max(1).min()<=.0025+1e-12)))
    return rows
def queries(x,name,masks,prices,budget,seed,epsilon):
    truth=x[['e1','e2','e3']].to_numpy(float);pred=predictions(x,'first_exit');exit_truth=np.maximum.accumulate(truth==0,axis=1).astype(float);loss=((pred-exit_truth[:,:,None])**2).mean((0,1));regret=loss-loss.min();rows=[]
    for rep in range(masks):
        public=truth.copy();public[np.random.default_rng(seed+rep).random(truth.shape)<.2]=np.nan
        for price in prices:
          for mode in ['point_only','block_only','mixed']:
            res=run_queries(pred,public,TrajectoryOracle(truth),mode,price,budget,epsilon);wrong=bool(res['certified'] and regret[res['model']]>epsilon+1e-10);assert not wrong
            rows.append(dict(study=name,mask_seed=rep,block_cost=price,mode=mode,n=len(x),budget=budget,initial_hidden=int(np.isnan(public).sum()),initial_G=res['initial_G'],initially_certified=res['initially_certified'],operations=res['operations'],point_queries=sum(t['kind']=='point' for t in res['trace']),block_queries=sum(t['kind']=='block' for t in res['trace']),total_cost=res['total_cost'],final_G=res['G'],certified=res['certified'],model=MODELS[res['model']],true_regret=float(regret[res['model']]),wrong_certificate=wrong))
        if (rep+1)%10==0:print(name,'mask',rep+1,flush=True)
    return rows
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True);x=pd.read_csv(a.input);x=x[x.observed_path.astype(str).str.lower().eq('true')].copy();start=time.monotonic();rows=[]
    for case in cfg['legacy_replays']:
        z=x.head(case['n']);rows+=queries(z,case['name'],case['masks'],case['prices'],case['budget'],cfg['mask_seed_base'],cfg['epsilon']);print('replayed',case['name'],flush=True)
    pd.DataFrame(rows).to_csv(out/'legacy_replay_runs.csv',index=False)
    legacy_info=information(x.head(1000),1,20261116);pd.DataFrame(legacy_info).to_csv(out/'legacy_P0_P1_corrected.csv',index=False)
    c=cfg['main'];eligible=x[(x.split=='test')&(x.start_year>=c['start_year_min'])&(x.start_year+3<=c['end_year_max'])&(x.start_adult==1)&(x.start_employment==1)&x.age.ge(18)&x.age.lt(65)].sort_values(['idind','start_year']).groupby('idind',as_index=False).tail(1).copy();eligible['hash']=eligible.idind.map(lambda v:hashlib.sha256(('RLMS-H4-correction-v2:'+str(int(v))).encode()).hexdigest());z=eligible.sort_values('hash').head(c['people_max']).reset_index(drop=True)
    assert z.idind.is_unique and set(z.idind).isdisjoint(set(x.loc[x.split=='train','idind']));assert (x.loc[x.split=='train','start_year']+3).max()<=2018
    mainrows=queries(z,'future_unique_person',c['masks'],cfg['block_prices'],c['budget'],cfg['mask_seed_base'],cfg['epsilon']);pd.DataFrame(mainrows).to_csv(out/'future_unique_person_runs.csv',index=False);pd.DataFrame(information(z,c['masks'],cfg['mask_seed_base'])).to_csv(out/'future_P0_P1.csv',index=False)
    allrows=pd.DataFrame(rows+mainrows);summary=allrows.groupby(['study','block_cost','mode']).agg(runs=('mask_seed','count'),certificate_rate=('certified','mean'),mean_cost=('total_cost','mean'),mean_G=('final_G','mean'),mean_operations=('operations','mean')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    paired=[];d=pd.DataFrame(mainrows)
    for price in cfg['block_prices']:
      frame=d[d.block_cost==price]
      for mode in ['block_only','mixed']:
        other=frame[frame['mode']==mode].set_index('mask_seed');point=frame[frame['mode']=='point_only'].set_index('mask_seed');both=other.certified&point.certified;delta=point.loc[both,'total_cost']-other.loc[both,'total_cost'];paired.append(dict(block_cost=price,mode=mode,paired_success_masks=int(both.sum()),mean_saved_cost_if_both_success=float(delta.mean()) if len(delta) else np.nan,cheaper_fraction_if_both_success=float((delta>0).mean()) if len(delta) else np.nan))
    pd.DataFrame(paired).to_csv(out/'paired_success_comparison.csv',index=False)
    root=Path(__file__).resolve().parents[1];source=[root/'src/block_queries.py',root/'src/trajectory_bounds.py',Path(__file__),root/'scripts/run_p2_block_queries.py'];manifest={'config':cfg,'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},'predictions_refit':False,'eligible_unique_people':len(eligible),'main_people':len(z),'main_future_year_min':int(z.start_year.min()+1),'main_future_year_max':int(z.start_year.max()+3),'main_person_disjoint_from_model_train':True,'legacy_runs':len(rows),'main_runs':len(mainrows),'false_certificates':int(allrows.wrong_certificate.sum()),'elapsed_seconds':time.monotonic()-start,'scope':'retrospective correction, legacy replay plus unique-person future cohort; scenario prices'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(summary[summary.study=='future_unique_person'].to_string(index=False),flush=True)
if __name__=='__main__':main()

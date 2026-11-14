"""CODA query ordering with common C1; separate post-primary extension."""
import argparse,contextlib,datetime,hashlib,io,json,platform,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def configure(repo,author):
    sys.path.insert(0,str(Path(repo)/'step_2/NLSY97/scripts'))
    from replicate import module
    h2=module('coda_extension_h2',Path(repo)/'step_2/H2-RLMS/src/active_selection.py')
    sys.path.insert(0,str(author))
    from coda.coda import CODA
    return h2,CODA

class CodaPolicy:
    def __init__(self,p,public,conditioned,settings,seed,CODA):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.selector=CODA(SimpleNamespace(preds=torch.tensor(np.stack([1-p.T,p.T],axis=-1),dtype=torch.float32)),**settings)
        seen=np.isfinite(public)
        if conditioned:
            for i in np.flatnonzero(seen):self.selector.add_label(int(i),int(public[i]),0.)
        else:self.selector.unlabeled_idxs=np.flatnonzero(~seen).tolist()
        assert self.selector.unlabeled_idxs==np.flatnonzero(~seen).tolist()
    def choose(self):
        with contextlib.redirect_stderr(io.StringIO()):
            i,score=self.selector.get_next_item_to_label()
        return int(i),float(score)
    def reveal(self,i,label,score):self.selector.add_label(i,label,score)

def run_policy(p,public,oracle,policy,cfg,seed,h2,CODA):
    seen=np.isfinite(public);s=h2.SelectionState.create(p,np.where(seen,public,0),seen)
    selector=CodaPolicy(p,public,policy=='CODA_conditioned',cfg['coda'],seed,CODA) if policy.startswith('CODA') else None
    rng=np.random.default_rng(seed);snapshots={};trace=[];cost=0;budgets=cfg['budgets']
    while True:
        d=s.certificate(cfg['epsilon'])
        for b in budgets:
            if b==cost or (d['certified'] and b>=cost):
                snapshots.setdefault(b,dict(budget=b,cost=cost,certified=d['certified'],G=d['G'],model_index=d['recommended_model']))
        if d['certified'] or cost>=max(budgets) or not len(s.hidden_indices):break
        if selector:i,score=selector.choose()
        elif policy=='random':i=int(rng.choice(s.hidden_indices));score=None
        elif policy=='Q2':
            from replicate import lookahead
            s._eligible_indices=s.hidden_indices
            i=lookahead(s,p[:,1],'Q2',h2);score=None
        else:raise ValueError(policy)
        assert not s.observed[i]
        answer=oracle.query(i)  # Only after selection, never passed to policy.
        s.reveal(i,answer)
        if selector:selector.reveal(i,answer,score)
        trace.append((i,answer));cost+=1
    for b in budgets:snapshots.setdefault(b,dict(budget=b,cost=cost,certified=d['certified'],G=d['G'],model_index=d['recommended_model']))
    return [snapshots[b] for b in budgets],trace

def main():
    ap=argparse.ArgumentParser()
    for name in ['repo','author','predictions','reference','out']:ap.add_argument('--'+name,required=True)
    ap.add_argument('--mask-start',type=int,default=0);ap.add_argument('--mask-count',type=int);a=ap.parse_args()
    root=Path(__file__).parent;cfg=json.loads((root/'protocol.json').read_text());lock=json.loads((root/'extension_lock.json').read_text())
    assert digest(root/'protocol.json')==lock['config_sha256'];assert digest(__file__)==lock['runner_sha256']
    for name,h in lock['author_sha256'].items():assert digest(Path(a.author)/name)==h,name
    for name,h in lock['reference_sha256'].items():assert digest(Path(a.repo)/name)==h,name
    torch.set_num_threads(cfg['torch_threads']);h2,CODA=configure(a.repo,a.author)
    from replicate import latest,predictions
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    event=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),lock_sha256=digest(root/'extension_lock.json'),scope=cfg['scope'],input_sha256=digest(a.predictions))
    (out/f'access_{a.mask_start:03d}.json').write_text(json.dumps(event,indent=2))
    d=latest(pd.read_csv(a.predictions));d=d[d[['y1','y2','y3']].notna().all(axis=1)].reset_index(drop=True);assert d.idind.is_unique
    p=predictions(d,3);y=d.y3.to_numpy();loss=((p-y[:,None])**2).mean(0);regret=loss-loss.min()
    ref=pd.read_csv(a.reference);rows=[];started=time.monotonic();stop=cfg['masks'] if a.mask_count is None else min(cfg['masks'],a.mask_start+a.mask_count)
    assert 0<=a.mask_start<stop<=cfg['masks']
    for rep in range(a.mask_start,stop):
        seen=np.random.default_rng(cfg['mask_seed']+rep).random(len(y))>=cfg['missing_fraction'];public=np.where(seen,y,np.nan)
        for policy in cfg['policies']:
            snaps,_=run_policy(p,public,h2.LabelOracle(y),policy,cfg,cfg['policy_seed']+rep,h2,CODA)
            for s in snaps:
                k=s['model_index'];wrong=bool(s['certified'] and regret[k]>cfg['epsilon']+1e-10);assert not wrong
                rows.append(dict(rep=rep,policy=policy,n=len(y),initial_hidden=int((~seen).sum()),**{k:v for k,v in s.items() if k!='model_index'},model=cfg['models'][k],true_regret=float(regret[k]),wrong_certificate=wrong))
            if policy in ('random','Q2'):
                orig=ref[(ref.rep==rep)&(ref.policy==policy)].iloc[0];v=snaps[-1]
                assert v['cost']==orig.cost and v['certified']==orig.certified and abs(v['G']-orig.G)<1e-10,(rep,policy)
        pd.DataFrame(rows).to_csv(out/f'runs_{a.mask_start:03d}.csv',index=False)
        print(f"mask {rep+1}/{cfg['masks']}, elapsed {time.monotonic()-started:.1f}s",flush=True)
    meta={**event,'n':len(y),'mask_start':a.mask_start,'mask_stop':stop,'elapsed_seconds':time.monotonic()-started,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'original_random_Q2_match':True,'false_certificates':sum(row['wrong_certificate'] for row in rows)}
    (out/f'manifest_{a.mask_start:03d}.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()


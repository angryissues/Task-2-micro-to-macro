#!/usr/bin/env python3
"""Small controlled P2 block-vs-point experiment for H4 first-exit target."""
import argparse, itertools, csv
from pathlib import Path
import numpy as np, pandas as pd
from src.trajectory_bounds import bounds

def score(pred, public):
    _,_,g=bounds(pred,public,target='first_exit',mode='p1'); return float(g.min())
def apply(public, truth, ids, vals):
    z=public.copy(); z[ids]=vals; return z
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--n',type=int,default=100); ap.add_argument('--budget',type=float,default=20); ap.add_argument('--masks',type=int,default=1); ap.add_argument('--block-costs',default='1.5,3,4.5'); a=ap.parse_args()
    block_costs=[float(v) for v in a.block_costs.split(',')]
    x=pd.read_csv(a.input); x=x[x.observed_path].head(a.n).copy(); truth=x[['e1','e2','e3']].to_numpy(float)
    pred=np.stack([np.column_stack([x[f'p_{m}_exit{h}'].to_numpy(float) for h in (1,2,3)]) for m in ('baseline','logistic','hgb')],axis=2)
    rows=[]
    for mask_seed in range(a.masks):
      rng=np.random.default_rng(20261117+mask_seed); public=truth.copy(); public[rng.random(public.shape)<.2]=np.nan
      for block_cost in block_costs:
       for mode in ('point_only','block_only','mixed'):
        z=public.copy(); spent=0.; ops=0
        while spent < a.budget:
          current=score(pred,z); candidates=[]
          for i in range(len(z)):
            hidden=np.flatnonzero(np.isnan(z[i]))
            if mode in ('point_only','mixed'):
              for h in hidden:
                if spent+1<=a.budget:
                  zz=z.copy(); zz[i,h]=truth[i,h]; gain=current-score(pred,zz); candidates.append((gain/1.,'point',i,[h],1.))
            if mode in ('block_only','mixed') and len(hidden) and spent+block_cost<=a.budget:
              # worst-case block outcome over all binary assignments
              worst=-1.
              for vals in itertools.product((0.,1.),repeat=len(hidden)):
                zz=z.copy(); zz[i,hidden]=vals; worst=max(worst,score(pred,zz))
              gain=current-worst; candidates.append((gain/block_cost,'block',i,hidden.tolist(),block_cost))
          if not candidates: break
          _,kind,i,ids,cost=max(candidates,key=lambda q:q[0]); z[i,ids]=truth[i,ids]; spent+=cost; ops+=1
          if score(pred,z)<=.0025: break
        rows.append({'mask_seed':mask_seed,'block_cost':block_cost,'mode':mode,'n':len(z),'initial_hidden':int(np.isnan(public).sum()),'operations':ops,'total_cost':spent,'final_G':score(pred,z),'certified':score(pred,z)<=.0025})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()

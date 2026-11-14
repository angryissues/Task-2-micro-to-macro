#!/usr/bin/env python3
"""Corrected P2 replay: point and block scores use only public information."""
import argparse
from pathlib import Path
import sys
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.block_queries import TrajectoryOracle,run_queries
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--n',type=int,default=100); ap.add_argument('--budget',type=float,default=20); ap.add_argument('--masks',type=int,default=1); ap.add_argument('--block-costs',default='1.5,3,4.5'); a=ap.parse_args()
    block_costs=[float(v) for v in a.block_costs.split(',')]
    x=pd.read_csv(a.input); x=x[x.observed_path.astype(str).str.lower().eq('true')].head(a.n).copy(); truth=x[['e1','e2','e3']].to_numpy(float)
    pred=np.stack([np.column_stack([x[f'p_{m}_exit{h}'].to_numpy(float) for h in (1,2,3)]) for m in ('baseline','logistic','hgb')],axis=2)
    rows=[]
    for mask_seed in range(a.masks):
      rng=np.random.default_rng(20261117+mask_seed); public=truth.copy(); public[rng.random(public.shape)<.2]=np.nan
      for block_cost in block_costs:
       for mode in ('point_only','block_only','mixed'):
        res=run_queries(pred,public,TrajectoryOracle(truth),mode,block_cost,a.budget)
        rows.append({'mask_seed':mask_seed,'block_cost':block_cost,'mode':mode,'n':len(public),'initial_hidden':int(np.isnan(public).sum()),'operations':res['operations'],'total_cost':res['total_cost'],'final_G':res['G'],'certified':res['certified'],'initially_certified':res['initially_certified'],'policy':'public_minimax_v2'})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()

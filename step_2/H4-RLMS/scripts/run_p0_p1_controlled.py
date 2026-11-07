#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import numpy as np
import pandas as pd
from src.trajectory_bounds import bounds
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--n',type=int,default=1000); ap.add_argument('--missing',type=float,default=.2); a=ap.parse_args()
    x=pd.read_csv(a.input); x=x[x.observed_path].head(a.n).copy(); rng=np.random.default_rng(20261116)
    obs=x[['e1','e2','e3']].to_numpy(float); hidden=rng.random(obs.shape)<a.missing; public=obs.copy(); public[hidden]=np.nan
    rows=[]
    for target,prefix in [('endpoint','p_hgb_h'),('multi_horizon','p_hgb_h'),('first_exit','p_hgb_exit')]:
      suffix='h' if prefix=='p_hgb_h' else 'exit'
      pred=np.stack([np.column_stack([x[f'p_{m}_{suffix}{h}'].to_numpy(float) for h in (1,2,3)]) for m in ('baseline','logistic','hgb')],axis=2)
      for mode in ('p0','p1'):
        lo,up,g=bounds(pred,public,target=target,mode=mode); width=np.max((up-lo)[~np.eye(up.shape[0],dtype=bool)]); rows.append({'target':target,'mode':mode,'n':len(x),'hidden_points':int(hidden.sum()),'G':float(g.min()),'max_pair_width':float(width),'certified_eps_0025':bool(g.min()<=.0025)})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()

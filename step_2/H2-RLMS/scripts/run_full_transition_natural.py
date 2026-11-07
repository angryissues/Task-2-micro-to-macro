#!/usr/bin/env python3
"""Run H2 policies on the full transition input with natural unknowns only."""
import argparse, csv
from pathlib import Path
import numpy as np
from src.active_selection import LabelOracle, SelectionState, q1, q2, random_policy, run_policy

def load(path):
    with open(path, newline='') as f: rows=list(csv.DictReader(f))
    p=np.array([[float(r['p_baseline']),float(r['p_logistic']),float(r['p_history'])] for r in rows])
    y=np.array([0 if r['target']=='' else int(float(r['target'])) for r in rows])
    observed=np.array([r['observed'].lower()=='true' for r in rows])
    return p,y,observed

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--budgets',default='0,50,100,200'); ap.add_argument('--epsilon',type=float,default=.0025)
    a=ap.parse_args(); p,y,obs=load(a.input); out=[]
    policies={'random':random_policy,'q1':q1,'q2':q2}
    for name,policy in policies.items():
      for budget in [float(x) for x in a.budgets.split(',')]:
        state=SelectionState.create(p,y,obs,costs=np.ones(len(y)))
        res=run_policy(state,LabelOracle(y),policy,a.epsilon,budget,rng=np.random.default_rng(20261115))
        out.append({'policy':name,'n_rows':len(y),'n_observed':int(obs.sum()),'n_unknown':int((~obs).sum()),
                    'budget':budget,'epsilon':a.epsilon,'status':res['status'],'certified':res['certified'],
                    'recommended_model':res['recommended_model'],'G':res['G'],'queries_obtained':res['queries_obtained'],
                    'total_cost':res['total_cost']})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print(f'wrote {len(out)} runs to {a.output}')
if __name__=='__main__': main()

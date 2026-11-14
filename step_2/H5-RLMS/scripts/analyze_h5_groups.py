#!/usr/bin/env python3
"""H5 full-cohort subgroup audit using frozen H1 predictions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=['p_baseline','p_logistic','p_hist_gradient_boosting']

def bounds(p, y, seen):
    # Exact C1 bounds for pairwise Brier differences within this subgroup.
    k=p.shape[1]; upper=np.zeros((k,k))
    for a in range(k):
        for b in range(k):
            if a==b: continue
            d=p[:,a]-p[:,b]
            row=p[:,a]**2-p[:,b]**2
            total=row.sum() + (2*d[seen]*y[seen]).sum()
            total += np.maximum(2*d[~seen],0).sum()
            upper[a,b]=total/len(y)
    return upper.max(axis=1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    d=pd.read_csv(a.input); d=d[d.target.notna()].copy(); d['target']=d.target.astype(int)
    d['age_group']=pd.cut(d.age, [17.999,34,49,64,np.inf], labels=['18-34','35-49','50-64','65+'])
    d['sex_group']=d.sex.map({1:'sex_code_1',2:'sex_code_2'}).fillna('unknown')
    # RLMS educ codes are retained individually; no unsupported binary mapping is assumed.
    d['education_group']=d.education.map(lambda x: f'educ_code_{int(x)}' if pd.notna(x) else 'unknown')
    p=d[MODELS].to_numpy(); y=d.target.to_numpy(); losses=((p-y[:,None])**2).mean(axis=0)
    global_winner=int(losses.argmin()); rows=[]
    for dimension in ['age_group','sex_group','education_group']:
        for label,g in d.groupby(dimension, observed=True):
            if len(g)==0: continue
            pg=g[MODELS].to_numpy(); yg=g.target.to_numpy(); ls=((pg-yg[:,None])**2).mean(axis=0); win=int(ls.argmin())
            rows.append({'dimension':dimension,'group':str(label),'n':len(g),'unknown_fraction':0.0,
                         'global_model':MODELS[global_winner],'global_model_brier':ls[global_winner],
                         'best_model':MODELS[win],'best_brier':ls[win],
                         'global_model_regret':ls[global_winner]-ls.min(),
                         'selected_global_is_group_best':global_winner==win,
                         'small_group':len(g)<100})
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'h5_full_observed_group_metrics.csv',index=False)
    pd.DataFrame([{'n':len(d),'global_winner':MODELS[global_winner],**{f'brier_{m}':losses[i] for i,m in enumerate(MODELS)},
                   'max_group_regret_age':max(r['global_model_regret'] for r in rows if r['dimension']=='age_group'),
                   'max_group_regret_sex':max(r['global_model_regret'] for r in rows if r['dimension']=='sex_group'),
                   'max_group_regret_education_code':max(r['global_model_regret'] for r in rows if r['dimension']=='education_group')}]).to_csv(out/'h5_global_summary.csv',index=False)
    # Controlled missingness: repeated hidden-label masks; G0/G1/G2 use exact subgroup C1.
    rng=np.random.default_rng(20261114); audit=[]; groups=d.age_group.astype(str).to_numpy()
    for rep in range(30):
        seen=rng.random(len(d))>=.2
        for gname in np.unique(groups):
            ix=np.flatnonzero(groups==gname)
            if len(ix)<100: continue
            gs=np.array([bounds(p[ix],y[ix],seen[ix])])
            audit.append({'mask':rep,'group':gname,'n':len(ix),'hidden_fraction':1-seen[ix].mean(),
                          **{f'G_{MODELS[j]}':gs[0,j] for j in range(len(MODELS))}})
    pd.DataFrame(audit).to_csv(out/'h5_age_group_c1_30_masks.csv',index=False)
    print(pd.read_csv(out/'h5_global_summary.csv').to_string(index=False))
    print(pd.DataFrame(rows).query("dimension=='age_group'").to_string(index=False))
if __name__=='__main__':
    raise SystemExit('Deprecated preliminary analysis with invalid bounds/split. Use prepare.py, analysis.py and report.py; see H5-RLMS/README.md.')

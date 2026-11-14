#!/usr/bin/env python3
"""Confirmatory H5 audit on frozen temporal test predictions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=['p_baseline','p_logistic','p_hist_gradient_boosting']; EPS=0.0025

def add_groups(d):
    d=d.copy(); d['age_group']=pd.cut(d.age,[17.999,34,49,64,np.inf],labels=['18-34','35-49','50-64','65+'])
    d['sex_group']=d.sex.map({1:'sex_code_1',2:'sex_code_2'}).fillna('unknown')
    d['education_group']=d.education.map(lambda x:f'educ_code_{int(x)}' if pd.notna(x) else 'unknown')
    return d

def c1(p,y,seen,series):
    out=[]
    for label, idx in series.groupby(series, observed=True).groups.items():
        ix=np.asarray(idx); pp=p[ix]; yy=y[ix]; ss=seen[ix]; n=len(ix); G=np.zeros(pp.shape[1])
        for a in range(pp.shape[1]):
            vals=[]
            for b in range(pp.shape[1]):
                if a==b: continue
                diff=pp[:,a]-pp[:,b]
                obs=(pp[ss,a]**2-pp[ss,b]**2-2*diff[ss]*yy[ss]).sum()
                hid=(pp[~ss,a]**2-pp[~ss,b]**2+np.maximum(-2*diff[~ss],0)).sum()
                vals.append((obs+hid)/n)
            G[a]=max(0.0,*vals)
        out.append((str(label),n,G))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); ap.add_argument('--masks',type=int,default=50); a=ap.parse_args()
    d=add_groups(pd.read_csv(a.input)); d=d[(d.split=='test') & d.target.notna()].copy().reset_index(drop=True); p=d[MODELS].to_numpy(); y=d.target.astype(int).to_numpy(); n=len(d)
    losses=((p-y[:,None])**2).mean(0); global_winner=int(losses.argmin()); rows=[]; decisions=[]; rng=np.random.default_rng(20261114)
    for mask in range(a.masks):
        seen=rng.random(n)>=.2
        gg=c1(p,y,seen,pd.Series(np.zeros(n))).pop()[2]
        for dim in ['age_group','sex_group','education_group']:
            gr=c1(p,y,seen,d[dim]); gm=np.vstack([x[2] for x in gr]); V=gm.max(0)/EPS; g1=int(V.argmin()); feasible=np.all(gm<=EPS+1e-12,axis=0)
            true_groups=[]
            for label,ix in d.groupby(dim,observed=True).groups.items():
                ix=np.asarray(ix); true_groups.append((str(label),len(ix),((p[ix]-y[ix,None])**2).mean(0)))
            tg=np.vstack([x[2] for x in true_groups]); g2=int(np.flatnonzero(feasible)[np.argmin(losses[np.flatnonzero(feasible)])]) if feasible.any() else None
            for method,sel in [('G0',global_winner),('G1',g1),('G2',g2)]:
                cert=sel is not None and ((method=='G0' and gg[sel]<=EPS+1e-12) or (method=='G1' and V[sel]<=1+1e-12) or (method=='G2' and feasible[sel]))
                decisions.append({'mask':mask,'dimension':dim,'method':method,'selected_model':MODELS[sel] if sel is not None else None,'certified':bool(cert),'true_global_regret':float(losses[sel]-losses.min()) if sel is not None else np.nan,'true_max_group_regret':float(np.max(tg[:,sel]-np.min(tg,axis=1))) if sel is not None else np.nan,'n_groups':len(gr),'hidden_fraction':1-seen.mean()})
            for label,sz,G in gr:
                for j,m in enumerate(MODELS): rows.append({'mask':mask,'dimension':dim,'group':label,'n':sz,'model':m,'G':G[j],'epsilon':EPS})
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/'h5_group_c1_bounds.csv',index=False); dec=pd.DataFrame(decisions); dec.to_csv(out/'h5_g0_g1_g2_decisions.csv',index=False)
    dec.groupby(['dimension','method']).agg(masks=('mask','nunique'),certificate_rate=('certified','mean'),mean_true_global_regret=('true_global_regret','mean'),mean_true_max_group_regret=('true_max_group_regret','mean')).reset_index().to_csv(out/'h5_confirmatory_summary.csv',index=False)
    pd.DataFrame([{'split':'test','n':n,'global_winner':MODELS[global_winner],**{f'brier_{m}':losses[i] for i,m in enumerate(MODELS)}}]).to_csv(out/'h5_test_global_metrics.csv',index=False)
    print(pd.read_csv(out/'h5_test_global_metrics.csv').to_string(index=False)); print(pd.read_csv(out/'h5_confirmatory_summary.csv').to_string(index=False))
if __name__=='__main__':
    raise SystemExit('Deprecated: incorrect temporal interpretation and oracle leakage in G0/G2. Use prepare.py, analysis.py and report.py; see H5-RLMS/README.md.')

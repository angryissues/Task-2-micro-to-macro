#!/usr/bin/env python3
"""Fit frozen annual and first-exit predictions for H4 trajectories."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES=['age','sex','education','start_employment']
def prep(dense=False):
    num=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    cat=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=not dense))])
    return ColumnTransformer([('num',num,['age']),('cat',cat,['sex','education','start_employment'])],verbose_feature_names_out=False)
def fit_one(train, x, target):
    y=train[target].astype(int); base=train.groupby('start_employment')[target].mean().to_dict(); overall=float(y.mean())
    out=pd.DataFrame({'p_baseline':x.start_employment.map(base).fillna(overall)})
    log=Pipeline([('preprocessor',prep(False)),('classifier',LogisticRegression(max_iter=2000,random_state=42))]); log.fit(train[FEATURES],y); out['p_logistic']=log.predict_proba(x[FEATURES])[:,1]
    hgb=Pipeline([('preprocessor',prep(True)),('classifier',HistGradientBoostingClassifier(max_iter=150,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.0,random_state=42))]); hgb.fit(train[FEATURES],y); out['p_hgb']=hgb.predict_proba(x[FEATURES])[:,1]
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
    x=pd.read_csv(a.input); x['observed_path']=x['complete_path'].astype(str).str.lower().eq('true')
    last=x.groupby('idind')['start_year'].max()+3; x['split']=x.idind.map(last).map(lambda y:'train' if y<2019 else ('validation' if y<=2021 else 'test'))
    train=x[(x.split=='train') & x.observed_path].copy()
    pred_parts=[]
    for h in (1,2,3):
        target=f'e{h}'; pp=fit_one(train,x,target); pp.columns=[f'{c}_h{h}' for c in pp.columns]; pred_parts.append(pp)
    for h in (1,2,3):
        x[f'first_exit_h{h}']=((x[[f'e{i}' for i in range(1,h+1)]]==0).any(axis=1)).astype(int)
    exit_parts=[]
    for h in (1,2,3):
        pp=fit_one(train,x,f'first_exit_h{h}'); pp.columns=[f'{c}_exit{h}' for c in pp.columns]; exit_parts.append(pp)
    out=pd.concat([x]+pred_parts+exit_parts,axis=1); out_dir=Path(a.output_dir); out_dir.mkdir(parents=True,exist_ok=True); out.to_csv(out_dir/'h4_frozen_predictions.csv',index=False)
    rows=[]
    for split,g in out[out.observed_path].groupby('split'):
      row={'split':split,'n':len(g)}
      for h in (1,2,3):
        row[f'brier_hgb_{h}']=float(np.mean((g[f'p_hgb_h{h}']-g[f'e{h}'])**2))
        row[f'brier_hgb_exit{h}']=float(np.mean((g[f'p_hgb_exit{h}']-g[f'first_exit_h{h}'])**2))
      rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir/'h4_prediction_metrics.csv',index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Fit the original H1 baseline/logistic/HGB stack on full transitions."""
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

FEATURES=['age','sex','education','current_employment']
def prep(dense=False):
    num=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    cat=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=not dense))])
    return ColumnTransformer([('num',num,['age']),('cat',cat,['sex','education','current_employment'])],verbose_feature_names_out=False)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
    x=pd.read_csv(a.input); x['observed']=x['target'].notna()
    last=x.groupby('idind')['target_year'].max(); x['split']=x['idind'].map(last).map(lambda y:'train' if y<2019 else ('validation' if y<=2021 else 'test'))
    train=x[(x.split=='train') & x.observed].copy(); y=train.target.astype(int)
    baseline=train.groupby('current_employment')['target'].mean().to_dict(); overall=float(y.mean()); x['p_baseline']=x.current_employment.map(baseline).fillna(overall)
    Xtr=train[FEATURES]; X=x[FEATURES]
    log=Pipeline([('preprocessor',prep(False)),('classifier',LogisticRegression(max_iter=2000,random_state=42))]); log.fit(Xtr,y); x['p_logistic']=log.predict_proba(X)[:,1]
    hgb=Pipeline([('preprocessor',prep(True)),('classifier',HistGradientBoostingClassifier(max_iter=150,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.0,random_state=42))]); hgb.fit(Xtr,y); x['p_hist_gradient_boosting']=hgb.predict_proba(X)[:,1]
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    x.to_csv(out/'full_transition_exact_model_input.csv',index=False)
    rows=[]
    for split,g in x[x.observed].groupby('split'):
        rows.append({'split':split,'n':len(g),**{f'brier_{c}':float(np.mean((g[c]-g.target)**2)) for c in ['p_baseline','p_logistic','p_hist_gradient_boosting']}})
    pd.DataFrame(rows).to_csv(out/'full_transition_exact_model_metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()

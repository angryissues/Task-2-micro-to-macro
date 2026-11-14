"""Train fixed NLSY97 candidate sets; confirmation metrics are not evaluated here."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES=['age','sex','current_employment']
def prep(dense):
    return ColumnTransformer([('age',Pipeline([('impute',SimpleImputer()),('scale',StandardScaler())]),['age']),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=not dense))]),['sex','current_employment'])])
def fit(train,target):
    d=train[train[target].notna()];y=d[target].astype(int);base=d.groupby('current_employment')[target].mean().to_dict();mean=float(y.mean());lr=Pipeline([('prep',prep(False)),('clf',LogisticRegression(max_iter=2000,random_state=42))]).fit(d[FEATURES],y);hg=Pipeline([('prep',prep(True)),('clf',HistGradientBoostingClassifier(max_iter=150,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.,random_state=42))]).fit(d[FEATURES],y);return base,mean,lr,hg
def predict(models,d):
    base,mean,lr,hg=models;return np.column_stack([d.current_employment.map(base).fillna(mean),lr.predict_proba(d[FEATURES])[:,1],hg.predict_proba(d[FEATURES])[:,1]])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panels',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.panels);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);train=pd.read_csv(root/'train_panel.csv.gz');splits={s:pd.read_csv(root/f'{s}_panel.csv.gz') for s in ['train','development','confirmation']};single={s:pd.read_csv(root/f'{s}_single_horizon_panel.csv.gz') for s in splits}
    ids=[set(splits[s].idind) for s in splits];assert all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i))
    manifest={'features':FEATURES,'education_excluded_from_predictors':'round-specific edited archival education may not match historical release vintage; used only as an archived grouping variable','models':{},'metrics_evaluated_on_confirmation':False}
    for h in (1,2,3):
        models=fit(train,f'y{h}');manifest['models'][f'annual_{h}']={'train_rows':int(train[f'y{h}'].notna().sum())}
        for s,d in splits.items():
            pred=predict(models,d)
            for k,m in enumerate(['baseline','logistic','hgb']):d[f'p_{m}_{h}']=pred[:,k]
        if h==1:
            for s,d in single.items():
                pred=predict(models,d)
                for k,m in enumerate(['baseline','logistic','hgb']):d[m]=pred[:,k]
    exittrain=train[train.current_employment==1].copy()
    for h in (1,2,3):
        col=f'exit{h}';known=exittrain[[f'y{j}' for j in range(1,h+1)]].notna().all(1);exittrain[col]=np.where(known,(exittrain[[f'y{j}' for j in range(1,h+1)]]==0).any(axis=1).astype(float),np.nan);models=fit(exittrain,col)
        for s,d in splits.items():
            pred=predict(models,d)
            for k,m in enumerate(['baseline','logistic','hgb']):d[f'p_{m}_exit{h}']=pred[:,k]
    for s,d in splits.items():d.to_csv(out/f'{s}_predictions.csv.gz',index=False)
    for s,d in single.items():d.to_csv(out/f'{s}_single_predictions.csv.gz',index=False)
    manifest['train_people']=int(train.idind.nunique());manifest['source_train_sha256']=hashlib.sha256((root/'train_panel.csv.gz').read_bytes()).hexdigest();(out/'model_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()

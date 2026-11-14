"""Build independent-person, forward-calendar H5 cohorts; microdata stay local."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
sys.path.insert(0,str(Path(__file__).parent))
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

def bucket(v): return int(hashlib.sha256(('H5-v2:'+str(int(v))).encode()).hexdigest()[:8],16)%10
def prep(dense):
    return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),['age']),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=not dense))]),['sex','education','current_employment'])])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sav',required=True);ap.add_argument('--transitions',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw,meta=pyreadstat.read_sav(a.sav,usecols=['idind','year','adult','j1'])
    raw['e']=raw.j1.map({1:1.,5:0.});raw.loc[raw.adult!=1,'e']=np.nan
    x=pd.read_csv(a.transitions).rename(columns={'target':'e1'})
    for h in (2,3):
        r=raw[['idind','year','e']].rename(columns={'year':'current_year','e':f'e{h}'})
        r['current_year']-=h;x=x.merge(r,on=['idind','current_year'],how='left',validate='one_to_one')
    x=x[x.age.between(18,65,inclusive='left') & x[['e1','e2','e3']].notna().all(axis=1)].copy()
    # Adult working-age target population. Fractional age is retained: 64.x is eligible.
    # Correct upper endpoint: age < 65.
    x['bucket']=x.idind.map(bucket);x['end_year']=x.current_year+3
    train=x[(x.bucket<6)&(x.end_year<=2018)].copy()
    val=x[(x.bucket.between(6,7))&(x.current_year>=2018)&(x.end_year<=2021)].copy()
    test=x[(x.bucket>=8)&(x.current_year>=2021)&(x.end_year<=2025)].copy()
    assert set(train.idind).isdisjoint(test.idind) and set(train.idind).isdisjoint(val.idind) and set(val.idind).isdisjoint(test.idind)
    features=['age','sex','education','current_employment'];pred={}
    for h in (1,2,3):
        y=train[f'e{h}'].astype(int);base=train.groupby('current_employment')[f'e{h}'].mean()
        lr=Pipeline([('prep',prep(False)),('clf',LogisticRegression(max_iter=2000,random_state=42))]).fit(train[features],y)
        hg=Pipeline([('prep',prep(True)),('clf',HistGradientBoostingClassifier(max_iter=150,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.,random_state=42))]).fit(train[features],y)
        for name,d in [('validation',val),('test',test)]:
            d[f'p_baseline_{h}']=d.current_employment.map(base).fillna(y.mean());d[f'p_logistic_{h}']=lr.predict_proba(d[features])[:,1];d[f'p_hgb_{h}']=hg.predict_proba(d[features])[:,1]
    meta_out={'cohorts':{},'sex_mapping':{'1':'male','2':'female'},'higher_education_codes':[21,22,23],'education_source':'SAV value labels educ: 21 diploma, 22/23 postgraduate; 19/20 incomplete higher education excluded from higher','features':features,'train_end_year':2018,'test_start_year':2021,'test_future_years':[2022,2025],'guarantee':'finite fixed complete-path cohort, unweighted; not population representativeness','selection':'complete adult binary future path and original cleaned H1 baseline record'}
    for name,d in [('train',train),('validation',val),('test',test)]:
        meta_out['cohorts'][name]={'rows':len(d),'people':int(d.idind.nunique()),'start_min':int(d.current_year.min()),'start_max':int(d.current_year.max()),'future_min':int(d.current_year.min()+1),'future_max':int(d.end_year.max())}
        if name!='train':d.to_csv(out/f'{name}_predictions.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(meta_out,indent=2));print(json.dumps(meta_out,indent=2),flush=True)
if __name__=='__main__':main()

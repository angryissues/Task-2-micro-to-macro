"""H6 T0/T1 pilot: frozen models, forward years, person-grouped domain OOF."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEATURES=['age','sex','education','current_employment'];MODELS=['baseline','logistic','hgb']
def prep(dense=False):
    return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),['age']),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=not dense))]),['sex','education','current_employment'])])
def bucket(v):return int(hashlib.sha256(('H6-v1:'+str(int(v))).encode()).hexdigest()[:8],16)%10
def balance(a,b,w):
    # SMD after common one-hot expansion; scaling by pooled unweighted SD.
    t=prep(True).fit(pd.concat([a,b])[FEATURES]);old=t.transform(a[FEATURES]);new=t.transform(b[FEATURES]);den=np.sqrt((old.var(0)+new.var(0))/2);use=den>1e-10
    before=np.abs(old.mean(0)-new.mean(0));after=np.abs(np.average(old,axis=0,weights=w)-new.mean(0));return float(np.mean(before[use]/den[use])),float(np.mean(after[use]/den[use]))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--config',required=True);ap.add_argument('--local',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True);local=Path(a.local);local.mkdir(parents=True,exist_ok=True)
    x=pd.read_csv(a.input);x=x[x.age.ge(18)&x.age.lt(65)&x.target.notna()].copy();x['bucket']=x.idind.map(bucket)
    train=x[(x.bucket<6)&(x.target_year<=cfg['train_target_year_max'])];future=x[x.bucket>=6].copy();assert set(train.idind).isdisjoint(future.idind)
    base=train.groupby('current_employment').target.mean();future['baseline']=future.current_employment.map(base).fillna(train.target.mean())
    log=Pipeline([('prep',prep()),('model',LogisticRegression(max_iter=2000,random_state=42))]).fit(train[FEATURES],train.target.astype(int))
    hgb=Pipeline([('prep',prep(True)),('model',HistGradientBoostingClassifier(max_iter=150,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.,random_state=42))]).fit(train[FEATURES],train.target.astype(int))
    future['logistic']=log.predict_proba(future[FEATURES])[:,1];future['hgb']=hgb.predict_proba(future[FEATURES])[:,1]
    future.to_csv(local/'h6_frozen_predictions.csv',index=False);rows=[];metric=[];weights_rows=[]
    for y0,y1 in cfg['temporal_pairs']:
        old=future[future.target_year==y0].reset_index(drop=True);new=future[future.target_year==y1].reset_index(drop=True);assert not old.empty and not new.empty
        features=pd.concat([old[FEATURES],new[FEATURES]],ignore_index=True);domain=np.r_[np.zeros(len(old)),np.ones(len(new))];ids=np.r_[old.idind,new.idind];D=np.zeros(len(features))
        for tr,te in GroupKFold(cfg['domain_oof_group_folds']).split(features,domain,ids):
            assert set(ids[tr]).isdisjoint(ids[te]);m=Pipeline([('prep',prep()),('domain',LogisticRegression(max_iter=2000,random_state=42))]).fit(features.iloc[tr],domain[tr]);D[te]=m.predict_proba(features.iloc[te])[:,1]
        prob=np.clip(D[:len(old)],1e-6,1-1e-6);ratios=prob/(1-prob)*len(old)/len(new);w=np.clip(ratios,*cfg['density_ratio_clip']);ess=w.sum()**2/(w*w).sum();before,after=balance(old,new,w)
        # Model choices use old labels only; new outcomes are reserved for final evaluation below.
        oldloss=(old[MODELS].to_numpy()-old.target.to_numpy()[:,None])**2;T0=int(oldloss.mean(0).argmin());T1=int(np.average(oldloss,axis=0,weights=w).argmin())
        selected=[('T0',T0),('T1',T1)];newloss=((new[MODELS].to_numpy()-new.target.to_numpy()[:,None])**2).mean(0)
        for name,k in selected:
            rows.append({'old_year':y0,'new_year':y1,'method':name,'selected_model':MODELS[k],'new_best_model':MODELS[int(newloss.argmin())],'new_true_regret':float(newloss[k]-newloss.min()),'exceeds_epsilon':bool(newloss[k]-newloss.min()>cfg['global_epsilon']+1e-12),'future_certificate':False,'interpretation':'past_choice_only' if name=='T0' else 'plugin_transfer','n_old':len(old),'n_new':len(new),'ess_weights':ess,'ess_fraction':ess/len(old),'weak_overlap':ess<.25*len(old),'domain_oof_auc':roc_auc_score(domain,D),'mean_abs_smd_before':before,'mean_abs_smd_after':after})
        for k,name in enumerate(MODELS):metric.append({'old_year':y0,'new_year':y1,'model':name,'old_brier':oldloss[:,k].mean(),'old_weighted_brier':np.average(oldloss[:,k],weights=w),'new_brier':newloss[k]})
        weights_rows.append({'old_year':y0,'new_year':y1,'min_ratio':float(w.min()),'max_ratio':float(w.max()),'clip_fraction':float(((ratios<.2)|(ratios>5)).mean()),'ess_fraction':ess/len(old)})
    pd.DataFrame(rows).to_csv(out/'t0_t1_transfer.csv',index=False);pd.DataFrame(metric).to_csv(out/'brier_by_pair.csv',index=False);pd.DataFrame(weights_rows).to_csv(out/'overlap.csv',index=False)
    provenance={'train_rows':len(train),'train_people':int(train.idind.nunique()),'train_future_year_max':int(train.target_year.max()),'model_weights_frozen_across_pairs':True,'person_disjoint_train_evaluation':True,'config':cfg,'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest()};(out/'provenance.json').write_text(json.dumps(provenance,indent=2));print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()

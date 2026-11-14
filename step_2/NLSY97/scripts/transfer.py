"""NLSY97 H6 T0/T1 on the already frozen h1 candidate predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from replicate import module,NAMES,EPS
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--config',required=True);ap.add_argument('--local',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();repo=Path(a.repo);root=Path(a.predictions);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);local=Path(a.local);local.mkdir(parents=True,exist_ok=True);cfg=json.loads(Path(a.config).read_text())['H6'];pilot=module('nls_transfer_helpers',repo/'step_2/H6-RLMS/scripts/pilot.py')
    data=pd.concat([pd.read_csv(root/f'{s}_single_predictions.csv.gz') for s in ['development','confirmation']],ignore_index=True).rename(columns={'y1':'target','target_year_1':'target_year'});data=data.sort_values('origin_week').drop_duplicates(['idind','target_year'],keep='last');data=data[data.target.notna()].copy();data.to_csv(local/'H6_input.csv',index=False);rows=[];metrics=[]
    for y0,y1 in cfg['temporal_pairs']:
        old=data[data.target_year==y0].reset_index(drop=True);new=data[data.target_year==y1].reset_index(drop=True);features=pd.concat([old[pilot.FEATURES],new[pilot.FEATURES]],ignore_index=True);domain=np.r_[np.zeros(len(old)),np.ones(len(new))];ids=np.r_[old.idind,new.idind];D=np.zeros(len(features))
        for tr,te in GroupKFold(5).split(features,domain,ids):
            assert set(ids[tr]).isdisjoint(ids[te]);clf=Pipeline([('prep',pilot.prep()),('classifier',LogisticRegression(max_iter=2000,random_state=42))]).fit(features.iloc[tr],domain[tr]);D[te]=clf.predict_proba(features.iloc[te])[:,1]
        pp=np.clip(D[:len(old)],1e-6,1-1e-6);r=np.clip(pp/(1-pp)*len(old)/len(new),.2,5);ess=r.sum()**2/(r*r).sum();before,after=pilot.balance(old,new,r);oldloss=(old[NAMES].to_numpy()-old.target.to_numpy()[:,None])**2;t0=int(oldloss.mean(0).argmin());t1=int(np.average(oldloss,axis=0,weights=r).argmin());ls=((new[NAMES].to_numpy()-new.target.to_numpy()[:,None])**2).mean(0)
        for method,k in [('T0',t0),('T1',t1)]:rows.append(dict(old_year=y0,new_year=y1,method=method,n_old=len(old),n_new=len(new),model=NAMES[k],new_best=NAMES[int(ls.argmin())],new_true_regret=float(ls[k]-ls.min()),exceeds_epsilon=bool(ls[k]-ls.min()>EPS+1e-12),future_certificate=False,scope='past_choice' if method=='T0' else 'plugin_transfer',ess_fraction=ess/len(old),weak_overlap=ess<.25*len(old),domain_oof_auc=roc_auc_score(domain,D),mean_smd_before=before,mean_smd_after=after))
        for k,m in enumerate(NAMES):metrics.append(dict(old_year=y0,new_year=y1,model=m,old_brier=oldloss[:,k].mean(),weighted_old_brier=np.average(oldloss[:,k],weights=r),new_brier=ls[k]))
    pd.DataFrame(rows).to_csv(out/'T0_T1.csv',index=False);pd.DataFrame(metrics).to_csv(out/'T0_T1_brier.csv',index=False);flat={'temporal_pairs':cfg['temporal_pairs'],'t2_eta_grid':cfg['T2_eta'],'t3_joint_tv_rho_grid':cfg['T3_joint_TV_rho'],'t4_query_budgets':cfg['T4_budgets']};(local/'H6_config.json').write_text(json.dumps(flat,indent=2));print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()

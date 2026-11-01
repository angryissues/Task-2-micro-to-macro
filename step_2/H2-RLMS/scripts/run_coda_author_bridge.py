#!/usr/bin/env python3
"""Use unmodified author CODA EIG ordering with H2's common Brier C1 stop rule."""
import argparse, csv, gzip, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.active_selection import LabelOracle, SelectionState

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--author-checkout',type=Path,required=True); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--mask-start',type=int,default=0); ap.add_argument('--mask-count',type=int); a=ap.parse_args()
    sys.path.insert(0,str(a.author_checkout)); from coda.coda import CODA
    cfg=json.loads(a.config.read_text()); seed=int(cfg.get('seed',20261101)); rng=np.random.default_rng(seed)
    with gzip.open(a.input,'rt',newline='') as fh: rows=list(csv.DictReader(fh))
    take=np.sort(rng.choice(len(rows),min(int(cfg['n_rows']),len(rows)),replace=False)); rows=[rows[i] for i in take]
    names=['p_baseline','p_logistic','p_hist_gradient_boosting']; p=np.array([[float(r[x]) for x in names] for r in rows]); y=np.array([int(float(r['y'])) for r in rows])
    budgets=[int(x) for x in cfg['budgets']]; out=[]
    n_masks=int(cfg['n_masks']); stop=n_masks if a.mask_count is None else min(n_masks,a.mask_start+a.mask_count)
    if a.mask_start < 0 or a.mask_start >= n_masks or stop <= a.mask_start: raise ValueError('invalid mask range')
    for mask_seed in range(a.mask_start, stop):
        mrng=np.random.default_rng(seed+mask_seed); hidden=np.zeros(len(y),bool); hidden[mrng.choice(len(y),round(len(y)*float(cfg['missing_fraction'])),replace=False)]=True
        state=SelectionState.create(p,y,~hidden); selector=CODA(SimpleNamespace(preds=torch.tensor(np.stack((1-p.T,p.T),axis=-1),dtype=torch.float32)),prefilter_n=int(cfg.get('coda_prefilter_n',128)),q='eig'); selector.unlabeled_idxs=np.flatnonzero(hidden).tolist(); oracle=LabelOracle(y); snapshots={}
        for step in range(max(budgets)+1):
            d=state.certificate(float(cfg['epsilon']))
            if step in budgets: snapshots[step]=(d,step,'certified' if d['certified'] else 'budget_exhausted')
            if d['certified'] or step==max(budgets):
                for b in budgets:
                    if b>=step and b not in snapshots: snapshots[b]=(d,step,'certified' if d['certified'] else 'budget_exhausted')
                break
            row,score=selector.get_next_item_to_label(); label=oracle.query(row); selector.add_label(row,label,score); state.reveal(row,label)
        losses=((p-y[:,None])**2).mean(0)
        for b in budgets:
            d,q,status=snapshots[b]; selected=d['recommended_model'] if d['certified'] else None; regret=None if selected is None else float(losses[selected]-losses.min())
            out.append({'policy':'coda_author_eig_bridge','mask_seed':mask_seed,'max_budget':b,'certified':d['certified'],'status':status,'total_cost':q,'queries_obtained':q,'G':d['G'],'selected_model':selected,'true_regret':regret,'wrong_certificate':bool(d['certified'] and regret>float(cfg['epsilon'])+1e-12),'coda_prefilter_n':int(cfg.get('coda_prefilter_n',128))})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as fh: w=csv.DictWriter(fh,fieldnames=out[0]); w.writeheader(); w.writerows(out)
    print('wrote',len(out),'rows to',a.output)
if __name__=='__main__': main()

#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np
from src.nonresponse import draw_initial_mask, initial_missing_probabilities, run_nonresponse
from src.scenario_c2 import ScenarioState, scenario_count_bounds
from scripts.run_h3_rlms import load_input, true_regret

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True,type=Path); ap.add_argument('--strata-input',required=True,type=Path); ap.add_argument('--config',required=True,type=Path); ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args(); cfg=json.loads(a.config.read_text()); seed=int(cfg.get('seed',20261111))
    p,y,age,low,names=load_input(a.input,a.strata_input,int(cfg.get('n_rows',1000)),seed)
    miss=float(cfg.get('missing_fraction',.2)); eps=float(cfg.get('epsilon',.0025)); ans=float(cfg.get('answer_probability',.5)); budgets=[float(x) for x in cfg.get('budgets',[100,200])]
    gammas=cfg.get('gammas', [cfg.get('gamma', 2.0)])
    gid=(2*(age>=50).astype(int)+(low>0.5).astype(int)).astype(int); out=[]
    for m in range(int(cfg.get('n_masks',30))):
        hidden=draw_initial_mask(initial_missing_probabilities(age,low,y,'mcar',miss),np.random.default_rng(seed+m)); response=np.full(len(y),ans)
        for gamma_raw in gammas:
            gamma = float('inf') if str(gamma_raw).lower() in {'inf', 'infinity'} else float(gamma_raw)
            assumption_holds = True
            for g in np.unique(gid):
                vis = (~hidden) & (gid == g); hid = hidden & (gid == g)
                lower_count, upper_count = scenario_count_bounds(int(y[vis].sum()), int(vis.sum()), int(hid.sum()), gamma)
                actual = int(y[hid].sum())
                assumption_holds = assumption_holds and (lower_count <= actual <= upper_count)
            for b in budgets:
                state=ScenarioState.create(p,y,~hidden,gid,gamma)
                # R3 development uses a cheap disagreement queue. The scenario
                # C2 certificate is the object of study; C2 lookahead is deferred.
                res=run_nonresponse(state,y,response,'r3',eps,b,seed=seed+10000*m,policy_answer_probabilities=response,lower_answer_probabilities=response)
                sel=res['recommended_model'] if res['certified'] else None; reg=true_regret(p,y,sel)
                out.append({'mask_seed':m,'gamma':gamma,'assumption_holds':assumption_holds,'budget':b,'certified':res['certified'],'status':res['status'],'G':res['G'],'total_cost':res['total_cost'],'attempts':res['queries_attempted'],'labels_obtained':res['labels_obtained'],'selected_model':sel,'true_regret':reg,'wrong_certificate':bool(res['certified'] and reg>eps+1e-12),'guarantee_type':'scenario_shift'})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=out[0]); w.writeheader(); w.writerows(out)
    print('wrote',len(out),'runs to',a.output)
if __name__=='__main__': main()

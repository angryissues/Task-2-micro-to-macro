#!/usr/bin/env python3
"""Person-level MCAR sensitivity experiment for H1 C0/C1."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from src.bounds import c0_bounds, c1_bounds, TOL
from src.certification import decide_from_bounds
from src.metrics import full_brier, true_regret_vector

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="configs/h1_default.json")
    ap.add_argument("--out", default="results/person_level_masks")
    ap.add_argument("--n-seeds", type=int, default=30)
    args=ap.parse_args()
    df=pd.read_csv(args.input)
    if "idind" not in df or "y" not in df:
        raise ValueError("Input must contain idind and y")
    cols=[c for c in df.columns if c.startswith("p_")]
    if len(cols)<2: raise ValueError("Need at least two p_* forecast columns")
    if df["y"].isna().any(): raise ValueError("Reference experiment requires known labels")
    p=df[cols].to_numpy(float); y=df["y"].to_numpy(float)
    ids=df["idind"].to_numpy()
    people=np.unique(ids)
    with open(args.config,encoding="utf-8") as f: cfg=json.load(f)
    eps_all=list(cfg["epsilon_grid"])+list(cfg.get("epsilon_extra",[]))
    R=full_brier(p,y,None)
    tr=true_regret_vector(R)
    rows=[]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    for m in cfg["missing_fractions"]:
        for seed in range(args.n_seeds):
            rng=np.random.default_rng(np.random.SeedSequence([47,seed,int(m*100)]))
            hidden_people=set(rng.choice(people,size=int(round(m*len(people))),replace=False).tolist())
            hidden=np.isin(ids,list(hidden_people))
            obs=~hidden
            yi=np.where(obs,y,np.nan)
            row_missing=float(hidden.mean())
            for method,fn in (("C0",c0_bounds),("C1",c1_bounds)):
                lo,up,g=fn(p,yi,obs,None)
                for eps in eps_all:
                    d=decide_from_bounds(lo,up,g,eps,"finite_exact")
                    rec=d["recommended_model"]
                    rows.append({
                      "method":method,"person_missing_fraction":m,
                      "actual_person_missing_fraction":len(hidden_people)/len(people),
                      "actual_row_missing_fraction":row_missing,"mask_seed":seed,
                      "epsilon":eps,"recommended":cols[rec],"G":d["worst_regret"],
                      "certified":bool(d["certified"]),
                      "reference_regret_full_test":float(tr[rec]),
                      "wrong_certificate":bool(d["certified"] and tr[rec] > eps + TOL),
                      "n_people_hidden":len(hidden_people),"n_rows_hidden":int(hidden.sum()),
                    })
    res=pd.DataFrame(rows)
    res.to_csv(out/"person_level_mask_rows.csv",index=False)
    summary=(res.groupby(["method","person_missing_fraction","epsilon"],as_index=False)
        .agg(certificate_rate=("certified","mean"),mean_G=("G","mean"),
             min_G=("G","min"),max_G=("G","max"),
             wrong_certificates=("wrong_certificate","sum"),
             repetitions=("mask_seed","nunique"),
             mean_actual_row_missing=("actual_row_missing_fraction","mean"),
             mean_full_test_regret=("reference_regret_full_test","mean")))
    summary.to_csv(out/"person_level_mask_summary.csv",index=False)
    details=(res.groupby(["person_missing_fraction"],as_index=False)
        .agg(mean_people_hidden=("actual_person_missing_fraction","mean"),
             mean_rows_hidden=("actual_row_missing_fraction","mean"),
             min_rows_hidden=("actual_row_missing_fraction","min"),
             max_rows_hidden=("actual_row_missing_fraction","max")))
    details.to_csv(out/"person_level_mask_fraction_check.csv",index=False)
    print(f"rows={len(df):,}; people={len(people):,}; output_rows={len(res):,}")
    print(details.to_string(index=False))
    print("\nC1 summary (certification rate and mean G):")
    print(summary[summary.method.eq("C1")].to_string(index=False))
if __name__=="__main__": main()

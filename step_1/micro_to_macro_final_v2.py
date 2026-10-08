"""Задача 2 (FINAL): какие микроошибки A2 разрушают макро-траекторию.
Основной эталон (Reference B) = модель из expanded_benchmark.py: median-импутер + HGB(250, 0.05, l2=1),
train target_year<=2021, температурное масштабирование, T подбирается по log_loss на 2022-2023 (сетка 0.5..3.0, не вручную).
Reference A (изотоническая калибровка) — только проверка устойчивости (--reference isotonic).
Показатель траектории: «доля исходной работающей когорты, сохранившей основную занятость» (employment survival).
Paired simulation: для каждой репликации r случайные числа U[step, person] и V[step, person] заданы заранее по seed r
и общие для эталона и всех искажений (по человеку, а не по позиции); случайность самих искажений — отдельный поток.
python micro_to_macro_final.py --a2 a2_pairs.csv.gz --out out_final [--reference temperature|isotonic] [--reps 40] [--steps 5]
"""
import argparse, json, warnings, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss
warnings.filterwarnings("ignore")
C = np.array(["different_job", "no_job", "same_job"])
F = ["year","age","sex","education","region","settlement","marital","occupation","wage","hours",
     "tenure","history_years","wage_change","occupation_changed","prior_job_change"]
PERS = ["tenure","history_years","prior_job_change","occupation_changed","wage_change"]  # state-dependence features
EPS = 1e-6

def prep(d):
    X = d[F].copy()
    for c in ["wage"]: X[c] = np.log1p(X[c].clip(lower=0))
    return X

class Ref:
    """HGB + изотоническая калибровка по классам (fit<=2021, calib 2022-23)."""
    def __init__(s, tr, cal):
        s.m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=1, random_state=42)
        s.m.fit(prep(tr), tr.target)
        raw = s.raw(cal); y = (cal.target.to_numpy(object)[:,None] == C[None,:]).astype(float)
        s.iso = [IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1-EPS).fit(raw[:,k], y[:,k]) for k in range(3)]
    def raw(s, d):
        r = s.m.predict_proba(prep(d)); out = np.zeros((len(d),3))
        for i, l in enumerate(s.m.classes_): out[:, np.where(C==l)[0][0]] = r[:,i]
        return out
    def __call__(s, d):
        r = s.raw(d); p = np.column_stack([s.iso[k].predict(r[:,k]) for k in range(3)])
        return p / p.sum(1, keepdims=True)

class RefFile:
    """Эталон ровно как в strong_llm.py: median-импутер + HGB, без калибровки, обучение target_year<=2021."""
    def __init__(s, tr, cal=None):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        s.m = make_pipeline(SimpleImputer(strategy="median"),
              HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=1, random_state=42))
        s.m.fit(prep(tr), tr.target)
    def __call__(s, d):
        r = s.m.predict_proba(prep(d)); out = np.zeros((len(d),3))
        for i, l in enumerate(s.m.classes_): out[:, np.where(C==l)[0][0]] = r[:,i]
        return out

class RefTemp:
    """Эталон ровно как в expanded_benchmark.py: median-импутер + HGB, температурное масштабирование (T по log_loss на 2022-23)."""
    def __init__(s, tr, cal):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        s.m = make_pipeline(SimpleImputer(strategy="median"),
              HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=1, random_state=42))
        s.m.fit(prep(tr), tr.target)
        raw = s.raw(cal); yy = cal.target.to_numpy(object)
        grid = np.linspace(0.5, 3.0, 251)
        loss = [log_loss(yy, s.scale(raw, t), labels=C) for t in grid]
        s.T = float(grid[int(np.argmin(loss))])
    def raw(s, d):
        r = s.m.predict_proba(prep(d)); out = np.zeros((len(d),3))
        for i, l in enumerate(s.m.classes_): out[:, np.where(C==l)[0][0]] = r[:,i]
        return out
    @staticmethod
    def scale(p, t):
        q = np.clip(p, 1e-12, 1) ** (1/t); return q / q.sum(1, keepdims=True)
    def __call__(s, d): return s.scale(s.raw(d), s.T)

def brier(y, p): return float(((p - (C[None,:]==y[:,None]))**2).sum(1).mean())
def ece(y, p, k=1, bins=10):
    t=(y==C[k]).astype(float); q=p[:,k]; b=np.minimum((q*bins).astype(int),bins-1); e=0
    for i in range(bins):
        m=b==i
        if m.any(): e+=m.mean()*abs(t[m].mean()-q[m].mean())
    return float(e)

# ---------- искажения ----------
class Pert:
    """Каждое искажение: fit(ref, step1 population) подбирает константы так, чтобы доли на шаге 1 совпали с эталоном;
    затем apply(d, P_ref, rng) -> вероятности на любом шаге с теми же (зафиксированными) константами."""
    name="ref"
    def fit(s, ref, d, P): s.m1 = P.mean(0)
    def apply(s, d, P, rng): return P
def renorm(p): p=np.clip(p,EPS,None); return p/p.sum(1,keepdims=True)
def match_logit(P, target, iters=60):
    """сдвиг в logit-пространстве, чтобы средние совпали с target"""
    L=np.log(renorm(P)); b=np.zeros(3)
    for _ in range(iters):
        Q=np.exp(L+b); Q/=Q.sum(1,keepdims=True); b+=0.8*(np.log(target)-np.log(Q.mean(0)))
    return b
def apply_shift(P,b): Q=np.exp(np.log(renorm(P))+b); return Q/Q.sum(1,keepdims=True)

class Shuffle(Pert):
    name="shuffle"
    def apply(s,d,P,rng): return P[rng.permutation(len(P))]
class Collapse(Pert):
    def __init__(s,lam): s.lam=lam; s.name=f"collapse_{lam}"
    def apply(s,d,P,rng): return renorm(s.m1 + s.lam*(P - s.m1))
class Subgroup(Pert):
    """no_job-риск: группе A ×(1+delta), группе B ×(1-delta*w); константа подбирается на шаге 1."""
    def __init__(s,kind,delta): s.kind=kind; s.delta=delta; s.name=f"subgroup_{kind}_{int(delta*100)}"
    def grp(s,d):
        if s.kind=="age": return np.where(d.age<35,1,np.where(d.age>=50,-1,0))
        if s.kind=="sex": return np.where(d.sex==2,1,-1)
        if s.kind=="educ": return np.where(d.education>=21,-1,1)   # без вуза: завышаем, с вузом: занижаем
    def raw(s,d,P):
        g=s.grp(d); f=np.where(g==1,1+s.delta,np.where(g==-1,1-s.delta,1.0))
        Q=P.copy(); Q[:,1]*=f; return renorm(Q)
    def fit(s,ref,d,P):
        s.m1=P.mean(0); Q=s.raw(d,P); s.k=P[:,1].mean()/Q[:,1].mean()
    def apply(s,d,P,rng):
        Q=s.raw(d,P); Q[:,1]*=s.k; return renorm(Q)
class Persistence(Pert):
    """Persistence degradation. Сила alpha = доля людей, у которых 5 признаков зависимости от прошлого состояния
    (tenure, history_years, prior_job_change, occupation_changed, wage_change) заменены значениями случайного другого
    человека когорты. alpha=0 — без искажения, alpha=1 — зависимость от прошлого убрана у всех. После замены вводится
    logit-сдвиг, возвращающий средние доли шага 1 к эталону."""
    def __init__(s,alpha): s.alpha=alpha; s.name=f"persistence_{alpha}"
    def fit(s,ref,d,P):
        s.ref=ref; s.pool=d[PERS].copy(); s.m1=P.mean(0)
        rng=np.random.default_rng(0); Q=s.pred(d,rng,fixed=True); s.b=match_logit(Q,P.mean(0))
    def pred(s,d,rng,fixed=False):
        d2=d.copy(); n=len(d2); sel=rng.random(n)<s.alpha
        idx=rng.integers(0,len(s.pool),sel.sum())
        for c in PERS: d2.loc[d2.index[sel],c]=s.pool[c].values[idx]
        return s.ref(d2)
    def apply(s,d,P,rng): return apply_shift(s.pred(d,rng),s.b)


# ---------- динамика когорты работников (правила не менялись) ----------
def step_state(d, y, v, rates):
    """Состояние на t+1 строится ТОЛЬКО из состояния на t и разыгранного исхода (никаких признаков target-года/будущих волн):
    same_job: tenure+=1, prior_job_change=0; different_job: tenure=0, prior_job_change=1; age+=1, year+=1, history_years+=1;
    occupation_changed ~ Bernoulli(rates[prior_job_change]) с эмпирическими долями когорты на старте (v — заранее заданное равномерное число);
    wage_change=0 (зарплата и часы не меняются)."""
    d=d.copy(); d["year"]+=1; d["age"]+=1; d["history_years"]+=1
    dj=(y=="different_job")
    d["tenure"]=np.where(dj,0,d["tenure"]+1); d["prior_job_change"]=dj.astype(float)
    d["occupation_changed"]=(v<np.where(dj,rates[1],rates[0])).astype(float)
    d["wage_change"]=0.0
    return d

def GROUPS(d0):
    return {"age<35":(d0.age<35).values,"age35-49":((d0.age>=35)&(d0.age<50)).values,"age50+":(d0.age>=50).values,
            "female":(d0.sex==2).values,"male":(d0.sex==1).values,
            "no_univ":(d0.education<21).values,"univ":(d0.education>=21).values}

def rollout(d0, ref, pert, steps, seed, rates, U, V):
    rng_p=np.random.default_rng(10**6+seed)           # случайность искажения (перемешивание, выбор людей) — отдельный поток
    d=d0.copy(); n0=len(d); rec=[]; alive=np.ones(n0,bool); idx=np.arange(n0); grp=GROUPS(d0)
    for k in range(1,steps+1):
        Pr=ref(d); P=pert.apply(d,Pr,rng_p) if pert.name!="ref" else Pr
        u=U[k-1,idx]                                   # общие числа по человеку
        yi=(u[:,None]>P.cumsum(1)).sum(1).clip(max=2); y=C[yi]
        alive[idx[y=="no_job"]]=False
        rec.append(dict(step=k, surv=alive.mean(), nojob_step=(y=="no_job").mean(), dj_step=(y=="different_job").mean(),
                        p_dj=P[:,0].mean(), p_no=P[:,1].mean(), p_same=P[:,2].mean(), n_alive=int(alive.sum()),
                        **{f"surv_{g}":alive[m].mean() for g,m in grp.items()}))
        keep=y!="no_job"; d=step_state(d[keep],y[keep],V[k-1,idx[keep]],rates); idx=idx[keep]
        if len(d)==0: break
    return rec

# ---------- набор искажений ----------
def build_perts():
    L=[("ref","Reference","reference",1.0,True,Pert())]
    L.append(("shuffle","Shuffle","shuffle",1.0,True,Shuffle()))
    for lam in [0.75,0.5,0.25,0.0]:
        lab={0.5:"Half collapse (λ=0.5)",0.0:"Mode collapse (λ=0)"}.get(lam,f"Collapse λ={lam}")
        p=Collapse(lam); L.append((p.name,lab,"collapse",1-lam,lam in(0.5,0.0),p))
    p=Collapse(1.0); L.append((p.name,"Collapse λ=1 (проверка тождества)","collapse",0.0,False,p))
    for al in [0.0,0.25,0.5,0.75,1.0]:
        p=Persistence(al); lab={0.5:"Persistence 50%",1.0:"Persistence 100%"}.get(al,f"Persistence {int(al*100)}%")
        L.append((p.name,lab,"persistence",al,al in(0.5,1.0),p))
    base={"age":0.5,"educ":0.5,"sex":0.4}; nm={"age":"Age bias","educ":"Education bias","sex":"Sex bias"}
    for kind in ["age","educ","sex"]:
        for dl in sorted({0.25,0.5,0.75,1.0,base[kind]}):
            p=Subgroup(kind,dl); L.append((p.name,f"{nm[kind]} (δ={dl})",f"subgroup_{kind}",dl,dl==base[kind],p))
    return L

def mono(vals, ses):
    """vals/ses упорядочены по силе искажения. strict: |значение| не убывает; tolerant: убывание допускается в пределах 1.96*SE разности"""
    v=np.abs(np.array(vals)); s=np.array(ses); strict=bool(np.all(np.diff(v)>=0))
    tol=bool(np.all(np.diff(v)>=-1.96*np.sqrt(s[1:]**2+s[:-1]**2)))
    return strict,tol

def main(a):
    from scipy.stats import spearmanr
    out=Path(a.out); out.mkdir(exist_ok=True, parents=True)
    d=pd.read_csv(a.a2).dropna(subset=["target_year"]).reset_index(drop=True); d["target"]=d["target"].astype(object)
    tr=d[d.target_year<=2021]; cal=d[(d.target_year>=2022)&(d.target_year<=2023)]; te=d[d.target_year>=2024].reset_index(drop=True)
    assert tr.target_year.max()<=2021 and cal.target_year.between(2022,2023).all() and te.target_year.min()>=2024
    ref=(RefTemp if a.reference=="temperature" else Ref)(tr,cal); Pte=ref(te); y=te.target.to_numpy(object)
    L=build_perts(); meta={n:dict(label=l,family=f,strength=s,is_baseline=b) for n,l,f,s,b,_ in L}
    # --- шаг 1 на тесте 2024-2025 ---
    rng=np.random.default_rng(1); rows=[]; g_te=GROUPS(te)
    for n,l,f,s,b,p in L:
        p.fit(ref,te,Pte)
        P=Pte if n=="ref" else p.apply(te,Pte,rng); sh=P.mean(0)
        sp=[float(spearmanr(Pte[:,k],P[:,k])[0]) if P[:,k].std()>0 else np.nan for k in (1,0)]
        gap=max(abs(P[m,1].mean()-(y[m]=="no_job").mean()) for m in g_te.values())
        rows.append(dict(model=n,label=l,family=f,strength=s,is_baseline=b,
            share_same_job=sh[2],share_different_job=sh[0],share_no_job=sh[1],survival_share_step1=1-sh[1],
            brier=brier(y,P),logloss=float(log_loss(y,P,labels=C)),ece_nojob=ece(y,P,1),
            auc_nojob=float(roc_auc_score(y=="no_job",P[:,1])),auc_different_job=float(roc_auc_score(y=="different_job",P[:,0])),
            spearman_nojob=sp[0],spearman_different_job=sp[1],sd_p_nojob=float(P[:,1].std()),max_subgroup_nojob_calib_gap=gap))
    one=pd.DataFrame(rows); r0=one[one.model=="ref"].iloc[0]
    one["max_abs_dev_pp_test"]=100*np.maximum.reduce([(one[c]-r0[c]).abs() for c in ["share_same_job","share_different_job","share_no_job"]])
    # --- многошаговая когорта 2023 ---
    c0=te[te.year==te.year.min()].reset_index(drop=True)
    r_oc=[c0[c0.prior_job_change==0].occupation_changed.mean(), c0[c0.prior_job_change==1].occupation_changed.mean()]
    rates=np.nan_to_num(r_oc,nan=0.1); P0=ref(c0); n0=len(c0)
    for n,l,f,s,b,p in L: p.fit(ref,c0,P0)
    recs=[]
    for r in range(a.reps):
        print('rep',r,flush=True); g=np.random.default_rng(1000+r); U=g.random((a.steps,n0)); V=g.random((a.steps,n0))   # общие числа для всех моделей
        for n,l,f,s,b,p in L:
            for rec in rollout(c0,ref,p,a.steps,1000+r,rates,U,V): rec.update(model=n,rep=r); recs.append(rec)
    R=pd.DataFrame(recs); R.to_csv(out/"rollout_raw_final.csv",index=False)
    # paired check: при lambda=1 и alpha=0 траектория обязана совпасть с эталоном
    key=["step","rep"]; base=R[R.model=="ref"].set_index(key)
    for ident in ["collapse_1.0","persistence_0.0"]:
        dd=(R[R.model==ident].set_index(key)["surv"]-base["surv"].reindex(R[R.model==ident].set_index(key).index)).abs().max()
        assert dd<1e-9, (ident,dd)
    sc=[c for c in R if c.startswith("surv_")]; cols=["surv","dj_step","nojob_step"]+sc
    # траектории (step 0..5), CI по 40 повторам
    traj=[]
    for n,g in R.groupby("model"):
        traj.append(dict(model=n,step=0,mean=1.0,lo=1.0,hi=1.0))
        for k,gg in g.groupby("step"):
            lo,hi=np.quantile(gg.surv,[.025,.975]); traj.append(dict(model=n,step=k,mean=gg.surv.mean(),lo=lo,hi=hi))
    T=pd.DataFrame(traj); T["label"]=T.model.map(lambda m:meta[m]["label"]); T.to_csv(out/"trajectories_final.csv",index=False)
    # дрейф относительно эталона (парные разности по репликации)
    rows=[]; sub=[]
    for n,g in R[R.model!="ref"].groupby("model"):
        g=g.set_index(key); diff=g[cols]-base[cols].reindex(g.index)
        for k,dd in diff.groupby(level=0):
            row=dict(model=n,label=meta[n]["label"],family=meta[n]["family"],strength=meta[n]["strength"],is_baseline=meta[n]["is_baseline"],step=k)
            for c in cols:
                lo,hi=np.quantile(dd[c],[.025,.975]); row[c+"_drift"]=dd[c].mean(); row[c+"_lo"]=lo; row[c+"_hi"]=hi
            sg={c:abs(dd[c].mean()) for c in sc}; mg=max(sg,key=sg.get)
            row["max_abs_subgroup_drift"]=sg[mg]; row["max_subgroup"]=mg[5:]
            rows.append(row)
            if k==a.steps:
                for c in sc:
                    lo,hi=np.quantile(dd[c],[.025,.975])
                    sub.append(dict(model=n,label=meta[n]["label"],family=meta[n]["family"],strength=meta[n]["strength"],is_baseline=meta[n]["is_baseline"],
                        step=k,group=c[5:],drift=dd[c].mean(),ci_lo=lo,ci_hi=hi,abs_drift=abs(dd[c].mean()),
                        is_max_group=(c==mg),max_abs_subgroup_drift=sg[mg]))
    D=pd.DataFrame(rows); D.to_csv(out/"drift_vs_reference_final.csv",index=False)
    S=pd.DataFrame(sub); S.to_csv(out/"subgroup_drift_final.csv",index=False)
    # шаг-1 доли в когорте (ожидаемые, по вероятностям модели) — проверка согласования на самой когорте
    st1=R[R.step==1].groupby("model")[["p_same","p_dj","p_no"]].mean(); rf=st1.loc["ref"]
    one["cohort_step1_max_abs_dev_pp"]=one.model.map(lambda m:100*float((st1.loc[m]-rf).abs().max()))
    one["aggregate_match_within_0.5pp"]=(one.max_abs_dev_pp_test<=0.5)&(one.cohort_step1_max_abs_dev_pp<=0.5)
    one.to_csv(out/"one_step_metrics_final.csv",index=False)
    # sensitivity: все уровни, плюс проверка монотонности
    d5=D[D.step==a.steps].set_index("model")
    sens=one.merge(d5[["surv_drift","surv_lo","surv_hi","max_abs_subgroup_drift","max_subgroup"]].rename(columns={
        "surv_drift":"drift_step5_survival","surv_lo":"drift_step5_ci_lo","surv_hi":"drift_step5_ci_hi"}),left_on="model",right_index=True,how="left")
    sens=sens[sens.family!="reference"].copy()

    # ВАЖНО: для aggregate drift используем настоящую paired SE по 40 репликациям,
    # а не восстанавливаем SE из ширины квантильного CI.
    agg_se_map={}
    for n,g in R[R.model!="ref"].groupby("model"):
        dd=g[g.step==a.steps].set_index("rep")
        base_rep=R[(R.model=="ref") & (R.step==a.steps)].set_index("rep")["surv"]
        diff=(dd["surv"]-base_rep.reindex(dd.index)).dropna()
        agg_se_map[n]=float(diff.std(ddof=1)/np.sqrt(len(diff)))
    sens["drift_step5_se"]=sens.model.map(agg_se_map)

    mrows=[]
    for fam,g in sens.groupby("family"):
        if fam=="shuffle": continue
        g=g.sort_values("strength"); s1,t1=mono(g.drift_step5_survival,g.drift_step5_se)
        # max subgroup drift — максимум по группам, поэтому его неопределённость
        # нельзя оценивать aggregate SE. Оставляем только point-estimate monotonicity.
        s2=bool(np.all(np.diff(np.abs(g.max_abs_subgroup_drift.to_numpy()))>=0))
        mrows.append(dict(family=fam,monotone_agg_drift_strict=s1,monotone_agg_drift_within_MC_noise=t1,
                          monotone_max_subgroup_drift_strict=s2,
                          monotone_max_subgroup_drift_within_MC_noise=np.nan))
    M=pd.DataFrame(mrows); sens=sens.merge(M,on="family",how="left"); sens.to_csv(out/"sensitivity_analysis_final.csv",index=False)
    M.to_csv(out/"monotonicity_final.csv",index=False)
    input_sha256 = hashlib.sha256(Path(a.a2).read_bytes()).hexdigest()
    json.dump({"reference":a.reference,"temperature_T_chosen_on_validation":getattr(ref,"T",None),"input_a2":str(Path(a.a2).resolve()),
               "input_a2_sha256":input_sha256,"n_train":len(tr),"n_calib":len(cal),"n_test":len(te),
               "cohort_n":n0,"cohort_year":int(c0.year.min()),"occ_changed_rates_[prior0,prior1]":[float(x) for x in rates],
               "reps":a.reps,"steps":a.steps,"paired_random_numbers":"U,V по (репликация, шаг, человек) общие для всех моделей; поток искажений отдельный"},
              open(out/"setup_final.json","w"),indent=1,ensure_ascii=False)
    print(one[one.is_baseline][["model","max_abs_dev_pp_test","cohort_step1_max_abs_dev_pp","brier","auc_nojob","spearman_nojob"]].round(4).to_string())
    print(D[(D.step==a.steps)&D.is_baseline][["model","surv_drift","surv_lo","surv_hi","max_abs_subgroup_drift","max_subgroup"]].round(4).to_string()); print(M.to_string())

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--a2",required=True); ap.add_argument("--out",default="out_final")
    ap.add_argument("--reps",type=int,default=40); ap.add_argument("--steps",type=int,default=5)
    ap.add_argument("--reference",choices=["isotonic","temperature"],default="temperature"); main(ap.parse_args())

import pandas as pd, numpy as np, matplotlib, json; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "results_final/B/"
A = "results_final/A/"
one=pd.read_csv(B+"one_step_metrics_final.csv"); D=pd.read_csv(B+"drift_vs_reference_final.csv"); S=pd.read_csv(B+"subgroup_drift_final.csv")
sens=pd.read_csv(B+"sensitivity_analysis_final.csv"); T=pd.read_csv(B+"trajectories_final.csv"); M=pd.read_csv(B+"monotonicity_final.csv")
DA=pd.read_csv(A+"drift_vs_reference_final.csv"); setup=json.load(open(B+"setup_final.json"))
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
base=[m for m in one[one.is_baseline].model]            # ref + 8 baseline
lab=dict(zip(one.model,one.label))
# 1 траектории
groups=[("Эталон, shuffle, collapse",["ref","shuffle","collapse_0.5","collapse_0.0"]),("Эталон и persistence",["ref","persistence_0.5","persistence_1.0"]),
        ("Эталон и subgroup bias",["ref","subgroup_age_50","subgroup_educ_50","subgroup_sex_40"])]
fig,ax=plt.subplots(1,3,figsize=(13,4),sharey=True); cm=plt.get_cmap("tab10")
for a,(t,ms) in zip(ax,groups):
    for i,m in enumerate(ms):
        g=T[T.model==m].sort_values("step"); c="k" if m=="ref" else cm(i)
        a.plot(g.step,g["mean"],marker="o",ms=3,color=c,label=lab[m]); a.fill_between(g.step,g.lo,g.hi,color=c,alpha=.18)
    a.set_title(t); a.set_xlabel("Шаг симуляции (год)"); a.legend(fontsize=7)
ax[0].set_ylabel("Доля исходной работающей когорты,\nсохранившей основную занятость")
plt.tight_layout(); plt.savefig("fig_trajectories.png",dpi=160); plt.close()
# 2 aggregate drift step5
d5=D[(D.step==5)&D.is_baseline].set_index("model").loc[[m for m in base if m!="ref"]]
fig,ax=plt.subplots(figsize=(7,3.8)); y=np.arange(len(d5))[::-1]
ax.barh(y,100*d5.surv_drift,color="#4c78a8"); ax.errorbar(100*d5.surv_drift,y,xerr=[100*(d5.surv_drift-d5.surv_lo),100*(d5.surv_hi-d5.surv_drift)],fmt="none",ecolor="k",capsize=2)
ax.set_yticks(y); ax.set_yticklabels(d5.label); ax.axvline(0,c="k",lw=.6); ax.set_xlabel("Drift_5 = S_error,5 − S_reference,5, п.п. (95% CI по 40 повторам)")
plt.tight_layout(); plt.savefig("fig_drift5.png",dpi=160); plt.close()
# 3 sensitivity
fams=[("collapse","Collapse: 1−λ"),("persistence","Persistence: доля α"),("subgroup_age","Age bias: δ"),("subgroup_educ","Education bias: δ"),("subgroup_sex","Sex bias: δ")]
fig,ax=plt.subplots(1,5,figsize=(15,3.6),sharey=True)
for a,(f,t) in zip(ax,fams):
    g=sens[sens.family==f].sort_values("strength")
    a.errorbar(g.strength,100*g.drift_step5_survival,yerr=[100*(g.drift_step5_survival-g.drift_step5_ci_lo),100*(g.drift_step5_ci_hi-g.drift_step5_survival)],marker="o",label="drift общей доли",capsize=2)
    a.plot(g.strength,100*g.max_abs_subgroup_drift,marker="s",color="#e45756",label="макс. |drift| подгруппы"); a.axhline(0,c="k",lw=.5); a.set_title(t); a.set_xlabel("сила искажения")
ax[0].set_ylabel("п.п. на шаге 5"); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("fig_sensitivity.png",dpi=160); plt.close()

# ================= таблицы и текст (все числа берутся из CSV последнего запуска) =================
pp=lambda x,n=1:f"{100*x:+.{n}f}"
p2=lambda x:f"{100*x:.2f}"
def md(df): 
    h="| "+" | ".join(df.columns)+" |\n|"+"|".join(["---"]*len(df.columns))+"|\n"
    return h+"\n".join("| "+" | ".join(str(v) for v in r)+" |" for r in df.values)
bm=[m for m in base]
o=one.set_index("model"); d5=D[D.step==5].set_index("model"); d1=D[D.step==1].set_index("model"); d3=D[D.step==3].set_index("model")
# robustness A vs B
a5=DA[DA.step==5].set_index("model")
rob=pd.DataFrame({"Perturbation":[lab[m] for m in bm if m!="ref"],"Drift A, п.п.":[pp(a5.surv_drift[m]) for m in bm if m!="ref"],
    "Drift B, п.п.":[pp(d5.surv_drift[m]) for m in bm if m!="ref"],"Difference B−A, п.п.":[pp(d5.surv_drift[m]-a5.surv_drift[m]) for m in bm if m!="ref"]})
rob.to_csv("robustness_A_vs_B_final.csv",index=False)
ordA=list(a5.loc[[m for m in bm if m!="ref"]].surv_drift.abs().sort_values().index); ordB=list(d5.loc[[m for m in bm if m!="ref"]].surv_drift.abs().sort_values().index)
from scipy.stats import spearmanr, kendalltau
va=[a5.surv_drift[m] for m in ordB]; vb=[d5.surv_drift[m] for m in ordB]; rho=spearmanr(np.abs(va),np.abs(vb))[0]
# таблица шага 1
t1=pd.DataFrame({"Model":[lab[m] for m in bm],"same_job, %":[f"{100*o.share_same_job[m]:.2f}" for m in bm],"different_job, %":[f"{100*o.share_different_job[m]:.2f}" for m in bm],
  "no_job, %":[f"{100*o.share_no_job[m]:.2f}" for m in bm],"Доля сохранивших занятость (шаг 1), %":[f"{100*o.survival_share_step1[m]:.2f}" for m in bm],
  "Макс. отклонение от reference, п.п. (тест 2024–25)":[f"{o.max_abs_dev_pp_test[m]:.2f}" for m in bm],"то же на когорте 2023, п.п.":[f"{o.cohort_step1_max_abs_dev_pp[m]:.2f}" for m in bm]})
bad=one[~one["aggregate_match_within_0.5pp"]][["label","max_abs_dev_pp_test","cohort_step1_max_abs_dev_pp"]]
# Spearman/AUC
sp=pd.DataFrame({"Error":[lab[m] for m in bm],"Spearman no_job":[("не определён (константа)" if pd.isna(o.spearman_nojob[m]) else f"{o.spearman_nojob[m]:.3f}") for m in bm],
  "Spearman different_job":[("не определён (константа)" if pd.isna(o.spearman_different_job[m]) else f"{o.spearman_different_job[m]:.3f}") for m in bm],
  "AUC no_job":[f"{o.auc_nojob[m]:.3f}" for m in bm],"AUC different_job":[f"{o.auc_different_job[m]:.3f}" for m in bm],"Brier":[f"{o.brier[m]:.4f}" for m in bm]})
# траектории
tr=T[T.model.isin(bm)].pivot(index="model",columns="step",values="mean").loc[bm]
trt=pd.DataFrame({"Model":[lab[m] for m in bm],**{f"шаг {k}":[f"{tr.loc[m,k]:.3f}" for m in bm] for k in range(6)}})
# дрейф
nb=[m for m in bm if m!="ref"]
dr=pd.DataFrame({"Error":[lab[m] for m in nb],"Drift step 1, п.п.":[pp(d1.surv_drift[m],2) for m in nb],"Drift step 3, п.п.":[pp(d3.surv_drift[m]) for m in nb],
  "Drift step 5, п.п.":[pp(d5.surv_drift[m]) for m in nb],"95% CI step 5":[f"[{pp(d5.surv_lo[m])}; {pp(d5.surv_hi[m])}]" for m in nb]})
# подгруппы
Sb=S[S.is_baseline].copy(); gs=["age<35","age35-49","age50+","female","male","no_univ","univ"]
sg=pd.DataFrame({"Error":[lab[m] for m in nb],**{g:[ (lambda r:f"{100*r.drift:+.1f}"+("*" if r.is_max_group else ""))(Sb[(Sb.model==m)&(Sb.group==g)].iloc[0]) for m in nb] for g in gs}})
sgm=pd.DataFrame({"Error":[lab[m] for m in nb],"Группа с максимальным расхождением":[d5.max_subgroup[m] for m in nb],
  "Макс. |drift|, п.п.":[f"{100*d5.max_abs_subgroup_drift[m]:.1f}" for m in nb],
  "95% CI этой группы":[ (lambda r:f"[{100*r.ci_lo:+.1f}; {100*r.ci_hi:+.1f}]")(Sb[(Sb.model==m)&(Sb.is_max_group)].iloc[0]) for m in nb]})
# sensitivity
sn=sens.copy(); sn["fam"]=sn.family
famname={"collapse":"Mode collapse (сила = 1−λ)","persistence":"Persistence degradation (сила = α)","subgroup_age":"Age bias (δ)","subgroup_educ":"Education bias (δ)","subgroup_sex":"Sex bias (δ)"}
sn=sn[~sn.model.isin(["collapse_1.0","persistence_0.0"])].sort_values(["family","strength"])
snt=pd.DataFrame({"Семейство":[famname.get(f,f) for f in sn.family],"Сила":[f"{s:.2f}" for s in sn.strength],"Brier":[f"{v:.4f}" for v in sn.brier],
  "AUC no_job":[f"{v:.3f}" for v in sn.auc_nojob],"Spearman no_job":[("—" if pd.isna(v) else f"{v:.3f}") for v in sn.spearman_nojob],
  "Drift step 5, п.п. [95% CI]":[f"{pp(r.drift_step5_survival)} [{pp(r.drift_step5_ci_lo)}; {pp(r.drift_step5_ci_hi)}]" for r in sn.itertuples()],
  "Макс. |drift| подгруппы, п.п.":[f"{100*v:.1f} ({g})" for v,g in zip(sn.max_abs_subgroup_drift,sn.max_subgroup)],
  "Согласование шага 1 ≤0,5 п.п.":["да" if v else "НЕТ" for v in sn["aggregate_match_within_0.5pp"]]})
mt=M.copy()
mt["monotone_agg_drift_strict"]=mt["monotone_agg_drift_strict"].map({True:"да",False:"нет"})
mt["monotone_agg_drift_within_MC_noise"]=mt["monotone_agg_drift_within_MC_noise"].map({True:"да",False:"нет"})
mt["monotone_max_subgroup_drift_strict"]=mt["monotone_max_subgroup_drift_strict"].map({True:"да",False:"нет"})
# Для max subgroup drift не используем aggregate SE: максимум по группам имеет
# отдельную неопределённость, поэтому tolerant/MC-noise проверку здесь не заявляем.
mt["monotone_max_subgroup_drift_within_MC_noise"]="не оценивается"
mt["family"]=mt.family.map(lambda f:famname.get(f,f))
mt=mt.rename(columns={"family":"Семейство",
  "monotone_agg_drift_strict":"drift общей доли не убывает (точечные оценки)",
  "monotone_agg_drift_within_MC_noise":"то же с допуском на MC-шум",
  "monotone_max_subgroup_drift_strict":"макс. subgroup drift не убывает",
  "monotone_max_subgroup_drift_within_MC_noise":"MC-устойчивость max subgroup drift"})
# финальная таблица + классы
sfam=M.set_index("family")
def sres(m):
    f=meta_f[m]
    if f in sfam.index:
        r=sfam.loc[f]; n=int((sens.family==f).sum())-(1 if f in("collapse","persistence") else 0)
        return f"{'монотонно' if r.monotone_agg_drift_strict else 'НЕ монотонно'} по {n} уровням (aggregate drift; с допуском на MC-шум: {'да' if r.monotone_agg_drift_within_MC_noise else 'нет'})"
    return "один уровень (не варьировался)"
meta_f=dict(zip(one.model,one.family))
fin=pd.DataFrame({"Error":[lab[m] for m in nb],"Brier step 1":[f"{o.brier[m]:.4f}" for m in nb],"AUC no_job":[f"{o.auc_nojob[m]:.3f}" for m in nb],
  "Spearman no_job":[("—" if pd.isna(o.spearman_nojob[m]) else f"{o.spearman_nojob[m]:.3f}") for m in nb],
  "Aggregate drift step 5, п.п.":[pp(d5.surv_drift[m]) for m in nb],"Max subgroup drift, п.п.":[f"{100*d5.max_abs_subgroup_drift[m]:.1f} ({d5.max_subgroup[m]})" for m in nb],
  "95% CI (aggregate)":[f"[{pp(d5.surv_lo[m])}; {pp(d5.surv_hi[m])}]" for m in nb],"Sensitivity result":[sres(m) for m in nb]})
cls=[]
for m in nb:
    ag=(d5.surv_lo[m]>0 or d5.surv_hi[m]<0) and abs(d5.surv_drift[m])>=0.01
    rk=pd.isna(o.spearman_nojob[m]) or o.spearman_nojob[m]<0.9
    cp=d5.max_abs_subgroup_drift[m]>=0.05 and abs(d5.surv_drift[m])<0.02
    cls.append((m,ag,rk,cp))
yn=lambda b:"да" if b else "нет"
ct=pd.DataFrame({"Error":[lab[m] for m,_,_,_ in cls],"1. Опасна для агрегированной динамики":[yn(a) for _,a,_,_ in cls],
  "2. Опасна для индивидуального ранжирования":[yn(b) for _,_,b,_ in cls],"3. Опасна в основном для состава подгрупп":[yn(c) for _,_,_,c in cls]})
ct.to_csv("classification_final.csv",index=False)
fin.to_csv("final_table_final.csv",index=False)
brier_rel={m:100*(o.brier[m]/o.brier["ref"]-1) for m in nb}
swaps=[(x,y) for k,x in enumerate(ordA) for y in ordA[k+1:] if ordB.index(x)>ordB.index(y)]
def _ov(x,y): return not (d5.surv_hi[x]<d5.surv_lo[y] or d5.surv_hi[y]<d5.surv_lo[x])
if not swaps: swaptxt="Порядок искажений в A и B совпадает."
else: swaptxt="Точный порядок совпадает не полностью: меняются местами "+"; ".join(f"{lab[x]} и {lab[y]} ({'доверительные интервалы B пересекаются' if _ov(x,y) else 'интервалы B не пересекаются'})" for x,y in swaps)+". Вывод: порядок искажений сохраняется частично; пары, поменявшиеся местами, статистически не различимы, если их интервалы пересекаются."
dif=rob['Difference B−A, п.п.'].map(lambda s:abs(float(s)))
swaptxt+=f" Разность B−A по |drift| составляет от {dif.min():.2f} до {dif.max():.2f} п.п.; B хуже откалиброван на шаге 1 (ECE no_job {o.ece_nojob['ref']:.3f})."
sa=lambda m:sens.set_index("model").loc[m]
T_=setup["temperature_T_chosen_on_validation"]
report=f"""---
title: "Task 2. Какие микроошибки A2 разрушают макро-траекторию (FINAL)"
---

*В рамках данного диагностического эксперимента относительно reference-модели.* Выводы описывают чувствительность симуляции к микроошибкам; они не доказаны для реальной экономики.

## 1. Что измеряется

Исследовательский вопрос: какие ошибки отдельного агента накапливаются в многошаговой симуляции, даже если на шаге 1 агрегированные доли переходов согласованы с reference.

**Reference (основной, B)** построен по `expanded_benchmark.py`: median-импутер + HistGradientBoostingClassifier (max_iter=250, learning_rate=0,05, l2_regularization=1, random_state=42), обучение на target_year ≤ 2021, калибровка 2022–2023, тест 2024–2025, температурное масштабирование. Температура выбирается по log loss на калибровочной выборке перебором сетки 0,5–3,0 (251 точка), а не задаётся вручную; получилось T = {T_}. Reference A (изотоническая калибровка) используется только как проверка устойчивости (раздел 9).

**Показатель траектории** — «доля исходной работающей когорты, сохранившей основную занятость» (employment survival of the initial working cohort), S_t. Когорта — {setup['cohort_n']} человек, работавших в {setup['cohort_year']} году. Это не employment rate населения.

**Paired simulation design.** В каждой из {setup['reps']} репликаций для всех моделей используются одни и те же случайные числа: матрицы U[шаг, человек] (разыгрывание исхода) и V[шаг, человек] (occupation_changed) заданы заранее по seed репликации и индексируются по человеку, а не по позиции в массиве. Случайность самих искажений (перемешивание, выбор людей для persistence) идёт отдельным потоком. Дрейф считается как парная разность reference(seed_i) − perturbation(seed_i); проверка: искажения-тождества (λ = 1 и α = 0) дают нулевую разность во всех репликациях.

## 2. Три уровня качества

- **A. Aggregate calibration** — совпадают ли общие доли same_job, different_job, no_job.
- **B. Individual ranking** — правильно ли риск распределён между людьми (AUC, Spearman-корреляция с reference).
- **C. Dynamic macro behavior** — накопленный эффект через несколько шагов (drift общей доли сохранивших занятость, drift по подгруппам).

Хорошее aggregate matching на шаге 1 не гарантирует ни правильного индивидуального ранжирования, ни правильной динамики через несколько шагов. Далее это проверяется на данных.

## 3. Проверка первого шага (уровень A)

Агрегированные доли переходов на первом шаге согласованы с reference; максимальное отклонение указано в п.п. (по трём классам). Порог согласования 0,5 п.п. выбран нами.

{md(t1)}

"""
if len(bad):
    report+="Не удовлетворяют условию согласования ≤ 0,5 п.п. (показываем явно): "+"; ".join(f"{r.label} (тест {r.max_abs_dev_pp_test:.2f}, когорта {r.cohort_step1_max_abs_dev_pp:.2f} п.п.)" for r in bad.itertuples())+".\n\n"
else: report+="Все искажения из полного набора укладываются в порог 0,5 п.п.\n\n"
report+=f"""В основном наборе (девять строк выше) максимальное отклонение — {max(o.max_abs_dev_pp_test[m] for m in nb):.2f} п.п. на тесте и {max(o.cohort_step1_max_abs_dev_pp[m] for m in nb):.2f} п.п. на когорте; в shuffle и collapse доли совпадают с reference точно (они сохраняют среднее по построению).

## 4. Индивидуальное ранжирование (уровень B)

{md(sp)}

Shuffle сохраняет агрегированные доли с точностью до {o.max_abs_dev_pp_test['shuffle']:.2f} п.п., но Spearman = {o.spearman_nojob['shuffle']:.3f} и AUC no_job = {o.auc_nojob['shuffle']:.3f}: связь риска с конкретными людьми полностью разрушена. При полном mode collapse все вероятности одинаковы, ранговая корреляция не определена, AUC = 0,5.

## 5. Траектории по 5 шагам (уровень C)

![Доля исходной работающей когорты, сохранившей основную занятость; полосы — 95% интервал по 40 повторам](fig_trajectories.png)

{md(trt)}

## 6. Aggregate drift

Drift_5 = S_error,5 − S_reference,5 (парные разности; в п.п.).

{md(dr)}

![Aggregate drift на шаге 5](fig_drift5.png)

## 7. Subgroup drift

Drift_g = S_error,g,5 − S_reference,g,5; * — группа с максимальным |drift|. Значения в п.п. Группы: возраст, пол, образование (no_univ — education < 21, univ — ≥ 21). Доверительные интервалы для всех групп — в `subgroup_drift_final.csv`.

{md(sg)}

{md(sgm)}

Смещения по полу и образованию на базовой силе меняют общую долю не более чем на {max(abs(d5.surv_drift['subgroup_educ_50']),abs(d5.surv_drift['subgroup_sex_40']))*100:.1f} п.п. (Brier хуже на {brier_rel['subgroup_educ_50']:.1f}% и {brier_rel['subgroup_sex_40']:.1f}%), но в одной из подгрупп сдвиг достигает {100*d5.max_abs_subgroup_drift['subgroup_educ_50']:.1f} и {100*d5.max_abs_subgroup_drift['subgroup_sex_40']:.1f} п.п. Смещение по возрасту на базовой силе: Brier хуже на {brier_rel['subgroup_age_50']:.1f}%, drift общей доли {pp(d5.surv_drift['subgroup_age_50'])} п.п., в группе age50+ {100*d5.max_abs_subgroup_drift['subgroup_age_50']:.1f} п.п.

## 8. Sensitivity analysis по силе микроошибки

Смысл параметров: **λ** (collapse) — P_λ = P_mean + λ(P_reference − P_mean), λ = 1 — reference, λ = 0 — полный collapse; сила на графике = 1−λ. **α** (persistence) — доля людей, у которых пять признаков зависимости от прошлого состояния (tenure, history_years, prior_job_change, occupation_changed, wage_change) заменены значениями случайного другого человека когорты; α = 0 — без искажения, α = 1 — у всех. **δ** (subgroup bias) — относительный сдвиг риска no_job: группе A умножается на 1+δ, группе B на 1−δ, затем общая константа возвращает среднее no_job к reference на шаге 1 (age: до 35 лет ×(1+δ), 50+ ×(1−δ); sex: женщины ×(1+δ), мужчины ×(1−δ); education: без университета ×(1+δ), с университетом ×(1−δ)). Базовые силы: age 0,5, education 0,5, sex 0,4 (на графиках sex дополнен уровнями 0,25/0,5/0,75/1,0).

{md(snt)}

{md(mt)}

![Чувствительность drift на шаге 5 к силе искажения](fig_sensitivity.png)

Вывод о монотонности aggregate drift делаем по таблице выше: проверяется, не убывает ли |drift| при усилении искажения по точечным оценкам и с допуском на Monte Carlo-шум. Для max subgroup drift приводится только монотонность точечных оценок, без использования aggregate SE, поскольку максимум по группам имеет отдельную неопределённость. Соседние уровни со слабым различием (например, education 0,25 и 0,5, sex 0,25 и 0,4) не интерпретируются как статистически различающиеся только на основании пересечения интервалов. Subgroup bias при слабой силе почти не виден по Brier, но при δ = 1 Brier хуже на {100*(sa('subgroup_age_100').brier/o.brier['ref']-1):.1f}% (age), {100*(sa('subgroup_educ_100').brier/o.brier['ref']-1):.1f}% (education), {100*(sa('subgroup_sex_100').brier/o.brier['ref']-1):.1f}% (sex), drift общей доли для age достигает {pp(sa('subgroup_age_100').drift_step5_survival)} п.п. и AUC no_job падает до {sa('subgroup_age_100').auc_nojob:.2f}. Значит, утверждение «subgroup bias не влияет на агрегат» верно только для слабых и умеренных смещений по полу и образованию.

## 9. Reference A против Reference B

{md(rob)}

Порядок искажений по |drift| на шаге 5 (по возрастанию) — A: {', '.join(lab[m] for m in ordA)}; B: {', '.join(lab[m] for m in ordB)}. Ранговая корреляция |drift| между A и B: {rho:.2f}. {swaptxt}

## 10. Итоговая таблица

{md(fin)}

Классификация по механизму. Правила заданы нами и применены к рассчитанным результатам: (1) 95% CI drift_5 не включает 0 и |drift_5| ≥ 1 п.п.; (2) Spearman no_job < 0,9 или не определён; (3) max |subgroup drift| ≥ 5 п.п. при |aggregate drift| < 2 п.п. Одна ошибка может попасть в несколько классов.

{md(ct)}

## 11. Главный вывод

Какие микроошибки отдельного агента реально разрушают макротраекторию в рамках данного диагностического эксперимента относительно reference-модели:

1. **Loss of persistence.** Ослабление зависимости от прошлого состояния накапливается во времени: drift общей доли {pp(d5.surv_drift['persistence_0.5'])} п.п. при α = 0,5 и {pp(d5.surv_drift['persistence_1.0'])} п.п. при α = 1 на шаге 5 при отклонении долей на шаге 1 не более {max(o.cohort_step1_max_abs_dev_pp['persistence_0.5'],o.cohort_step1_max_abs_dev_pp['persistence_1.0']):.2f} п.п.
2. **Loss of heterogeneity.** Схлопывание вероятностей к среднему даёт drift {pp(d5.surv_drift['collapse_0.5'])} п.п. при λ = 0,5 и {pp(d5.surv_drift['collapse_0.0'])} п.п. при λ = 0; максимальное расхождение по подгруппам до {100*d5.max_abs_subgroup_drift['collapse_0.0']:.1f} п.п. (группа {d5.max_subgroup['collapse_0.0']}). При λ = 0,5 ранжирование не меняется (Spearman = {o.spearman_nojob['collapse_0.5']:.2f}, AUC прежний), но drift уже {pp(d5.surv_drift['collapse_0.5'])} п.п.: для траектории важна не только упорядоченность риска, но и его амплитуда.
3. **Loss of individual risk ranking.** Shuffle сохраняет агрегированные доли шага 1, но Spearman no_job = {o.spearman_nojob['shuffle']:.2f}, и к шагу 5 drift {pp(d5.surv_drift['shuffle'])} п.п., в группе {d5.max_subgroup['shuffle']} — {100*d5.max_abs_subgroup_drift['shuffle']:.1f} п.п.

**Subgroup bias** на базовой силе почти не виден по aggregate Brier (+{brier_rel['subgroup_age_50']:.1f}…+{max(brier_rel['subgroup_educ_50'],brier_rel['subgroup_sex_40']):.1f}%) и для пола и образования почти не меняет общую долю сохранивших занятость, но заметно меняет состав и результаты подгрупп (до {100*max(d5.max_abs_subgroup_drift['subgroup_age_50'],d5.max_abs_subgroup_drift['subgroup_educ_50'],d5.max_abs_subgroup_drift['subgroup_sex_40']):.1f} п.п.). Для возраста при сильном смещении меняется и общая доля (раздел 8).

## 12. Технические проверки

- **Обновление состояния** после каждого перехода: same_job — tenure += 1, prior_job_change = 0; different_job — tenure = 0, prior_job_change = 1; age += 1, year += 1, history_years += 1; occupation_changed разыгрывается с эмпирическими долями {setup['occ_changed_rates_[prior0,prior1]'][0]:.3f} (после same_job) и {setup['occ_changed_rates_[prior0,prior1]'][1]:.3f} (после different_job), оценёнными по признакам когорты в момент старта; wage_change = 0, зарплата, часы и прочие признаки не меняются. Правила не менялись.
- **Future leakage:** состояние t+1 строится только из состояния t и разыгранного исхода; признаки target-года и последующих волн не используются. Разбиения: обучение target_year ≤ 2021, калибровка 2022–2023, тест ≥ 2024 (в коде есть проверка), температура выбирается только по калибровочной выборке. Исходы тестовой когорты в симуляции не используются.
- **Найдена и исправлена техническая ошибка:** в предыдущей версии случайные числа не были полностью парными: искажения shuffle и persistence расходовали общий генератор до розыгрыша исхода, поэтому числа reference и искажения расходились, а индексирование шло по позиции в массиве выживших. Теперь U и V заданы по человеку и шагу, поток искажения отдельный; эксперимент перезапущен полностью, все числа выше получены этим запуском.
- **Воспроизводимость Reference B:** параметры и порядок обработки совпадают с функциями `fit_a2`, `temperature_scale`, `choose_temperature` из `expanded_benchmark.py` (импутер median, те же гиперпараметры, та же сетка температур и критерий log loss; признаки A2_FEATURES; wage логарифмируется). Температура в отчёте берётся непосредственно из `setup_final.json`, а не задаётся вручную.
- **Входные данные:** SHA-256 текущего `a2_pairs.csv.gz` фиксируется в `setup_final.json`: `{setup.get('input_a2_sha256','не записан')}`. Это позволяет проверить, что все итоговые таблицы получены из одного и того же файла.

## 13. Ограничения

- `no_job` — absorbing state; переходов no_job → job в текущей A2-таблице нет, поэтому показатель не является общим employment rate населения.
- Это диагностический эксперимент относительно reference-модели, а не прогноз реальной пятилетней динамики экономики; реальных многолетних траекторий для валидации reference нет.
- Доверительные интервалы учитывают только Monte Carlo variability (40 повторов); неопределённость обучения модели в них не входит.
- Силы искажений — сценарные параметры; результаты показывают относительную чувствительность симуляции к микроошибкам, а не причинный эффект в реальной экономике.
- Пороги 0,5 п.п., 1 п.п., 2 п.п., 5 п.п. и 0,9 для согласования и классификации выбраны аналитиком.
"""
open("report_final.md","w").write(report)

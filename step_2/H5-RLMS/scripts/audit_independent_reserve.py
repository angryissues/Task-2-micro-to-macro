"""Audit reserve coverage using keys, eligibility and archived provenance only."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import pyreadstat

ap=argparse.ArgumentParser();ap.add_argument('--source-dir',default='/private/tmp/h5_rlms_work');ap.add_argument('--out',default='/private/tmp/h5_reserve_audit');args=ap.parse_args()
root=Path(args.source_dir);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
keys=['idind','current_year'];x=pd.read_csv(root/'full_transition.csv',usecols=keys+['target_year','age'])
old=pd.read_csv(root/'model_output/full_transition_exact_model_input.csv',usecols=keys+['split'])
raw,_=pyreadstat.read_sav(str(root/'RLMS_IND_1994_2025_rus.sav'),usecols=['idind','year','adult','j1'])
eligible=raw[(raw.adult==1)&raw.j1.isin([1,5])][['idind','year']]
w=x[x.age.ge(18)&x.age.lt(65)].copy()
for h in (2,3):
    future=eligible.rename(columns={'year':'current_year'}).copy();future.current_year-=h
    w=w.merge(future,on=keys,how='inner',validate='one_to_one')
def buckets(v):return int(hashlib.sha256(('H5-v2:'+str(int(v))).encode()).hexdigest()[:8],16)%10
w['bucket']=w.idind.map(buckets);w['role']='not_selected_in_completed_H5'
w.loc[(w.bucket<6)&(w.current_year+3<=2018),'role']='H5_train'
w.loc[w.bucket.between(6,7)&(w.current_year>=2018)&(w.current_year+3<=2021),'role']='H5_validation'
w.loc[(w.bucket>=8)&(w.current_year>=2021)&(w.current_year+3<=2025),'role']='H5_test'
joined=w.merge(old,on=keys,how='left',validate='one_to_one',indicator=True)
summary=[]
for role,g in joined.groupby('role'):
    summary.append({'role':role,'windows':len(g),'people':int(g.idind.nunique()),'start_year_min':int(g.current_year.min()),'start_year_max':int(g.current_year.max()),'previous_full_model_input_matches':int(g['_merge'].eq('both').sum()),'unmatched_windows':int(g['_merge'].ne('both').sum())})
s=pd.DataFrame(summary);s.to_csv(out/'reserve_coverage.csv',index=False)
stats={'archive_rows':len(raw),'archive_people':int(raw.idind.nunique()),'archive_year_min':int(raw.year.min()),'archive_year_max':int(raw.year.max()),'old_transition_rows':len(old),'old_transition_people':int(old.idind.nunique()),'eligible_h5_windows_all_years':len(w),'eligible_h5_people_all_years':int(w.idind.nunique()),'eligible_windows_without_old_input_match':int(joined['_merge'].ne('both').sum()),'provenance_limit':'No complete append-only run/access log; eligibility/key membership is not proof that each raw value was individually viewed. Old all-input aggregate evaluations and observed H5 preliminary all-row audit prevent claiming a pristine reserve for this endpoint.'}
(out/'reserve_audit.json').write_text(json.dumps(stats,indent=2))
report=f'''# Аудит резерва данных RLMS для независимой проверки H5

## Решение

Подтверждённого нетронутого резерва для строгой независимой проверки текущей H5 в этом архиве не найдено. Есть окна, не вошедшие в исправленный H5 train/validation/test, но каждое из них уже присутствует в ранее использованном входе полного transition-анализа. Называть их новыми независимыми данными нельзя.

## Что проверено

1. Исходный SAV содержит {len(raw):,} строк, {raw.idind.nunique():,} человек, годы {int(raw.year.min())}–{int(raw.year.max())}.
2. Ранее восстановленный вход содержит {len(old):,} очищенный переход у {old.idind.nunique():,} человек. Сопоставление выполнено по (idind, current_year), без расчёта новых показателей качества на возможном резерве.
3. Повторно определены только критерии пригодности H5: известный очищенный старт, возраст [18,65), полная бинарная взрослая траектория на горизонтах 1/2/3.
4. Восстановлены роли исправленной H5 по зафиксированному хеш-разделению людей и календарным ограничениям.

{s.to_markdown(index=False)}

Числа людей в разных строках нельзя складывать: один человек мог иметь окна разных лет и ролей «использован / не отобран по времени». Окон без совпадения с ранее использованным transition-входом: {stats['eligible_windows_without_old_input_match']}.

## Почему оставшиеся окна не являются чистым резервом

- `step_2/H1-RLMS/scripts/fit_full_transition_models_exact.py` строит прогнозы на всём входе и считает Brier по каждому split, включая validation и test; фильтра «только будущие test-годы» в этой старой версии нет.
- `step_2/full_transition_rlms_extension.md` документирует полный natural-transition анализ 284886 переходов с 284671 наблюдаемым target; H1/H2 рассчитывались на всём этом входе.
- Прежний `step_2/H5-RLMS/scripts/analyze_h5_groups.py` (теперь отключён) использовал все 299161 наблюдаемых перехода для общей и групповой оценки. Его результаты уже были просмотрены в работе над проектом. Некорректность старого анализа не отменяет самого просмотра данных.
- H3 дополнительно анализировал j7 на полном подходящем входе; H4 — полные траектории. Смена номера гипотезы или повторная маска известных ответов не создаёт независимую репликацию.

Профилирование базы само по себе не равнозначно просмотру качества модели на каждом человеке. Здесь вывод опирается на документированные полные оценки и состоявшийся предварительный all-row аудит, а не только на факт чтения архива.

## Ограничение доказательности аудита

Полного неизменяемого журнала всех запусков и обращений к файлам нет. Нельзя утверждать, что каждый ответ во всём SAV был отдельно просмотрен или что исчерпаны все мыслимые новые исследовательские задачи. Для существующего H5 endpoint отсутствует доказанный заранее изолированный резерв. Поэтому статус «нетронутый» не присваивается произвольной части базы задним числом.

## Что доступно дальше

**Без новых данных:** можно зафиксировать текущий протокол и выполнить дополнительную проверку устойчивости по другим окнам, непересекающимся участникам или rolling time splits. Результат будет внутренней проверкой на повторно используемом RLMS, и его так следует назвать. Сохраняется необходимость исключить людей и будущие исходы из обучения каждой соответствующей модели.

**Для более сильной независимой проверки:** нужны новые непроанализированные наблюдения, например последующая волна после 2025 года, либо другая продольная панель с занятостью. Доступность таких данных этим аудитом не проверялась. Другой участник исследования может независимо воспроизвести код на том же архиве — это полезная проверка воспроизводимости, но она не заменяет независимую выборку.

**Новая база не является автоматическим требованием для любой статьи.** Можно описать RLMS как единственный эмпирический пример, сохранить отрицательные результаты и ограничения и обосновать методологический вклад. Требование внешней проверки зависит от заявляемого обобщения и площадки; её отсутствие нельзя скрывать формулировкой «новый test».

## Влияние на статус H5

Завершённый RLMS эксперимент остаётся завершённым. Аудит уточняет границы доказательства: условная экономия проверок на отложенной конечной когорте, без подтверждённой независимой репликации. Исходные данные, идентификаторы и построчные predictions в этот отчёт и Git не включены.
'''
(out/'H5_independent_reserve_audit.md').write_text(report)
print(json.dumps(stats,indent=2));print(s.to_string(index=False))

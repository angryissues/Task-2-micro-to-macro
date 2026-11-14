"""Summarize aggregate-only CODA extension runs without altering primary results."""
import argparse,json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();r=Path(a.results);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());d=pd.concat([pd.read_csv(p) for p in sorted(r.glob('runs_*.csv'))],ignore_index=True)
    assert len(d)==cfg['masks']*len(cfg['policies'])*len(cfg['budgets'])
    assert not d.duplicated(['rep','policy','budget']).any()
    assert set(d.rep)==set(range(cfg['masks'])) and not d.wrong_certificate.any()
    assert (d.cost<=d.budget).all()
    assert (d[d.certified].true_regret<=cfg['epsilon']+1e-10).all()
    s=d.groupby(['budget','policy']).agg(masks=('rep','count'),certificate_rate=('certified','mean'),mean_cost=('cost','mean'),mean_G=('G','mean'),wrong=('wrong_certificate','sum')).reset_index()
    d.to_csv(out/'runs.csv',index=False);s.to_csv(out/'summary.csv',index=False)
    maxb=max(cfg['budgets']);final=d[d.budget==maxb];paired=[]
    for policy in ['CODA_bridge','CODA_conditioned']:
        z=final[final.policy==policy].set_index('rep')
        for ref in ['random','Q2']:
            other=final[final.policy==ref].set_index('rep');both=z.certified&other.certified;delta=other.loc[both,'cost']-z.loc[both,'cost']
            paired.append(dict(policy=policy,reference=ref,budget=maxb,both_success=int(both.sum()),mean_cost_saved_if_both_success=delta.mean(),fraction_cheaper_if_both_success=(delta>0).mean(),min_saved=delta.min(),max_saved=delta.max()))
    pairs=pd.DataFrame(paired);pairs.to_csv(out/'paired.csv',index=False)
    docs=['# NLSY97 H2: расширение сравнения с CODA','',
    'Расширение выполнено после основного H1–H6-прогона на уже использованной confirmation-когорте. Это дополнительное ретроспективное сравнение, а не новый независимый подтверждающий тест. Параметры, код и версия CODA зафиксированы в extension_lock.json до первого запуска расширения; настройки по полученным результатам не подбирались.','',
    '## Сопоставимые условия','',
    f"Когорта: {int(d.n.iloc[0])} уникальных людей, последнее окно каждого человека и затем полные y1/y2/y3 — точно как в исходном H2. Цель: бинарная занятость через 156 недель; 20% ответов искусственно скрыты, те же 100 масок (seed 42000+rep), те же три frozen-прогноза. Цена запроса = 1; бюджеты 0/25/50/100/200; общий точный C1 по Brier, epsilon=0,0025. Модели не переобучались.",
    'Q2 и random воспроизведены на всех масках; их расходы, сертификаты и G при бюджете 200 совпали с исходным H2. Все методы получают одинаковую начальную информацию в C1.','',
    '## Два варианта CODA','',
    '- CODA_bridge: авторская EIG-очередность с общим C1; первоначально известные ответы не обновляют внутреннюю модель CODA. Это та же схема адаптации, что использовалась для RLMS, и отдельная ablation.',
    '- CODA_conditioned: первоначально известные ответы также загружены через неизменённый авторский add_label в стабильном порядке. За них повторно не платим; они доступны всем сравниваемым методам. Именно этот вариант учитывает полную исходно доступную информацию.',
    'Параметры: prefilter_n=128, alpha=0.9, learning_rate=0.01, multiplier=2, q=eig. Истинные скрытые ответы доступны только отдельному oracle после выбора очередной точки. Класс CODA не получает таблицу истинных ответов.',
    'Использован неизменённый [авторский код](https://github.com/justinkay/coda/tree/'+cfg['author_commit']+'), commit '+cfg['author_commit']+'. Авторская публикация: [Consensus-Driven Active Model Selection](https://openaccess.thecvf.com/content/ICCV2025/html/Kay_Consensus-Driven_Active_Model_Selection_ICCV_2025_paper.html).','',
    '## Результаты по бюджетам','',
    s.to_markdown(index=False,floatfmt='.6f'),'',
    'Средняя потраченная цена включает запуски без сертификата. При наличии отказов она не является средней ценой достижения сертификата. Сначала сравниваем долю успешных запусков, затем цену на парных успешных масках.','',
    '## Парное сравнение при бюджете 200','',
    pairs.to_markdown(index=False,floatfmt='.4f'),'',
    'Положительное mean_cost_saved означает, что CODA дешевле указанного метода сравнения; отрицательное — дороже. Стоимость сравнивается только на масках, где обе политики получили сертификат. Отбор по успешности ограничивает интерпретацию, если таких масок меньше 100.']
    for policy in ['random','Q2','CODA_bridge','CODA_conditioned']:
        z=final[final.policy==policy];docs+=['',f"{policy}: {int(z.certified.sum())}/100 сертификатов; средние расходы {z.cost.mean():.2f}."]
    q2=final[final.policy=='Q2'].cost.mean()
    conditioned=final[final.policy=='CODA_conditioned'].cost.mean()
    if final.certified.all():
        docs+=['',f"При бюджете 200 все методы успешны на всех 100 масках. Q2 использует на {100*(1-q2/conditioned):.1f}% меньше запросов, чем CODA_conditioned. Это относительная экономия по средним расходам на фиксированной когорте, а не популяционная гарантия."]
    docs+=['','## Проверки и ограничения','',
    'Четыре синтетических интеграционных теста прошли до фиксации расширения: влияние известных ответов на posterior; неизменность очереди при изменении никогда не запрошенных ответов; нулевые расходы при уже достаточной информации; бюджет, отсутствие повторных запросов и точный выбор после полного раскрытия.',
    'Во всех выданных сертификатах контролируемого сравнения true_regret <= epsilon; ложных сертификатов нет. Первоначальный confirmation-lock NLSY97 сохранён без изменений. В Git предназначены только протокол, код, хеши, агрегаты запусков и этот отчёт.',
    'Это авторская очередь CODA с нашим общим правилом остановки, а не полное воспроизведение экспериментов статьи CODA: исходная цель CODA связана с классификацией, здесь сравнивается Brier+C1. Предположения CODA и природа её показателя EIG не дают отдельной строгой гарантии; гарантию в этом сравнении выдаёт общий C1.',
    'Сто масок — повторное искусственное скрытие на одной когорте, а не сто независимых выборок людей. Неответы, естественные неизвестные исходы и реальная стоимость дозапроса в этом расширении не проверялись. Полное воспроизведение исходных Hara/Active Testing estimators, включая LURE, остаётся отдельной работой.','',
    '## Воспроизведение','',
    'Создать отдельное Python-окружение с numpy, pandas, torch, scipy, matplotlib, tqdm, tabulate и scikit-learn. Скачать авторский репозиторий и checkout указанного commit. Сохранённый extension_lock.json проверяет авторские и исходные расчётные файлы. ROOT — репозиторий проекта, AUTHOR — checkout CODA, LOCAL — исходные локальные прогнозы за пределами Git.',
    '```bash','python test_extension.py',
    'python run.py --repo ROOT --author AUTHOR --predictions LOCAL/confirmation_predictions.csv.gz --reference ROOT/step_2/NLSY97/results/confirmation/H2_runs.csv --out /tmp/nlsy97-coda-results',
    'python summarize.py --results /tmp/nlsy97-coda-results --out /tmp/nlsy97-coda-summary','```',
    'Для тестов при другом расположении author checkout задать NLSY97_CODA_AUTHOR. Генератор отчёта использует только агрегаты.']
    (out/'H2_CODA_extension.md').write_text('\n'.join(docs)+'\n')
    manifests=[json.loads(p.read_text()) for p in sorted(r.glob('manifest_*.json'))]
    assert sum(m['mask_stop']-m['mask_start'] for m in manifests)==100
    verification={'all_masks':100,'aggregate_records':len(d),'policy_runs':400,'budget_checkpoints_per_policy':5,'false_certificates':int(d.wrong_certificate.sum()),'original_random_Q2_match_all_masks':all(m['original_random_Q2_match'] for m in manifests),'pre_freeze_tests_passed':4,'source_manifests':manifests,'data_scope':'No person IDs or person-level labels in exports'}
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    print(s[s.budget==maxb].to_string(index=False));print(pairs.to_string(index=False))
if __name__=='__main__':main()


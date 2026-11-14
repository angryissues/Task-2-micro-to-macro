import argparse,json
from pathlib import Path
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();r=Path(a.results);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);d=pd.read_csv(r/'t0_t1_transfer.csv');p=json.loads((r/'provenance.json').read_text())
table=d[['old_year','new_year','method','selected_model','new_best_model','new_true_regret','n_old','n_new','ess_fraction']].to_markdown(index=False,floatfmt='.6f')
text=f'''# H6 — первые результаты T0/T1

Исторический промежуточный отчёт после пилота двух вариантов из пяти. Полный эксперимент впоследствии завершён: актуальный статус и результаты находятся в H6_final_report.md и H6_closure_status.md.

Модели обучены один раз на {p['train_rows']} переходах {p['train_people']} человек, target_year<=2014. Оценочные люди не пересекаются с обучающими. Группы людей определены стабильным хешем, параметры кандидатов фиксированы для всех временных пар.

{table}

Во всех трёх парах прошлый победитель HGB остался лучшим среди трёх зафиксированных кандидатов. T1 не изменил выбор. Нулевой regret относится к относительному выбору кандидата, а не к безошибочному предсказанию занятости.

T1 обучает logistic domain-classifier только на age/sex/education/current_employment. OOF-разбиение GroupKFold(5) по idind исключает попадание человека одновременно в train и validation доменного классификатора. Отношение плотностей обрезается до [0.2,5]; ESS составляет 96.5–99.2% старой когорты, порог weak_overlap=25% не сработал. Баланс наблюдаемых признаков улучшился, однако эмпирической выгоды для выбора модели в этом пилоте нет. Это не проверка неизменности P(y|x).

Ни T0, ни T1 не выдают сертификат качества на будущей когорте. Старые labels используются для выбора; новые labels только для последующего оценивания. Данные 2016–2024 уже наступили, поэтому это исторический backtest, а не доступ к будущим ответам в реальном времени.

Далее по протоколу: T2/T3 как условные проверки устойчивости, T4 как платная проверка новой конечной когорты, MAR/MNAR и контролируемые контрпримеры. Прежние анализы RLMS означают, что этот эксперимент нельзя представлять как независимую репликацию на ранее не просмотренном архиве.
''';out.write_text(text);print(out)

# H3: статус реального слоя RLMS

## Что уже сделано

Для исходного `RLMS_IND_1994_2025_rus.sav` зафиксировано отдельное
определение реального item nonresponse для вопроса `j7` (работал ли
респондент по основному месту работы за последние 30 дней):

- в знаменатель входят только `adult=1` и `j1 in {1,2,3,4}`;
- значения `j1=5` и отсутствие взрослого опросника считаются структурным
  пропуском;
- специальные коды отказа/не знаю/нет ответа в `j7` считаются неизвестными.

Итог профиля: 212 594 eligible-наблюдения, из них 212 370 наблюдаемых
ответов и 224 явных неизвестных (0,105% eligible). Для 224 неизвестных
сформированы признаки и предсказания трёх exploratory-моделей; значение
`y_j7` намеренно не восстанавливалось.

## Что проверено

На 20%-ном holdout наблюдаемых ответов рассчитан Brier score:

| модель | Brier | размер holdout |
|---|---:|---:|
| status-rate baseline | 0.02565 | 42 474 |
| logistic: age + education + status + year | 0.02877 | 42 474 |
| history: age + education + status | 0.02876 | 42 474 |

Это только проверка пригодности моделей для ранжирования/приоритизации
контактов. Она не доказывает истинные значения для неизвестных строк.

## Граница завершения H3

Контролируемая RLMS-часть H3 уже закрыта отдельным отчётом. Реальный слой
теперь воспроизводимо подготовлен и профилирован, но ещё не является
сертификацией C1 на естественных неизвестных: для 224 строк нет наблюдаемого
`y_j7`, поэтому нельзя честно проверить ошибку сертификата без дополнительного
дизайна верификации (повторный контакт, внешний источник или явное
идентификационное допущение).

Текущий результат следует называть **real-incomplete benchmark / response
model screening**, а не imputed truth и не закрытием H3 на естественной
неполноте.

## Чувствительность к неизвестным

Для 224 неизвестных дополнительно посчитан диапазон без скрытых допущений.
Наблюдаемая доля положительных ответов равна 0,94114. Если все неизвестные
ответы отрицательные, итоговая доля равна 0,94014; если все положительные —
0,94120. Таким образом, даже полный worst/best-case диапазон составляет
около 0,106 процентного пункта.

Model-based expected rates находятся внутри этого диапазона:

- status-rate baseline: 0,94087;
- logistic: 0,94092;
- history: 0,94092.

По статусам неизвестность неравномерна. Для `j1=1` неизвестны 154 из
204 072 наблюдений (0,075%), а для `j1=2` — 67 из 7 049 (0,950%). Поэтому
агрегированный диапазон мал, но при анализе подгруппы `j1=2` граница шире:
доля положительных ответов лежит между 0,03830 и 0,04781 при полном
worst/best-case диапазоне.

Это не заменяет верификацию неизвестных, но показывает, насколько именно
неопределённость по текущему `j7` влияет на агрегированную долю.

## Дизайн верификации

Добавлен воспроизводимый план повторной проверки. При 95% доверии и
консервативном предположении `p=0.5` для оценки доли с точностью ±5 п.п.
нужно проверить 142 из 224 неизвестных без возвращения; для подгруппы `j1=2`
— 58 из 67. Если допустима точность ±10 п.п., плановые объёмы составляют 68
и 40 наблюдений соответственно.

Это объёмы для оценки доли, а не для построчной сертификации. Если требуется
строго подтвердить C1 для всех естественных неизвестных без допущений,
нужна проверка всех 224 строк (или явно обоснованная модельная гарантия).
Протокол фиксирует seed выборки, источник повторного ответа, timestamp и
запрет на перезапись исходного специального кода.

Для практического продолжения подготовлен шаблон
`real_j7_verification_template.csv`. В него нужно заполнить только
`verified_y_j7` (0/1), `verification_source`, `verification_timestamp` и
`verifier_note`. Скрипт `evaluate_real_j7_verification.R` проверяет ключ
`(idind, year)`, отбрасывает незаполненные строки и считает coverage,
положительную долю и Brier score трёх моделей. На пустом шаблоне ожидаемо
получается coverage=0 и отсутствуют Brier scores.

Дополнительно проверена панельная история того же `idind`. У 204 из 224
неизвестных (91,1%) есть хотя бы один наблюдаемый ответ `j7` в другой волне;
у 124 (55,4%) есть ответы и до, и после неизвестной волны. В 100 случаях
соседние наблюдаемые ответы согласуются. Это полезная диагностическая и
приоритизационная информация, но не доказательство истинного ответа в
конкретной неизвестной волне: между волнами значение `j7` могло измениться.

На этой основе построена очередь повторной проверки. Приоритет P1 получили
24 строки с противоречивыми соседними ответами и 20 строк без панельной
истории; P2 — 80 строк только с одним соседом; P3 — 100 строк, где оба
соседних ответа согласуются. Внутри группы строки упорядочены по модельной
неопределённости.

## Артефакты

## Full-data H3 benchmark

Для всех 212 370 наблюдаемых eligible-ответов собран full-data benchmark с
тремя prediction-колонками, бинарным `y_j7` и strata-признаками. На нём
выполнен один sanity-run H3 с MCAR pseudo-nonresponse на уровне общей доли
естественных неизвестных (0,105%), бюджетом 50 и случайной политикой. В
случайно скрытой маске было 245 строк; сертификат получен после одной
успешной дополнительной метки, `G=0`, ложных сертификатов нет.

Это проверка масштабирования и арифметики на полном размере данных, а не
проверка истинных 224 естественных неизвестных: их ответы по-прежнему не
наблюдаются.

- `results/real_profile/real_j7_benchmark_metadata.csv`
- `results/real_profile/real_j7_observed_eligible.csv`
- `results/real_profile/real_j7_unknown_eligible.csv`
- `results/real_profile/real_j7_unknown_predictions.csv`
- `results/real_profile/real_j7_observed_holdout_metrics.csv`
- `results/real_profile/real_j7_unknown_sensitivity.csv`
- `results/real_profile/real_j7_unknown_sensitivity_metadata.csv`
- `results/real_profile/real_j7_subgroup_sensitivity.csv`
- `results/real_profile/real_j7_verification_design.csv`
- `results/real_profile/real_j7_verification_protocol.csv`
- `results/real_profile/real_j7_verification_template.csv`
- `results/real_profile/real_j7_verification_metrics.csv`
- `results/real_profile/real_j7_panel_history_by_unknown.csv`
- `results/real_profile/real_j7_panel_history_summary.csv`
- `results/real_profile/real_j7_verification_queue.csv`
- `results/real_profile/real_j7_verification_queue_summary.csv`
- `results/real_profile/real_full_j7_h3_benchmark.csv`
- `results/real_profile/real_j7_full_h3_input.csv`
- `results/real_profile/real_j7_full_h3_strata.csv`
- `configs/real_full_j7_benchmark.json`
- `scripts/profile_real_rlms.R`
- `scripts/build_real_j7_benchmark.R`
- `scripts/fit_real_j7_models.R`
- `scripts/summarize_real_j7_sensitivity.R`
- `scripts/summarize_real_j7_subgroup_sensitivity.R`
- `scripts/design_real_j7_verification.R`
- `scripts/prepare_real_j7_verification_template.R`
- `scripts/evaluate_real_j7_verification.R`
- `scripts/profile_real_j7_panel_history.R`
- `scripts/rank_real_j7_verification_queue.R`
- `scripts/build_full_real_j7_h3_input.R`

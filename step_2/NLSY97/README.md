# NLSY97 — подготовка и основная внешняя проверка H1–H6

Основной confirmation-прогон H1→H6 выполнен по зафиксированному протоколу. Итоги: `report/replication_summary.md`, общая методика `report/methodology.md`, отдельные отчёты `report/H1_NLSY97.md` … `H6_NLSY97.md`. Подготовительный технический отчёт сохранён как `report/data_acquisition.md`.

Результаты условные и неодинаковые: H2 даёт около 15.6% экономии Q2 против random; H3 C1 работает с контролируемыми неответами; H4 информация пути не улучшила G в данном наборе кандидатов, а дешёвые блоки помогают при фиксированном бюджете; H5 не нашла общей подходящей модели для всех малых/крупных групп; H6 сохранила старого победителя, но не получила строгого нового сертификата при малых бюджетах. Нельзя объявлять все шесть гипотез положительно доказанными.

## Границы завершения

Дополнительно выполнено [сравнение с авторской очередью CODA](extensions/coda/report/H2_CODA_extension.md) на тех же 100 масках H2: 76,05 запроса для bridge и 76,80 с загрузкой известных ответов против 66,15 у Q2. При бюджете 200 все методы получили 100/100 сертификатов, Q2 дешевле на каждой маске. Протокол и код расширения зафиксированы отдельно в extensions/coda/; основной confirmation-lock не менялся. Это расширение на уже использованных данных, а не новый независимый тест.

Выполнена основная внешняя конечнокогортная проверка, включая все T0–T4 H6. Это не полное воспроизведение всех авторских paper-estimators/CODA из большого плана. Месячная/недельная H4-сетка как отдельный confirmatory target, расширенный paper-baseline benchmark и UKHLS/PSID — отдельные расширения. В ходе переноса обнаружен oracle point-gain в старом RLMS H4 P2. Исправление и пересчёт RLMS завершены; см. `report/H4_RLMS_audit.md`.

## Данные и сохранение holdout

Персональные данные находятся вне Git, в `Сергей финашка/NLSY97/`. Полный CSV имеет 8984 человека и 305325 столбцов; выделены 11843 поля. Первоначальный split: train 5392, development 1787, confirmation 1805 человек. `configs/split_protocol_v1.json` сохранён до персонального извлечения.

Затем на train/development исправлены технические ошибки и зафиксированы `configs/experiment_protocol_v1.json` и `configs/confirmation_lock.json` до первых подтверждающих метрик. Это локальная фиксация, а не внешняя preregistration. Код расчёта и общий конфиг после confirmation-lock не менялись; целостность проверяется перед каждым этапом. `results/confirmation/confirmation_access.jsonl` содержит времена первого последовательного расчёта H1–H6. Подтверждающая часть теперь использована; повторный запуск не является новым независимым тестом.

В репозиторий включены только код, протоколы, агрегаты, графики и отчёты. Сырые/выделенные данные, исходные словари с индивидуальными/глобальными частотами и построчные прогнозы в Git не включаются.

## Воспроизведение

LOCAL — отдельная папка вне репозитория; ROOT — корень Task-2-micro-to-macro. Нужны Python с numpy/pandas/scipy/scikit-learn/openpyxl/tabulate/matplotlib.

1. Получить официальные компоненты и выделить переменные скриптами `fetch_members.py`, `prepare_extract.py` (см. технический отчёт и их CLI). Официальный архив: https://nlsinfo.org/cohort-data/nlsy97_all_1997-2023.zip.
2. Скачать официальный crosswalk из Appendix 7: https://nlsinfo.org/content/cohorts/nlsy97/other-documentation/codebook-supplement/appendix-7-continuous-month-scheme.
3. Для каждого split train/development/confirmation:

```
python step_2/NLSY97/scripts/build_panel.py --prepared LOCAL/prepared --crosswalk LOCAL/week_crosswalk.xlsx --out LOCAL/panel --split train
python step_2/NLSY97/scripts/fit_models.py --panels LOCAL/panel --out LOCAL/predictions
```

`build_panel.py` повторить с development/confirmation до fit_models. Временные блоки определяются по концу целевой недели. Основные признаки исключают education; employment-признак предшествует месяцу интервью. Естественные неизвестные outcomes не импутируются.

4. Проверки и development:

```
python step_2/NLSY97/scripts/checks.py --repo ROOT --out LOCAL/checks.json
python step_2/NLSY97/scripts/replicate.py --repo ROOT --predictions LOCAL/predictions --config step_2/NLSY97/configs/experiment_protocol_v1.json --out LOCAL/development --split development
python step_2/NLSY97/scripts/paths_groups.py --repo ROOT --predictions LOCAL/predictions --config step_2/NLSY97/configs/experiment_protocol_v1.json --out LOCAL/development --split development
```

5. Для нового, действительно ещё не открытого confirmation-протокола можно создать lock с `freeze.py --repo ROOT --config CONFIG --out LOCK`. Для воспроизведения ЭТОГО результата использовать сохранённый confirmation_lock.json, не создавать новый якобы независимый holdout задним числом.

```
python step_2/NLSY97/scripts/run_locked.py --repo ROOT --predictions LOCAL/predictions --config step_2/NLSY97/configs/experiment_protocol_v1.json --lock step_2/NLSY97/configs/confirmation_lock.json --out LOCAL/confirmation --local LOCAL/transfer
python step_2/NLSY97/scripts/reports.py --results LOCAL/confirmation --predictions LOCAL/predictions --panels LOCAL/panel --lock step_2/NLSY97/configs/confirmation_lock.json --out LOCAL/report
```

Runner проверяет хеши вычислительных скриптов и общих реализаций H1/H2/H5/H6 перед запуском. Время в журнале нового воспроизведения будет новым; это не меняет факта, что первоначальная confirmation-часть уже использована. Коды endpoint, определение трёхточечного first-exit и ограничения исторической доступности изложены в methodology.md.

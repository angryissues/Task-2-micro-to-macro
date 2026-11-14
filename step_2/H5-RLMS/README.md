# H5-RLMS — проверка подгрупп и стоимость сертификата

Актуальный результат — `report/H5_final_report.md`, данные экспериментов — `results/completed_v2/`. Завершён RLMS scope на горизонтах 1/2/3 года, включая G0/G1/G2, три стратегии запросов, неравномерные пропуски и MAR/MNAR.

В сценариях с гарантированным ответом групповой выбор запросов уменьшил среднюю цену того же сертификата: примерно 12% против глобального Q3 при равномерных пропусках и 58% при повышенных пропусках молодых. Преимущество не универсально; в MNAR-сценарии сертификаты не получены. Числа относятся к конечной отложенной когорте RLMS и контролируемым проверкам известных ответов.

Старый каталог `results/confirmatory/` и скрипты `analyze_h5_groups.py`, `analyze_h5_confirmatory.py` являются недействительными предварительными расчётами: неверное описание временного теста, использование oracle в выборе G0/G2 и недокументированные группы. Их результаты не служат доказательством. Старые числовые таблицы сохранены только локально и исключены из Git; в репозитории оставлено пояснение INVALIDATED.md. Прямой запуск старых скриптов отключён.

## Воспроизведение

Нужны Python с numpy, pandas, scikit-learn, pyreadstat, tabulate, matplotlib и R с haven/dplyr. Версии среды и контрольные суммы исходников сохранены в verification.json. Все микроданные направляйте в отдельную локальную папку, например `/private/tmp/h5_local`, вне репозитория.

1. Извлечь исходный SAV из локального архива.
2. Запустить существующий `step_2/H1-RLMS/scripts/build_full_h1_transition_from_sav.R SAV LOCAL/full_transition.csv` через Rscript.
3. `python step_2/H5-RLMS/scripts/prepare.py --sav SAV --transitions LOCAL/full_transition.csv --out LOCAL/cohorts`
4. `python step_2/H5-RLMS/scripts/analysis.py --input LOCAL/cohorts/test_predictions.csv --out LOCAL/results_test --masks 100 --active-masks 50 --people 100000 --budget 2000`
5. `python step_2/H5-RLMS/scripts/analysis.py --input LOCAL/cohorts/validation_predictions.csv --out LOCAL/results_validation --masks 30 --active-masks 10 --people 300 --budget 500`
6. `python step_2/H5-RLMS/scripts/report.py --results LOCAL/results_test --provenance LOCAL/cohorts/provenance.json --out LOCAL/report`

`--people 100000` включает всех test-участников: фактически 1697. Active-анализ берёт одно последнее окно человека; полный аудит включает все 3037 подходящих окон. Отбор людей не использует истинные outcomes.

Результаты ретроспективные: прежняя часть test уже была просмотрена. Исправленное разделение предотвращает обучение на test, но не позволяет объявить эксперимент новым пререгистрированным исследованием. Естественные неизвестные ответы не восстанавливаются и не оцениваются как известная истина.

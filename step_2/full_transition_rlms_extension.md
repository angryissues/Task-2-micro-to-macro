# Full transition RLMS extension: H1/H2/H3

## Вход

Из полного `RLMS_IND_1994_2025_rus.sav` восстановлен H1 endpoint:
`j1_t -> j1_{t+1}`, где `j1=1` означает работу, `j1=5` — отсутствие работы.
Для natural-nonresponse extension оставлены только переходы, где оба интервью
имеют `adult=1`; специальные nonresponse-коды сохраняются как `target=NA`.

Итоговый вход: 284 886 переходов, 284 671 наблюдаемый target и 215
неизвестных target. Модельные predictions обучены на раннем person-disjoint
периоде и применены ко всему входу. В exact run использован исходный стек H1:
Transition baseline, Logistic Regression и HistGradientBoosting.

## H1: exact C1

На всём входе рассчитаны точные finite-cohort C1 bounds. При равных весах:

- `n_rows=284886`;
- `n_observed=284671`;
- `n_unknown=215`;
- рекомендована модель `p_hist_gradient_boosting`;
- `G=0`;
- сертификат проходит при `epsilon=0.0025`, `0.005` и `0.01`.

## H2: natural-transition active loop

На том же public state с 215 hidden rows прогнаны `random`, `q1` и `q2` при
бюджетах 0, 50, 100 и 200. Все 12 запусков сертифицировались сразу,
рекомендовали `p_hist_gradient_boosting`, с `G=0` и нулевым числом запросов.

Это означает, что в данном полном transition-входе активный выбор не имеет
дополнительной работы: строгая граница уже нулевая. Результат не следует
интерпретировать как доказательство универсального превосходства q1/q2 над
random; это характеристика конкретных frozen predictions и endpoint.

## H3: natural nonresponse

В отличие от controlled H3 здесь 215 строк действительно оставлены
неизвестными. Их labels не импутировались. C1-certification корректна как
finite-cohort bound, но проверить истинный regret на этих 215 строках нельзя
без внешнего ответа. Поэтому natural H3 закрывается как bounds/sensitivity
analysis, а не как проверка восстановленной истины.

Построчные входы и predictions сохранены только локально в `/private/tmp` и
не включаются в Git.

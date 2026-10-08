# Task 2. From real data to testable economic agents

## Research question

Какие ошибки индивидуальных оценок риска перехода занятости,
сохраняя агрегированные доли переходов на первом шаге,
приводят к существенному искажению макродинамики
в многошаговой симуляции?

## Current status

### Step 1 — baseline experiment

Выполнен диагностический эксперимент на данных A2.

Исследованы следующие типы микроошибок:

1. Shuffle индивидуальных risk scores
2. Collapse heterogeneity
3. Persistence degradation
4. Subgroup bias по возрасту
5. Subgroup bias по образованию
6. Subgroup bias по полу

Проведено сравнение двух reference-моделей:

- Reference B — temperature scaling
- Reference A — isotonic calibration

## Reproducibility

- 40 Monte Carlo replications
- 5 simulation steps
- Reference B: temperature scaling, T = 0.97
- Reference A: isotonic calibration
- Cohort: 6684 человека, работавших в 2023 году

## Repository structure

```text
step_1/
    code
    input data
    experiment results
    final report

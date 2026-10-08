# Task 2. From real data to testable economic agents

## Research question

Какие ошибки индивидуальных оценок риска перехода занятости,
сохраняя агрегированные доли переходов на первом шаге,
приводят к существенному искажению макродинамики
в многошаговой симуляции?

## Current status

### Step 1 — baseline experiment

Выполнен диагностический эксперимент на данных A2.

Использованы две reference-модели:

- Reference B — temperature scaling;
- Reference A — isotonic calibration.

Основной эксперимент:

- cohort: 6684 человека, работавших в 2023 году;
- Monte Carlo replications: 40;
- simulation steps: 5;
- paired random numbers;
- aggregate matching threshold: 0.5 п.п.

### Tested micro-errors

1. Shuffle individual risk scores
2. Collapse of heterogeneity
3. Persistence degradation
4. Subgroup bias by age
5. Subgroup bias by education
6. Subgroup bias by sex

## Main result

Некоторые микроошибки практически не видны на первом шаге
по агрегированным долям, но накапливаются в многошаговой
симуляции.

Наиболее выраженный эффект дают:

- loss of persistence;
- loss of heterogeneity;
- loss of individual risk ranking.

Subgroup bias может почти не менять агрегатную динамику,
но существенно менять результаты отдельных групп.

## Repository structure

```text
step_1/
    current experiment, code, data and results

step_2/
    future refinement

step_3/
    future refinement

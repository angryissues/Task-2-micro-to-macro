#!/usr/bin/env python3
"""Run the first controlled RLMS H2 pilot without exposing future labels to policies."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from src.active_selection import (LabelOracle, SelectionState, active_testing_brier_adapter, hara_sampling,
                                  make_stratified_random, q0, q1, q2, q3, random_policy, run_policy,
                                  uncertainty_sampling)


def read_csv(path, required):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not set(required).issubset(rows[0]):
        raise ValueError(f'{path} does not contain required columns {required}')
    return rows


def load_input(input_path, q_path, strata_path, n_rows, seed):
    model_cols = ['p_baseline', 'p_logistic', 'p_hist_gradient_boosting']
    rows = read_csv(input_path, ['idind', 'target_year', 'y', *model_cols])
    q_rows = read_csv(q_path, ['idind', 'target_year', 'q_auxiliary_logistic'])
    q_by_key = {(r['idind'], r['target_year']): float(r['q_auxiliary_logistic']) for r in q_rows}
    if len(q_by_key) != len(q_rows):
        raise ValueError('auxiliary q file has duplicate (idind, target_year) keys')
    missing_q = [r for r in rows if (r['idind'], r['target_year']) not in q_by_key]
    if missing_q:
        raise ValueError(f'auxiliary q is missing {len(missing_q)} input rows')
    strata_rows = read_csv(strata_path, ['idind', 'target_year', 'age', 'current_employment'])
    strata_by_key = {(r['idind'], r['target_year']): r for r in strata_rows}
    missing_strata = [r for r in rows if (r['idind'], r['target_year']) not in strata_by_key]
    if missing_strata:
        raise ValueError(f'strata input is missing {len(missing_strata)} input rows')
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(rows), size=min(n_rows, len(rows)), replace=False))
    chosen = [rows[i] for i in take]
    p = np.array([[float(r[c]) for c in model_cols] for r in chosen])
    y = np.array([int(float(r['y'])) for r in chosen])
    q = np.array([q_by_key[(r['idind'], r['target_year'])] for r in chosen])
    def age_band(age):
        return '18_34' if age < 35 else ('35_49' if age < 50 else '50_plus')
    strata = np.array([
        f"emp_{int(float(strata_by_key[(r['idind'], r['target_year'])]['current_employment']))}__"
        f"age_{age_band(float(strata_by_key[(r['idind'], r['target_year'])]['age']))}"
        for r in chosen
    ])
    return p, y, q, strata, model_cols


def true_regret(predictions, labels, selected_model):
    losses = ((predictions - labels[:, None]) ** 2).mean(axis=0)
    return float(losses[selected_model] - losses.min()) if selected_model is not None else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--q-input', required=True, type=Path)
    parser.add_argument('--strata-input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--mask-start', type=int, default=0)
    parser.add_argument('--mask-count', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    seed = int(config.get('seed', 20261101))
    p, y, q, strata, names = load_input(args.input, args.q_input, args.strata_input,
                                         int(config.get('n_rows', 1000)), seed)
    missing = float(config.get('missing_fraction', .2))
    n_masks = int(config.get('n_masks', 10))
    mask_start = args.mask_start
    mask_stop = n_masks if args.mask_count is None else min(n_masks, mask_start + args.mask_count)
    if mask_start < 0 or mask_start >= n_masks or mask_stop <= mask_start:
        raise ValueError('requested mask range is outside the configured n_masks')
    epsilon = float(config.get('epsilon', .01))
    price_values = np.asarray(config.get('price_values', [1.0]), dtype=float)
    budgets = [float(x) for x in config.get('budgets', [config.get('max_budget', 200)])]
    policies = {'random': random_policy, 'stratified_random': None, 'uncertainty': uncertainty_sampling,
                'hara_brier_adapter': hara_sampling, 'active_testing_brier_adapter': active_testing_brier_adapter,
                'q0': q0, 'q1': q1, 'q2': q2, 'q3': q3}
    wanted = config.get('policies', list(policies))
    output_rows = []
    trace_rows = []
    for mask_seed in range(mask_start, mask_stop):
        rng = np.random.default_rng(seed + mask_seed)
        hidden = np.zeros(len(y), dtype=bool)
        hidden[rng.choice(len(y), size=round(len(y) * missing), replace=False)] = True
        observed = ~hidden
        cost_rng = np.random.default_rng(seed + 50_000 + mask_seed)
        costs = cost_rng.choice(price_values, size=len(y), replace=True)
        for policy_name in wanted:
            for budget in budgets:
                policy_seed = seed + 10_000 + mask_seed
                state = SelectionState.create(p, y, observed, costs=costs)
                policy = (make_stratified_random(strata, np.random.default_rng(policy_seed))
                          if policy_name == 'stratified_random' else policies[policy_name])
                result = run_policy(state, LabelOracle(y), policy, epsilon, budget,
                                    q=q, rng=np.random.default_rng(policy_seed))
                selected = result['recommended_model'] if result['certified'] else None
                output_rows.append({
                    'policy': policy_name, 'mask_seed': mask_seed, 'n_rows': len(y),
                    'missing_fraction': missing, 'epsilon': epsilon, 'max_budget': budget,
                    'price_values': '|'.join(map(str, price_values.tolist())),
                    'status': result['status'], 'certified': result['certified'],
                    'recommended_model': result['recommended_model'], 'selected_model': selected,
                    'selected_model_name': names[selected] if selected is not None else '',
                    'G': result['G'], 'total_cost': result['total_cost'],
                    'queries_obtained': result['queries_obtained'],
                    'true_regret': true_regret(p, y, selected),
                    'wrong_certificate': bool(result['certified'] and true_regret(p, y, selected) > epsilon + 1e-12),
                })
                trace_rows.append({'policy': policy_name, 'mask_seed': mask_seed, 'max_budget': budget,
                                   'trace': result['trace']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    trace_path = args.output.with_name(f'{args.output.stem}_traces.json')
    trace_path.write_text(json.dumps(trace_rows, ensure_ascii=False))
    print(f'wrote {len(output_rows)} runs to {args.output}; traces to {trace_path}')


if __name__ == '__main__':
    main()

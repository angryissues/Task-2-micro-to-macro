#!/usr/bin/env python3
"""Run an H3 controlled RLMS nonresponse development experiment."""
from __future__ import annotations
import argparse, csv, gzip, json
from pathlib import Path
import numpy as np

def expit(x):
    x = np.clip(np.asarray(x, dtype=float), -35, 35)
    return 1.0 / (1.0 + np.exp(-x))

def logit(x):
    x = np.asarray(x, dtype=float)
    return np.log(x / (1.0 - x))
from src.nonresponse import SelectionState, initial_missing_probabilities, run_nonresponse, draw_initial_mask

def read_csv(path, required):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as fh: rows = list(csv.DictReader(fh))
    if not rows or not set(required).issubset(rows[0]):
        raise ValueError(f'{path} does not contain required columns {required}')
    return rows

def load_input(input_path, strata_path, n_rows, seed):
    cols = ['p_baseline', 'p_logistic', 'p_hist_gradient_boosting']
    rows = read_csv(input_path, ['idind', 'target_year', 'y', *cols])
    sr = read_csv(strata_path, ['idind', 'target_year', 'age', 'education'])
    by_key = {(r['idind'], r['target_year']): r for r in sr}
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(rows), min(n_rows, len(rows)), replace=False))
    chosen = [rows[i] for i in take]
    p = np.array([[float(r[c]) for c in cols] for r in chosen])
    y = np.array([int(float(r['y'])) for r in chosen])
    age = np.array([float(by_key[(r['idind'], r['target_year'])]['age']) for r in chosen])
    education_values = [by_key[(r['idind'], r['target_year'])]['education'] for r in chosen]
    numeric_education = np.array([float(x) if str(x).strip() else np.nan for x in education_values])
    numeric_education[np.isnan(numeric_education)] = np.nanmedian(numeric_education)
    education = numeric_education
    low = (education <= np.nanmedian(education)).astype(float)
    return p, y, age, low, cols

def true_regret(predictions, labels, selected):
    if selected is None: return None
    losses = ((predictions - labels[:, None]) ** 2).mean(axis=0)
    return float(losses[selected] - losses.min())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=Path); ap.add_argument('--strata-input', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path); ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--mask-start', type=int, default=0); ap.add_argument('--mask-count', type=int)
    a = ap.parse_args(); cfg = json.loads(a.config.read_text())
    seed = int(cfg.get('seed', 20261108))
    p, y, age, low, names = load_input(a.input, a.strata_input, int(cfg.get('n_rows', 1000)), seed)
    n_masks = int(cfg.get('n_masks', 30)); start = a.mask_start
    stop = n_masks if a.mask_count is None else min(n_masks, start + a.mask_count)
    if start < 0 or start >= n_masks or stop <= start: raise ValueError('invalid mask range')
    mechanism = cfg.get('missing_mechanism', 'mcar'); miss = float(cfg.get('missing_fraction', .2))
    answer_p = float(cfg.get('answer_probability', 1.0)); response_mechanism = cfg.get('response_mechanism', 'mcar')
    budgets = [float(x) for x in cfg.get('budgets', [100, 200])]
    policies = cfg.get('policies', ['r0', 'r1', 'r2']); eps = float(cfg.get('epsilon', .0025))
    lower_margin = float(cfg.get('r2_lower_margin', .15)); exploration = float(cfg.get('r2_exploration', .1))
    out, traces = [], []
    for mask_seed in range(start, stop):
        miss_p = initial_missing_probabilities(age, low, y, mechanism, miss, float(cfg.get('mnar_kappa', 0.0)))
        hidden = draw_initial_mask(miss_p, np.random.default_rng(seed + mask_seed))
        if response_mechanism == 'mcar':
            response_p = np.full(len(y), answer_p)
        elif response_mechanism == 'mar':
            response_p = 1.0 - initial_missing_probabilities(
                age, low, np.zeros_like(y), 'mar', 1.0 - answer_p)
        else:
            raise ValueError('response_mechanism must be mcar or mar in this MVP')
        model_mode = cfg.get('response_model_mode', 'oracle')
        if model_mode == 'oracle':
            policy_p = response_p.copy()
        elif model_mode == 'logit_shift_plus1':
            policy_p = expit(logit(np.clip(response_p, 1e-6, 1 - 1e-6)) + 1.0)
        elif model_mode == 'logit_shift_minus1':
            policy_p = expit(logit(np.clip(response_p, 1e-6, 1 - 1e-6)) - 1.0)
        elif model_mode == 'complement':
            policy_p = 1.0 - response_p
        else:
            raise ValueError('unknown response_model_mode')
        lower_p = np.maximum(0.0, policy_p - lower_margin)
        for policy in policies:
            for budget in budgets:
                state = SelectionState.create(p, y, ~hidden)
                initial_observed = int((~hidden).sum())
                res = run_nonresponse(state, y, response_p, policy, eps, budget,
                                      seed=seed + 10000 * mask_seed,
                                      lower_answer_probabilities=lower_p, exploration=exploration,
                                      policy_answer_probabilities=policy_p)
                selected = res['recommended_model'] if res['certified'] else None
                regret = true_regret(p, y, selected)
                out.append({'policy': policy, 'mask_seed': mask_seed, 'n_rows': len(y),
                            'missing_mechanism': mechanism, 'initial_missing_fraction': float(hidden.mean()),
                            'response_mechanism': response_mechanism,
                            'answer_probability_mean': float(response_p.mean()), 'epsilon': eps, 'max_budget': budget,
                            'status': res['status'], 'certified': res['certified'], 'selected_model': selected,
                            'selected_model_name': names[selected] if selected is not None else '', 'G': res['G'],
                            'total_cost': res['total_cost'], 'queries_attempted': res['queries_attempted'],
                            'labels_obtained': res['labels_obtained'],
                            'initial_labels': initial_observed,
                            'additional_labels': res['labels_obtained'] - initial_observed,
                            'true_regret': regret,
                            'wrong_certificate': bool(res['certified'] and regret > eps + 1e-12)})
                traces.append({'policy': policy, 'mask_seed': mask_seed, 'max_budget': budget, 'trace': res['trace']})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    a.output.with_name(a.output.stem + '_traces.json').write_text(json.dumps(traces, ensure_ascii=False))
    print(f'wrote {len(out)} runs to {a.output}')

if __name__ == '__main__': main()

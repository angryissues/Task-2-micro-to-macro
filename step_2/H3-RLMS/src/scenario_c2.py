"""Scenario C2 state for H3 R3.

The Gamma bounds are assumptions about the initially hidden labels, not
distribution-free guarantees. They are frozen before adaptive attempts.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

H1_BOUNDS = Path(__file__).resolve().parents[2] / "H1-RLMS" / "src" / "bounds.py"
spec = importlib.util.spec_from_file_location("h1_bounds_for_h3", H1_BOUNDS)
mod = importlib.util.module_from_spec(spec); sys.modules["h1_bounds_for_h3"] = mod
assert spec.loader is not None; spec.loader.exec_module(mod)
c2_bounds = mod.c2_bounds
scenario_count_bounds = mod.scenario_count_bounds


@dataclass
class ScenarioState:
    predictions: np.ndarray
    labels: np.ndarray
    observed: np.ndarray
    group_ids: np.ndarray
    count_bounds: dict
    costs: np.ndarray
    weights: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    g_all: np.ndarray
    spent: float = 0.0

    @classmethod
    def create(cls, predictions, labels, observed, group_ids, gamma, costs=None):
        p = np.asarray(predictions, float); y = np.asarray(labels, float)
        seen = np.asarray(observed, bool); gid = np.asarray(group_ids)
        n = len(y); c = np.ones(n) if costs is None else np.asarray(costs, float)
        w = np.full(n, 1.0 / n)
        bounds = {}
        for g in np.unique(gid):
            visible = seen & (gid == g); hidden = (~seen) & (gid == g)
            bounds[int(g)] = scenario_count_bounds(int(y[visible].sum()), int(visible.sum()),
                                                    int(hidden.sum()), float(gamma))
        public_y = np.zeros(n); public_y[seen] = y[seen]
        lo, up, g_all, status = c2_bounds(p, public_y, seen, gid, bounds, w)
        if status != "ok": raise ValueError("initial scenario bounds infeasible")
        return cls(p, public_y, seen.copy(), gid, bounds, c, w, lo, up, g_all)

    @property
    def n_rows(self): return self.predictions.shape[0]
    @property
    def n_models(self): return self.predictions.shape[1]
    @property
    def hidden_indices(self): return np.flatnonzero(~self.observed)
    @property
    def value(self): return float(np.min(self.g_all))
    @property
    def worst_regret(self): return self.g_all

    def certificate(self, epsilon):
        model = int(np.argmin(self.g_all)); g = float(self.g_all[model])
        return {"recommended_model": model, "G": g, "certified": bool(g <= epsilon + 1e-12),
                "guarantee_type": "scenario_shift"}

    def _copy_with(self, row_id, label):
        y = self.labels.copy(); seen = self.observed.copy(); y[row_id] = label; seen[row_id] = True
        b = dict(self.count_bounds); g = self.group_ids[row_id]
        if int(g) in b:
            remaining = int(np.sum((~seen) & (self.group_ids == g)))
            a, z = b[int(g)]; b[int(g)] = (max(0, int(a) - int(label)), min(remaining, int(z) - int(label)))
        lo, up, ga, status = c2_bounds(self.predictions, y, seen, self.group_ids, b, self.weights)
        return y, seen, b, lo, up, ga, status

    def projected_value(self, row_id, label):
        *_, ga, status = self._copy_with(int(row_id), int(label))
        return float(np.min(ga)) if status == "ok" else float("inf")

    def reveal(self, row_id, label):
        y, seen, b, lo, up, ga, status = self._copy_with(int(row_id), int(label))
        if status != "ok": raise ValueError("scenario assumption became infeasible")
        self.labels, self.observed, self.count_bounds = y, seen, b
        self.lower, self.upper, self.g_all = lo, up, ga
        self.spent += float(self.costs[row_id])

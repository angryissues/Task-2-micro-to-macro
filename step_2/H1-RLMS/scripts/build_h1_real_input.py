#!/usr/bin/env python3
"""
Build the real-mode input for H1 from the existing Task 2 train/test splits.
Models and preprocessing mirror experiments_clean-3(2).ipynb.
Run from the Task 2 project root or pass --data-dir.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["age", "sex", "education", "current_employment"]
TARGET = "target"

def make_preprocessor(dense: bool):
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False if dense else True,
        )),
    ])
    return ColumnTransformer(
        [
            ("num", num, ["age"]),
            ("cat", cat, ["sex", "education", "current_employment"]),
        ],
        verbose_feature_names_out=False,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="step_2/data_processed",
                    help="folder containing train.csv.gz and test.csv.gz")
    ap.add_argument("--out", default="h1_real_input.csv.gz")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv.gz"
    test_path = data_dir / "test.csv.gz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. "
            "Pass --data-dir pointing to data_processed."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required = {"idind", "target_year", TARGET, *FEATURES}
    for name, df in [("train", train), ("test", test)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
            raise ValueError(f"{name}.{TARGET} must contain only 0/1 labels")
    if not set(train["idind"]).isdisjoint(set(test["idind"])):
        raise ValueError("Person IDs overlap between train and test")
    if not train["target_year"].max() < test["target_year"].min():
        raise ValueError("Expected train years to precede test years")

    # Same baseline as in the notebook: transition probabilities estimated on train only.
    transition_probabilities = train.groupby("current_employment")[TARGET].mean().to_dict()
    baseline = test["current_employment"].map(transition_probabilities).to_numpy(float)
    if not np.isfinite(baseline).all():
        raise ValueError("A test current_employment category was absent from train")

    X_train = train[FEATURES].copy()
    y_train = train[TARGET].astype(int)

    logistic_model = Pipeline([
        ("preprocessor", make_preprocessor(dense=False)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    logistic_model.fit(X_train, y_train)

    hgb_model = Pipeline([
        ("preprocessor", make_preprocessor(dense=True)),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])
    hgb_model.fit(X_train, y_train)

    p_logistic = logistic_model.predict_proba(test[FEATURES])[:, 1]
    p_hgb = hgb_model.predict_proba(test[FEATURES])[:, 1]

    # H1 real_mode.py expects y and p_* columns. Metadata is retained for traceability.
    result = pd.DataFrame({
        "idind": test["idind"].to_numpy(),
        "target_year": test["target_year"].to_numpy(),
        "y": test[TARGET].astype(int).to_numpy(),
        "p_baseline": baseline,
        "p_logistic": p_logistic,
        "p_hist_gradient_boosting": p_hgb,
    })
    pcols = ["p_baseline", "p_logistic", "p_hist_gradient_boosting"]
    P = result[pcols].to_numpy(float)
    if not np.isfinite(P).all() or ((P < 0) | (P > 1)).any():
        raise ValueError("Forecasts must be finite probabilities in [0, 1]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, compression="gzip")

    brier = {c: float(np.mean((result["y"].to_numpy() - result[c].to_numpy()) ** 2))
             for c in pcols}
    print(f"Saved: {out.resolve()}")
    print(f"Rows: {len(result):,}; unique people: {result['idind'].nunique():,}")
    print("Forecast columns:", ", ".join(pcols))
    print("Test Brier scores (descriptive check):")
    for c, v in brier.items():
        print(f"  {c}: {v:.6f}")
    print("Note: labels are all known here. This file is suitable for a baseline real-mode run;")
    print("      use --artificial only as a separate experiment hiding additional known labels.")

if __name__ == "__main__":
    main()

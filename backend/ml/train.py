"""Train and compare models, then persist the best one + metadata.

Run from the backend/ directory:

    python -m ml.train

Produces (in backend/artifacts/):
    model.pkl            best estimator + scaler + feature columns
    model_metadata.json  metrics for every model, winner, feature importance
    wc_matches.parquet   cleaned WC matches used by the API for live features
"""
from __future__ import annotations

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACTS_DIR, CLASS_LABELS, FEATURE_COLUMNS, FEATURE_DISPLAY,
    MODEL_PATH, METADATA_PATH, save_wc_matches,
)
from .data_prep import load_world_cup_matches
from .features import build_training_frame

warnings.filterwarnings("ignore")

VALIDATION_FROM_YEAR = 2014  # last few tournaments held out for validation


def _gradient_boost():
    """Prefer XGBoost, fall back to LightGBM, then sklearn GradientBoosting."""
    try:
        from xgboost import XGBClassifier
        return "XGBoost", XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            tree_method="hist", random_state=42,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        return "LightGBM", LGBMClassifier(
            n_estimators=400, max_depth=-1, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        return "GradientBoosting", GradientBoostingClassifier(random_state=42)


def _candidates():
    name, gb = _gradient_boost()
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=1.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        name: gb,
    }


def _feature_importance(model, scaler, X_val) -> dict:
    """Best-effort importance, normalised to sum to 1."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_)).mean(axis=0)
    else:
        imp = np.ones(len(FEATURE_COLUMNS))
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return {
        FEATURE_COLUMNS[i]: round(float(imp[i]), 4)
        for i in np.argsort(imp)[::-1]
    }


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading World Cup matches...")
    wc = load_world_cup_matches()
    print(f"  {len(wc)} World Cup finals matches "
          f"({wc['year'].min()}-{wc['year'].max()})")

    print("Engineering features...")
    frame = build_training_frame(wc)

    train = frame[frame["year"] < VALIDATION_FROM_YEAR]
    valid = frame[frame["year"] >= VALIDATION_FROM_YEAR]
    if len(valid) < 20:  # tiny dataset safety net
        cut = int(len(frame) * 0.8)
        train, valid = frame.iloc[:cut], frame.iloc[cut:]

    X_train, y_train = train[FEATURE_COLUMNS].values, train["target"].values
    X_val, y_val = valid[FEATURE_COLUMNS].values, valid["target"].values
    print(f"  train={len(X_train)} rows, validation={len(X_val)} rows")

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    results, fitted = {}, {}
    for name, clf in _candidates().items():
        print(f"Training {name}...")
        clf.fit(X_train_s, y_train)
        proba = clf.predict_proba(X_val_s)
        pred = proba.argmax(axis=1)
        results[name] = {
            "accuracy": round(float(accuracy_score(y_val, pred)), 4),
            "f1_macro": round(float(f1_score(y_val, pred, average="macro")), 4),
            "log_loss": round(float(log_loss(y_val, proba, labels=[0, 1, 2])), 4),
        }
        fitted[name] = clf
        print(f"  acc={results[name]['accuracy']} "
              f"f1={results[name]['f1_macro']} "
              f"logloss={results[name]['log_loss']}")

    # Winner = lowest validation log loss (well-calibrated probabilities matter
    # for an explainable probability app).
    best_name = min(results, key=lambda n: results[n]["log_loss"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # Persist the model bundle.
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": best_model,
                "scaler": scaler,
                "model_name": best_name,
                "feature_columns": FEATURE_COLUMNS,
                "class_labels": CLASS_LABELS,
            },
            f,
        )

    # Persist clean WC matches for the API's live feature builder.
    wc_path = save_wc_matches(wc)

    metadata = {
        "best_model": best_name,
        "validation_from_year": VALIDATION_FROM_YEAR,
        "n_matches": int(len(wc)),
        "n_train_rows": int(len(X_train)),
        "n_val_rows": int(len(X_val)),
        "metrics": results,
        "class_labels": CLASS_LABELS,
        "feature_importance": _feature_importance(best_model, scaler, X_val_s),
        "feature_display": FEATURE_DISPLAY,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved -> {MODEL_PATH.name}, {METADATA_PATH.name}, "
          f"{wc_path.name}")


if __name__ == "__main__":
    main()

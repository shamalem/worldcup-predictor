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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACTS_DIR, CLASS_LABELS, FEATURE_COLUMNS, FEATURE_DISPLAY,
    MODEL_PATH, METADATA_PATH, SCORE_MODEL_PATH, SCORE_METADATA_PATH,
    ELO_RATINGS_PATH, FORM_RATINGS_PATH, save_wc_matches,
)
from .data_prep import load_world_cup_matches
from .features import build_training_frame
from .score_model import fit_score_model, evaluate as evaluate_score
from .elo import load_all_internationals, compute_elo
from .form import compute_form

# Recency: a World Cup this many years in the past gets half the training weight.
# 4 years = one World Cup cycle. Swept empirically: this maximises validation
# accuracy while keeping log loss flat; shorter half-lives (2-3y) overfit and hurt.
RECENCY_HALF_LIFE_YEARS = 4.0

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


def _param_grid(name: str) -> dict:
    """Small, sensible hyperparameter grids per model (kept modest so the
    build/train step stays fast)."""
    if name == "LogisticRegression":
        return {"C": [0.1, 0.3, 1.0, 3.0]}
    if name == "RandomForest":
        return {"max_depth": [6, 10, None], "min_samples_leaf": [1, 3, 5]}
    if name == "XGBoost":
        return {"max_depth": [3, 4], "learning_rate": [0.03, 0.05],
                "n_estimators": [300, 500], "min_child_weight": [1, 3]}
    if name == "LightGBM":
        return {"max_depth": [-1, 6], "learning_rate": [0.03, 0.05],
                "n_estimators": [300, 500]}
    if name == "GradientBoosting":
        return {"max_depth": [2, 3], "learning_rate": [0.05, 0.1],
                "n_estimators": [200, 400]}
    return {}


def _tune(name: str, estimator, X, y):
    """Grid-search hyperparameters with 4-fold CV on log loss; return the best
    estimator (unfitted) and its params. Falls back to the base estimator on
    any failure so training never breaks."""
    grid = _param_grid(name)
    if not grid:
        return estimator, {}
    try:
        search = GridSearchCV(
            estimator, grid, scoring="neg_log_loss", cv=4, n_jobs=-1)
        search.fit(X, y)
        return search.best_estimator_, search.best_params_
    except Exception as e:
        print(f"  [warn] tuning {name} failed ({e}); using defaults")
        return estimator, {}


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


def _compute_elo_safe():
    """Compute Elo + recent form from all internationals; degrade gracefully."""
    try:
        all_df = load_all_internationals()
        pre_match, final_ratings = compute_elo(all_df)
        pre_form, final_form = compute_form(all_df)
        print(f"  rated {len(final_ratings)} teams from {len(all_df)} internationals")
        return pre_match, final_ratings, pre_form, final_form
    except Exception as e:
        print(f"  [warn] Elo/form unavailable ({e}); training without them")
        return None, None, None, None


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading World Cup matches...")
    wc = load_world_cup_matches()
    print(f"  {len(wc)} World Cup finals matches "
          f"({wc['year'].min()}-{wc['year'].max()})")

    print("Computing Elo + recent-form ratings over all internationals...")
    pre_match, final_ratings, pre_form, final_form = _compute_elo_safe()

    print("Engineering features...")
    frame = build_training_frame(wc, pre_match=pre_match,
                                 final_ratings=final_ratings,
                                 pre_form=pre_form, final_form=final_form)

    train = frame[frame["year"] < VALIDATION_FROM_YEAR]
    valid = frame[frame["year"] >= VALIDATION_FROM_YEAR]
    if len(valid) < 20:  # tiny dataset safety net
        cut = int(len(frame) * 0.8)
        train, valid = frame.iloc[:cut], frame.iloc[cut:]

    X_train, y_train = train[FEATURE_COLUMNS].values, train["target"].values
    X_val, y_val = valid[FEATURE_COLUMNS].values, valid["target"].values
    print(f"  train={len(X_train)} rows, validation={len(X_val)} rows")

    # Recency weighting: recent World Cups inform the model more than 1950s ones.
    # weight = 0.5 ** (years_ago / half_life)
    max_year = int(frame["year"].max())
    years_ago = (max_year - train["year"].values).astype(float)
    sample_weight = 0.5 ** (years_ago / RECENCY_HALF_LIFE_YEARS)
    print(f"  recency weighting on (half-life {RECENCY_HALF_LIFE_YEARS:.0f}y)")

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    results, fitted = {}, {}
    for name, clf in _candidates().items():
        print(f"Tuning + training {name}...")
        # 1) Find good hyperparameters via CV (without sample weights).
        best_est, best_params = _tune(name, clf, X_train_s, y_train)
        if best_params:
            print(f"  best params: {best_params}")
        # 2) Refit the chosen config with recency sample weights.
        try:
            best_est.fit(X_train_s, y_train, sample_weight=sample_weight)
        except TypeError:
            best_est.fit(X_train_s, y_train)
        proba = best_est.predict_proba(X_val_s)
        pred = proba.argmax(axis=1)
        results[name] = {
            "accuracy": round(float(accuracy_score(y_val, pred)), 4),
            "f1_macro": round(float(f1_score(y_val, pred, average="macro")), 4),
            "log_loss": round(float(log_loss(y_val, proba, labels=[0, 1, 2])), 4),
        }
        fitted[name] = best_est
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

    # Persist current Elo ratings for inference-time strength lookup.
    if final_ratings:
        import json as _json
        ELO_RATINGS_PATH.write_text(
            _json.dumps({k: round(v, 2) for k, v in final_ratings.items()}, indent=2))
    # Persist current recent-form ratings for inference.
    if final_form:
        import json as _json
        FORM_RATINGS_PATH.write_text(
            _json.dumps({k: round(v, 4) for k, v in final_form.items()}, indent=2))

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

    # ------------------------------------------------------------------ #
    # Also train the exact-scoreline (Poisson + Dixon-Coles) model.
    # ------------------------------------------------------------------ #
    try:
        _train_score_model(wc)
    except Exception as e:  # never let the score model break the main train
        print(f"[warn] score model training skipped: {e}")


def _train_score_model(wc: pd.DataFrame) -> None:
    """Fit the Poisson/Dixon-Coles scoreline model and save its artifacts.

    Attack/defence strengths are fitted over *all* internationals (recency-
    weighted), which is denser and more current than World-Cup matches alone.
    Strengths are fitted on pre-validation data for honest held-out metrics,
    then the served model is refit on everything.
    """
    print("\nTraining scoreline model (Poisson + Dixon-Coles)...")
    try:
        all_df = load_all_internationals()
        all_df["_yr"] = all_df["date"].dt.year
    except Exception as e:
        print(f"  [warn] all-internationals unavailable ({e}); using WC-only strengths")
        all_df = None

    half_life = 8.0
    train_df = wc[wc["year"] < VALIDATION_FROM_YEAR]
    val_df = wc[wc["year"] >= VALIDATION_FROM_YEAR]
    if len(train_df) < 50 or len(val_df) < 10:
        train_df, val_df = wc, wc

    # Honest eval: strengths from internationals strictly before validation.
    eval_strength = None
    if all_df is not None:
        eval_strength = all_df[all_df["_yr"] < VALIDATION_FROM_YEAR]
    eval_model = fit_score_model(
        train_df, strength_df=eval_strength, recency_half_life=half_life,
        ref_year=VALIDATION_FROM_YEAR - 1)
    metrics = evaluate_score(eval_model, val_df)
    print(f"  exact-score acc={metrics.get('exact_score_accuracy')} "
          f"outcome acc={metrics.get('outcome_accuracy')} "
          f"goals MAE={metrics.get('total_goals_mae')}  rho={eval_model.rho}")

    # Served model: refit strengths on ALL internationals (most current).
    served = fit_score_model(
        wc, strength_df=all_df, recency_half_life=half_life)
    with open(SCORE_MODEL_PATH, "wb") as f:
        pickle.dump(served, f)

    score_metadata = {
        "model_name": "Poisson + Dixon-Coles",
        "n_matches": int(len(wc)),
        "strength_source": "all internationals (recency-weighted)" if all_df is not None else "World Cup only",
        "recency_half_life_years": half_life,
        "league_avg_goals": round(served.league_avg, 4),
        "host_multiplier": round(served.host_mult, 4),
        "rho": round(served.rho, 4),
        "validation_from_year": VALIDATION_FROM_YEAR,
        "metrics": metrics,
    }
    with open(SCORE_METADATA_PATH, "w") as f:
        json.dump(score_metadata, f, indent=2)
    print(f"Saved -> {SCORE_MODEL_PATH.name}, {SCORE_METADATA_PATH.name}")


if __name__ == "__main__":
    main()

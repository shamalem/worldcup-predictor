"""Explainability returns readable, team-named reasons."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml.config import FEATURE_COLUMNS
from ml.features import build_training_frame
from ml.explain import explain, confidence_level


def test_explanations_are_team_named(fake_wc_df):
    frame = build_training_frame(fake_wc_df)
    X = frame[FEATURE_COLUMNS].values
    y = frame["target"].values
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(
        scaler.transform(X), y)
    bundle = {"model": model, "scaler": scaler,
              "feature_columns": FEATURE_COLUMNS, "model_name": "RandomForest"}

    row = dict(zip(FEATURE_COLUMNS, X[0]))
    x_scaled = scaler.transform([X[0]])
    proba = model.predict_proba(x_scaled)[0]
    reasons, contribs = explain(bundle, row, x_scaled, proba.tolist(),
                                "Brazil", "Germany")
    assert 1 <= len(reasons) <= 4
    assert all(isinstance(s, str) and s for s in reasons)
    assert any(("Brazil" in s or "Germany" in s) for s in reasons)
    assert confidence_level(proba.tolist()) in {"High", "Medium", "Low"}

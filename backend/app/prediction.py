"""Prediction service. Loads the trained artifacts once and serves predictions."""
from __future__ import annotations

import json
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from ml.config import (
    MODEL_PATH, METADATA_PATH, STAGE_ORDER, CLASS_LABELS,
    ELO_RATINGS_PATH, FORM_RATINGS_PATH, wc_matches_available, load_wc_matches,
)
from ml.data_prep import list_teams, list_years
from ml.features import build_feature_vector
from ml.explain import explain, confidence_level


class ModelService:
    def __init__(self) -> None:
        self.bundle = None
        self.metadata: dict = {}
        self.wc: Optional[pd.DataFrame] = None
        self.elo_ratings: dict = {}
        self.form_ratings: dict = {}
        self._load()

    # ----------------------------------------------------------------- load --
    def _load(self) -> None:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.bundle = pickle.load(f)
        if METADATA_PATH.exists():
            self.metadata = json.loads(METADATA_PATH.read_text())
        if wc_matches_available():
            self.wc = load_wc_matches()
        if ELO_RATINGS_PATH.exists():
            self.elo_ratings = json.loads(ELO_RATINGS_PATH.read_text())
        if FORM_RATINGS_PATH.exists():
            self.form_ratings = json.loads(FORM_RATINGS_PATH.read_text())

    @property
    def ready(self) -> bool:
        return self.bundle is not None and self.wc is not None

    # --------------------------------------------------------------- catalog --
    def teams(self) -> list[str]:
        return list_teams(self.wc) if self.wc is not None else []

    def years(self) -> list[int]:
        return list_years(self.wc) if self.wc is not None else []

    def stages(self) -> list[str]:
        return list(STAGE_ORDER.keys())

    # --------------------------------------------------------------- predict --
    def predict(
        self,
        team_a: str,
        team_b: str,
        year: int,
        stage: str,
        neutral: bool,
        host: str = "none",
    ) -> dict:
        if not self.ready:
            raise RuntimeError(
                "Model not trained. Run `python -m ml.train` first.")
        if team_a == team_b:
            raise ValueError("Team A and Team B must be different.")

        a_is_host = host == "A"
        b_is_host = host == "B"
        effective_neutral = bool(neutral) and host == "none"

        row = build_feature_vector(
            self.wc, team_a, team_b, year, stage,
            neutral=effective_neutral, a_is_host=a_is_host, b_is_host=b_is_host,
            final_ratings=self.elo_ratings or None,
            final_form=self.form_ratings or None,
        )
        x = np.array([[row[c] for c in self.bundle["feature_columns"]]], dtype=float)
        x_scaled = self.bundle["scaler"].transform(x)

        proba = self.bundle["model"].predict_proba(x_scaled)[0]
        # Ensure 3-class ordering [A win, draw, B win].
        proba = np.asarray(proba, dtype=float)
        predicted_idx = int(np.argmax(proba))

        reasons, contributions = explain(
            self.bundle, row, x_scaled, proba.tolist(), team_a, team_b)

        result_text = {
            0: f"{team_a} win",
            1: "Draw",
            2: f"{team_b} win",
        }[predicted_idx]

        return {
            "team_a": team_a,
            "team_b": team_b,
            "predicted_result": result_text,
            "probabilities": {
                "team_a": round(float(proba[0]), 4),
                "draw": round(float(proba[1]), 4),
                "team_b": round(float(proba[2]), 4),
            },
            "confidence": confidence_level(proba.tolist()),
            "reasons": reasons,
            "contributions": contributions,
            "model_name": self.bundle.get("model_name", "unknown"),
        }


# Singleton used by routers.
service = ModelService()

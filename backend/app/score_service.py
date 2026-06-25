"""Scoreline prediction service. Loads the Poisson/Dixon-Coles model once."""
from __future__ import annotations

import json
import pickle
from typing import Optional

from ml.config import SCORE_MODEL_PATH, SCORE_METADATA_PATH
from ml.score_model import ScoreModel, predict_score


class ScoreService:
    def __init__(self) -> None:
        self.model: Optional[ScoreModel] = None
        self.metadata: dict = {}
        self._load()

    def _load(self) -> None:
        if SCORE_MODEL_PATH.exists():
            with open(SCORE_MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
        if SCORE_METADATA_PATH.exists():
            self.metadata = json.loads(SCORE_METADATA_PATH.read_text())

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
        host: str = "none",
    ) -> dict:
        if not self.ready:
            raise RuntimeError(
                "Score model not trained. Run `python -m ml.train` first.")
        if team_a == team_b:
            raise ValueError("Team A and Team B must be different.")

        a_is_host = host == "A"
        b_is_host = host == "B"
        result = predict_score(
            self.model, team_a, team_b,
            a_is_host=a_is_host, b_is_host=b_is_host,
        )
        result["model_name"] = self.metadata.get("model_name", "Poisson + Dixon-Coles")
        return result


# Singleton used by routers.
score_service = ScoreService()

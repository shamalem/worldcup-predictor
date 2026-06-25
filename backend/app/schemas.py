"""Pydantic schemas for the API contract."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Stage = Literal["Group Stage", "Round of 16", "Quarterfinal", "Semifinal", "Final"]


class PredictRequest(BaseModel):
    team_a: str = Field(..., examples=["Brazil"])
    team_b: str = Field(..., examples=["Germany"])
    year: int = Field(..., ge=1930, le=2030, examples=[2014])
    stage: Stage = "Group Stage"
    neutral: bool = True
    host: Optional[Literal["A", "B", "none"]] = "none"


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float


class PredictResponse(BaseModel):
    team_a: str
    team_b: str
    predicted_result: str
    probabilities: dict  # {"team_a": .., "draw": .., "team_b": ..}
    confidence: str
    reasons: List[str]
    contributions: List[FeatureContribution]
    model_name: str


class TeamsResponse(BaseModel):
    teams: List[str]
    years: List[int]
    stages: List[str]


class ModelPerformanceResponse(BaseModel):
    best_model: str
    n_matches: int
    metrics: dict
    feature_importance: dict
    feature_display: dict
    class_labels: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ----------------------------- score prediction ----------------------------- #
class ScorePredictRequest(BaseModel):
    team_a: str = Field(..., examples=["Brazil"])
    team_b: str = Field(..., examples=["Germany"])
    neutral: bool = True
    host: Optional[Literal["A", "B", "none"]] = "none"


class Scoreline(BaseModel):
    team_a: int
    team_b: int
    probability: float
    label: str


class MostLikelyScore(BaseModel):
    team_a: int
    team_b: int
    probability: float
    text: str


class ScorePredictResponse(BaseModel):
    team_a: str
    team_b: str
    expected_goals: dict          # {"team_a": .., "team_b": ..}
    most_likely_score: MostLikelyScore
    top_scorelines: List[Scoreline]
    outcome_probabilities: dict   # {"team_a": .., "draw": .., "team_b": ..}
    rho: float
    model_name: str

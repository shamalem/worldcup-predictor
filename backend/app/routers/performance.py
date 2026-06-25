"""Model performance + feature importance endpoint."""
from fastapi import APIRouter, HTTPException

from ..prediction import service
from ..schemas import ModelPerformanceResponse

router = APIRouter(tags=["model"])


@router.get("/model-performance", response_model=ModelPerformanceResponse)
def model_performance():
    md = service.metadata
    if not md:
        raise HTTPException(503, "No model metadata. Train the model first.")
    return ModelPerformanceResponse(
        best_model=md.get("best_model", "unknown"),
        n_matches=md.get("n_matches", 0),
        metrics=md.get("metrics", {}),
        feature_importance=md.get("feature_importance", {}),
        feature_display=md.get("feature_display", {}),
        class_labels=md.get("class_labels", ["Team A win", "Draw", "Team B win"]),
    )

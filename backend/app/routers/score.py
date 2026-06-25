"""POST /predict-score — predict the exact scoreline distribution."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..score_service import score_service
from ..schemas import ScorePredictRequest, ScorePredictResponse

router = APIRouter(tags=["score"])


@router.post("/predict-score", response_model=ScorePredictResponse)
def predict_score(req: ScorePredictRequest):
    if not score_service.ready:
        raise HTTPException(
            503, "Score model not trained. Run `python -m ml.train`.")
    try:
        return score_service.predict(
            team_a=req.team_a,
            team_b=req.team_b,
            neutral=req.neutral,
            host=req.host or "none",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))

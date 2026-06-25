"""POST /predict — predict a World Cup match outcome and store the request."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models_db import Prediction
from ..prediction import service
from ..schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    if not service.ready:
        raise HTTPException(503, "Model not trained. Run `python -m ml.train`.")
    try:
        result = service.predict(
            team_a=req.team_a,
            team_b=req.team_b,
            year=req.year,
            stage=req.stage,
            neutral=req.neutral,
            host=req.host or "none",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    # Persist the prediction (best-effort; a DB hiccup shouldn't fail the call).
    try:
        db.add(Prediction(
            team_a=result["team_a"],
            team_b=result["team_b"],
            year=req.year,
            stage=req.stage,
            neutral=req.neutral,
            host=req.host or "none",
            predicted_result=result["predicted_result"],
            prob_a=result["probabilities"]["team_a"],
            prob_draw=result["probabilities"]["draw"],
            prob_b=result["probabilities"]["team_b"],
            confidence=result["confidence"],
            reasons=json.dumps(result["reasons"]),
        ))
        db.commit()
    except Exception:
        db.rollback()

    return result

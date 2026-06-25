"""Catalog endpoint: teams, years, and stages the UI can offer."""
from fastapi import APIRouter, HTTPException

from ..prediction import service
from ..schemas import TeamsResponse

router = APIRouter(tags=["teams"])


@router.get("/teams", response_model=TeamsResponse)
def get_teams():
    if not service.ready:
        raise HTTPException(503, "Model artifacts not loaded. Train the model first.")
    return TeamsResponse(
        teams=service.teams(),
        years=service.years(),
        stages=service.stages(),
    )

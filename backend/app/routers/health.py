"""Health-check endpoint."""
from fastapi import APIRouter

from ..config import settings
from ..prediction import service
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=service.ready,
        version=settings.api_version,
    )

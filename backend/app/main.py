"""FastAPI application entrypoint."""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import SessionLocal, init_db
from .prediction import service
from .routers import health, predict, performance, teams


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.api_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for r in (health.router, teams.router, predict.router, performance.router):
        app.include_router(r, prefix="/api")

    @app.on_event("startup")
    def _startup():
        init_db()
        _sync_model_metadata()

    @app.get("/")
    def root():
        return {"name": settings.app_name, "docs": "/docs", "health": "/api/health"}

    return app


def _sync_model_metadata():
    """Mirror the trained-model metadata into the DB for the metadata table."""
    if not service.metadata:
        return
    from .models_db import ModelMetadata
    db = SessionLocal()
    try:
        md = service.metadata
        row = db.get(ModelMetadata, 1)
        if row is None:
            row = ModelMetadata(id=1)
            db.add(row)
        row.best_model = md.get("best_model", "unknown")
        row.metrics_json = json.dumps(md.get("metrics", {}))
        row.importance_json = json.dumps(md.get("feature_importance", {}))
        row.n_matches = md.get("n_matches", 0)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


app = create_app()

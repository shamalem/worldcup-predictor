"""ORM models for teams, stored WC matches, user predictions, model metadata."""
from __future__ import annotations

from datetime import datetime, date

from sqlalchemy import (
    Boolean, Date, DateTime, Float, Integer, String, Text, func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)


class Match(Base):
    """A historical World Cup finals match (mirrors the cleaned dataset)."""
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_date: Mapped[date] = mapped_column(Date, index=True)
    year: Mapped[int] = mapped_column(Integer, index=True)
    home_team: Mapped[str] = mapped_column(String(120), index=True)
    away_team: Mapped[str] = mapped_column(String(120), index=True)
    home_score: Mapped[int] = mapped_column(Integer)
    away_score: Mapped[int] = mapped_column(Integer)
    neutral: Mapped[bool] = mapped_column(Boolean, default=False)
    country: Mapped[str] = mapped_column(String(120), nullable=True)
    stage: Mapped[str] = mapped_column(String(40), default="Group Stage")
    result: Mapped[str] = mapped_column(String(1))  # H / D / A


class Prediction(Base):
    """A prediction the user requested, stored for audit / history."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    team_a: Mapped[str] = mapped_column(String(120))
    team_b: Mapped[str] = mapped_column(String(120))
    year: Mapped[int] = mapped_column(Integer)
    stage: Mapped[str] = mapped_column(String(40))
    neutral: Mapped[bool] = mapped_column(Boolean)
    host: Mapped[str] = mapped_column(String(10), nullable=True)  # A / B / none
    predicted_result: Mapped[str] = mapped_column(String(20))
    prob_a: Mapped[float] = mapped_column(Float)
    prob_draw: Mapped[float] = mapped_column(Float)
    prob_b: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(10))
    reasons: Mapped[str] = mapped_column(Text)  # JSON-encoded list of strings


class ModelMetadata(Base):
    """Latest trained-model metadata snapshot (one row, id=1)."""
    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    best_model: Mapped[str] = mapped_column(String(60))
    metrics_json: Mapped[str] = mapped_column(Text)
    importance_json: Mapped[str] = mapped_column(Text)
    n_matches: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

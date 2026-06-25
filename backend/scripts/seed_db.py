"""Seed the DB with teams + World Cup matches from the cleaned artifacts.

Usage (from backend/):  python -m scripts.seed_db
Requires that `python -m ml.train` has already produced wc_matches.parquet.
"""
from __future__ import annotations

import pandas as pd

from app.database import SessionLocal, init_db
from app.models_db import Team, Match
from ml.config import wc_matches_available, load_wc_matches


def main():
    if not wc_matches_available():
        raise SystemExit("WC matches cache not found. Run `python -m ml.train` first.")

    init_db()
    wc = load_wc_matches()
    db = SessionLocal()
    try:
        # Reset
        db.query(Match).delete()
        db.query(Team).delete()

        teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))
        db.add_all(Team(name=str(t)) for t in teams)

        for _, m in wc.iterrows():
            db.add(Match(
                match_date=pd.to_datetime(m["date"]).date(),
                year=int(m["year"]),
                home_team=str(m["home_team"]),
                away_team=str(m["away_team"]),
                home_score=int(m["home_score"]),
                away_score=int(m["away_score"]),
                neutral=bool(m["neutral"]),
                country=str(m.get("country", "")),
                stage=str(m["stage"]),
                result=str(m["result"]),
            ))
        db.commit()
        print(f"Seeded {len(teams)} teams and {len(wc)} matches.")
    finally:
        db.close()


if __name__ == "__main__":
    main()

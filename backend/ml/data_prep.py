"""Load the raw results.csv and reduce it to clean FIFA World Cup finals matches."""
from __future__ import annotations

import pandas as pd

from .config import RESULTS_CSV, WORLD_CUP_TOURNAMENT, STAGE_ORDER


def _infer_stage(row: pd.Series) -> str:
    """The public dataset only labels the tournament ('FIFA World Cup'), not the
    round. We don't have a reliable per-match round column, so we default every
    historical match to 'Group Stage' for training. The *user* supplies the stage
    at prediction time, and stage is only a small contextual feature, so this keeps
    training honest without inventing data.

    If you later enrich results.csv with a 'stage' column, this function will use
    it automatically.
    """
    stage = row.get("stage")
    if isinstance(stage, str) and stage in STAGE_ORDER:
        return stage
    return "Group Stage"


def load_world_cup_matches(csv_path=RESULTS_CSV) -> pd.DataFrame:
    """Return a tidy dataframe of World Cup finals matches sorted by date.

    Columns: date, year, home_team, away_team, home_score, away_score,
             neutral, country, stage, result (H/D/A from home perspective).
    """
    df = pd.read_csv(csv_path)

    # Keep only the finals tournament, exclude qualification.
    df = df[df["tournament"] == WORLD_CUP_TOURNAMENT].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["year"] = df["date"].dt.year

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # neutral arrives as TRUE/FALSE text or bool depending on pandas version.
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )

    df["stage"] = df.apply(_infer_stage, axis=1)

    def result(r):
        if r["home_score"] > r["away_score"]:
            return "H"
        if r["home_score"] < r["away_score"]:
            return "A"
        return "D"

    df["result"] = df.apply(result, axis=1)

    df = df.sort_values("date").reset_index(drop=True)
    cols = [
        "date", "year", "home_team", "away_team", "home_score",
        "away_score", "neutral", "country", "stage", "result",
    ]
    return df[cols]


def list_teams(wc_df: pd.DataFrame) -> list[str]:
    teams = pd.unique(
        pd.concat([wc_df["home_team"], wc_df["away_team"]], ignore_index=True)
    )
    return sorted(map(str, teams))


def list_years(wc_df: pd.DataFrame) -> list[int]:
    return sorted(int(y) for y in wc_df["year"].unique())

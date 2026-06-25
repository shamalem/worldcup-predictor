"""Shared test fixtures: a synthetic World Cup dataset so tests run with no
network and no downloaded data."""
import pandas as pd
import pytest


@pytest.fixture
def fake_wc_df():
    rows = [
        # date, home, away, hs, as, neutral, country, stage, year
        ("1930-07-13", "Brazil", "Germany", 2, 1, True, "Uruguay", "Group Stage"),
        ("1934-05-27", "Germany", "Brazil", 1, 1, True, "Italy", "Group Stage"),
        ("1950-06-24", "Brazil", "Germany", 3, 0, False, "Brazil", "Group Stage"),
        ("1954-06-16", "Germany", "Argentina", 2, 2, True, "Switzerland", "Final"),
        ("1970-06-21", "Brazil", "Argentina", 4, 1, True, "Mexico", "Final"),
        ("1990-07-08", "Germany", "Argentina", 1, 0, True, "Italy", "Final"),
        ("2002-06-30", "Brazil", "Germany", 2, 0, True, "Japan", "Final"),
        ("2014-07-08", "Germany", "Brazil", 7, 1, False, "Brazil", "Semifinal"),
    ]
    df = pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "neutral", "country", "stage"])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["result"] = df.apply(
        lambda r: "H" if r.home_score > r.away_score
        else ("A" if r.home_score < r.away_score else "D"), axis=1)
    return df.sort_values("date").reset_index(drop=True)

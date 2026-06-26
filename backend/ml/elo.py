"""Team strength ratings (Elo) computed over the *entire* international record.

The outcome model is trained only on World Cup matches, but a team's strength is
better estimated from every international it has played (friendlies, qualifiers,
continental tournaments, World Cups). This module replays the full results.csv
chronologically and produces, for any fixture, each side's Elo rating **before**
that match (so it is leakage-free), plus the final/current rating for every team
(used at inference time).

The update follows the well-known World Football Elo scheme:

    R'   = R + K * G * (W - We)
    We   = 1 / (1 + 10^(-dr / 400))          # expected result
    dr   = R_home - R_away + H               # H = home-field bonus (0 if neutral)

* ``K`` scales with **match importance** (a World Cup game moves ratings far more
  than a friendly).
* ``G`` scales with the **margin of victory** (a 4-0 counts more than a 1-0).
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .config import RESULTS_CSV

BASE_RATING = 1500.0
HOME_ADVANTAGE = 100.0  # rating points added to a non-neutral home side


def _k_for(tournament: str) -> float:
    """Importance weight K for a match, from its tournament label."""
    t = (tournament or "").lower()
    if t == "fifa world cup":
        return 60.0
    if "world cup qualification" in t:
        return 40.0
    # Major continental finals tournaments.
    majors = (
        "uefa euro", "copa am", "african cup of nations", "africa cup of nations",
        "afc asian cup", "gold cup", "confederations cup", "nations league",
        "oceania nations cup", "concacaf",
    )
    if any(m in t for m in majors):
        return 40.0
    if "qualification" in t or "qualifier" in t:
        return 30.0
    if "friendly" in t:
        return 20.0
    return 30.0


def _g_for(goal_diff: int) -> float:
    """Margin-of-victory multiplier G."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) * 0.125


def load_all_internationals(csv_path=RESULTS_CSV) -> pd.DataFrame:
    """Load every international match (not just World Cup) for Elo."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.lower().isin(["true", "1"])
    )
    return df.sort_values("date").reset_index(drop=True)


def _match_key(date, home: str, away: str) -> Tuple[str, str, str]:
    d = pd.Timestamp(date).date().isoformat()
    return (d, str(home), str(away))


def compute_elo(all_df: pd.DataFrame):
    """Replay all internationals; return (pre_match_lookup, final_ratings).

    ``pre_match_lookup[(date, home, away)] = (home_elo_before, away_elo_before)``
    ``final_ratings[team] = current rating after the whole history``
    """
    ratings: Dict[str, float] = {}
    pre_match: Dict[Tuple[str, str, str], Tuple[float, float]] = {}

    for _, m in all_df.iterrows():
        home, away = str(m["home_team"]), str(m["away_team"])
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        neutral = bool(m["neutral"])

        r_home = ratings.get(home, BASE_RATING)
        r_away = ratings.get(away, BASE_RATING)

        # Record pre-match ratings (leakage-free snapshot).
        pre_match[_match_key(m["date"], home, away)] = (r_home, r_away)

        h = 0.0 if neutral else HOME_ADVANTAGE
        dr = r_home - r_away + h
        we_home = 1.0 / (1.0 + 10 ** (-dr / 400.0))

        if hs > as_:
            w_home = 1.0
        elif hs < as_:
            w_home = 0.0
        else:
            w_home = 0.5

        k = _k_for(str(m.get("tournament", "")))
        g = _g_for(hs - as_)
        delta = k * g * (w_home - we_home)

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    return pre_match, ratings


def elo_diff_for_match(pre_match, date, home: str, away: str,
                       final_ratings=None) -> Tuple[float, float]:
    """Return (home_elo, away_elo) before a match.

    Falls back to current/base ratings if the exact match isn't in the lookup
    (e.g. a hypothetical future fixture at inference time).
    """
    key = _match_key(date, home, away)
    if key in pre_match:
        return pre_match[key]
    if final_ratings is not None:
        return (final_ratings.get(home, BASE_RATING),
                final_ratings.get(away, BASE_RATING))
    return (BASE_RATING, BASE_RATING)

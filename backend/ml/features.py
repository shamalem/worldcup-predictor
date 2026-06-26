"""World Cup feature engineering.

Two entry points share ONE feature definition so training and serving never drift:

* ``build_training_frame``  -> chronological, leakage-free rows for model fitting.
* ``build_feature_vector``  -> a single row for a hypothetical future match,
  used by the FastAPI ``/predict`` endpoint.

A team's features are computed only from matches that happened *before* the match
being scored, so the model never peeks at the result it is predicting.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd

from .config import FEATURE_COLUMNS, STAGE_ORDER, KNOCKOUT_STAGES
from .elo import elo_diff_for_match, BASE_RATING
from .form import form_for_match, NEUTRAL_FORM

YEAR_MIN, YEAR_MAX = 1930, 2026


# --------------------------------------------------------------------------- #
# Accumulator helpers
# --------------------------------------------------------------------------- #
def _new_team_stats() -> Dict[str, float]:
    return {"mp": 0, "w": 0, "gf": 0, "ga": 0, "ko": 0}


def _team_block(stats: Dict, team: str, prefix: str) -> Dict[str, float]:
    s = stats.get(team, _new_team_stats())
    mp = s["mp"]
    win_rate = s["w"] / mp if mp else 0.0
    gf = s["gf"] / mp if mp else 0.0
    ga = s["ga"] / mp if mp else 0.0
    return {
        f"{prefix}_win_rate": round(win_rate, 4),
        f"{prefix}_goals_for": round(gf, 4),
        f"{prefix}_goals_against": round(ga, 4),
        f"{prefix}_goal_diff": round(gf - ga, 4),
        f"{prefix}_matches_played": float(mp),
        f"{prefix}_knockout_played": float(s["ko"]),
    }


def _h2h_block(h2h: Dict[Tuple[str, str], Dict], a: str, b: str) -> Dict[str, float]:
    rec = h2h.get((a, b), {"matches": 0, "a_wins": 0})
    matches = rec["matches"]
    rate = rec["a_wins"] / matches if matches else 0.5
    return {"h2h_a_win_rate": round(rate, 4), "h2h_matches": float(matches)}


def _row(stats, h2h, a, b, *, neutral, stage, year,
         a_is_host, b_is_host, elo_diff=0.0,
         a_form=NEUTRAL_FORM, b_form=NEUTRAL_FORM) -> Dict[str, float]:
    """Assemble a full feature dict for the match (team a in slot A)."""
    feats: Dict[str, float] = {}
    feats.update(_team_block(stats, a, "a"))
    feats.update(_team_block(stats, b, "b"))
    feats.update(_h2h_block(h2h, a, b))
    feats["a_is_host"] = float(bool(a_is_host))
    feats["b_is_host"] = float(bool(b_is_host))
    feats["stage_ordinal"] = float(STAGE_ORDER.get(stage, 0))
    feats["is_knockout"] = float(stage in KNOCKOUT_STAGES)
    feats["neutral"] = float(bool(neutral))
    feats["year_norm"] = round((year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN), 4)
    feats["elo_diff"] = round(float(elo_diff), 2)
    feats["elo_expectation"] = round(1.0 / (1.0 + 10 ** (-float(elo_diff) / 400.0)), 4)
    feats["a_recent_form"] = round(float(a_form), 4)
    feats["b_recent_form"] = round(float(b_form), 4)
    # Guarantee column order / completeness.
    return {c: feats[c] for c in FEATURE_COLUMNS}


def _update(stats, h2h, home, away, hs, as_, stage):
    for t in (home, away):
        stats.setdefault(t, _new_team_stats())
    stats[home]["mp"] += 1
    stats[away]["mp"] += 1
    stats[home]["gf"] += hs
    stats[home]["ga"] += as_
    stats[away]["gf"] += as_
    stats[away]["ga"] += hs
    if hs > as_:
        stats[home]["w"] += 1
    elif as_ > hs:
        stats[away]["w"] += 1
    if stage in KNOCKOUT_STAGES:
        stats[home]["ko"] += 1
        stats[away]["ko"] += 1

    for (x, y) in ((home, away), (away, home)):
        rec = h2h.setdefault((x, y), {"matches": 0, "a_wins": 0})
        rec["matches"] += 1
    if hs > as_:
        h2h[(home, away)]["a_wins"] += 1
    elif as_ > hs:
        h2h[(away, home)]["a_wins"] += 1


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_training_frame(wc_df: pd.DataFrame, pre_match=None,
                         final_ratings=None, pre_form=None,
                         final_form=None) -> pd.DataFrame:
    """Return X+y rows. Each match produces TWO rows (original + mirrored A/B)
    so the model is symmetric and can't learn a "slot A is special" bias.

    ``pre_match``/``final_ratings`` carry leakage-free Elo; ``pre_form``/
    ``final_form`` carry leakage-free recent form. All come from the ml.elo /
    ml.form helpers.
    """
    stats: Dict[str, Dict] = defaultdict(_new_team_stats)
    h2h: Dict[Tuple[str, str], Dict] = {}
    rows = []

    for _, m in wc_df.iterrows():
        home, away = str(m["home_team"]), str(m["away_team"])
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        neutral, stage, year = bool(m["neutral"]), m["stage"], int(m["year"])
        host = not neutral  # non-neutral => home_team had host/home advantage

        if pre_match is not None:
            home_elo, away_elo = elo_diff_for_match(
                pre_match, m["date"], home, away, final_ratings)
        else:
            home_elo = away_elo = BASE_RATING

        if pre_form is not None:
            home_form, away_form = form_for_match(
                pre_form, m["date"], home, away, final_form)
        else:
            home_form = away_form = NEUTRAL_FORM

        # Orientation 1: A = home
        r1 = _row(stats, h2h, home, away, neutral=neutral, stage=stage,
                  year=year, a_is_host=host, b_is_host=False,
                  elo_diff=home_elo - away_elo,
                  a_form=home_form, b_form=away_form)
        r1["target"] = {"H": 0, "D": 1, "A": 2}[m["result"]]
        r1["year"] = year
        rows.append(r1)

        # Orientation 2 (mirror): A = away
        r2 = _row(stats, h2h, away, home, neutral=neutral, stage=stage,
                  year=year, a_is_host=False, b_is_host=host,
                  elo_diff=away_elo - home_elo,
                  a_form=away_form, b_form=home_form)
        r2["target"] = {"H": 2, "D": 1, "A": 0}[m["result"]]
        r2["year"] = year
        rows.append(r2)

        _update(stats, h2h, home, away, hs, as_, stage)

    return pd.DataFrame(rows)


def build_accumulators(wc_df: pd.DataFrame, as_of_year: int):
    """Replay history strictly *before* ``as_of_year`` and return (stats, h2h)."""
    stats: Dict[str, Dict] = defaultdict(_new_team_stats)
    h2h: Dict[Tuple[str, str], Dict] = {}
    past = wc_df[wc_df["year"] < as_of_year]
    for _, m in past.iterrows():
        _update(stats, h2h, str(m["home_team"]), str(m["away_team"]),
                int(m["home_score"]), int(m["away_score"]), m["stage"])
    return stats, h2h


def build_feature_vector(
    wc_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    year: int,
    stage: str,
    neutral: bool,
    a_is_host: bool = False,
    b_is_host: bool = False,
    final_ratings=None,
    final_form=None,
) -> Dict[str, float]:
    """One feature row for a hypothetical match (used at inference time)."""
    stats, h2h = build_accumulators(wc_df, year)
    if final_ratings is not None:
        elo_diff = (final_ratings.get(team_a, BASE_RATING)
                    - final_ratings.get(team_b, BASE_RATING))
    else:
        elo_diff = 0.0
    if final_form is not None:
        a_form = final_form.get(team_a, NEUTRAL_FORM)
        b_form = final_form.get(team_b, NEUTRAL_FORM)
    else:
        a_form = b_form = NEUTRAL_FORM
    return _row(
        stats, h2h, team_a, team_b,
        neutral=neutral, stage=stage, year=year,
        a_is_host=a_is_host, b_is_host=b_is_host, elo_diff=elo_diff,
        a_form=a_form, b_form=b_form,
    )

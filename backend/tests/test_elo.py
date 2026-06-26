"""Tests for the importance-weighted Elo rating module."""
import pandas as pd

from ml.elo import compute_elo, _k_for, _g_for, BASE_RATING, elo_diff_for_match


def _mini_history():
    """A tiny international history: a strong team that keeps winning."""
    rows = [
        ("2018-01-01", "Strongland", "Weakland", 3, 0, True, "Friendly"),
        ("2018-06-01", "Strongland", "Weakland", 2, 0, True, "FIFA World Cup"),
        ("2019-01-01", "Midland", "Weakland", 1, 0, True, "Friendly"),
        ("2019-06-01", "Strongland", "Midland", 2, 1, True, "FIFA World Cup"),
    ]
    df = pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "neutral", "tournament"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_world_cup_weighs_more_than_friendly():
    assert _k_for("FIFA World Cup") > _k_for("Friendly")
    assert _k_for("FIFA World Cup qualification") > _k_for("Friendly")


def test_bigger_win_moves_rating_more():
    assert _g_for(4) > _g_for(2) > _g_for(1)


def test_winner_gains_loser_loses():
    df = _mini_history()
    pre, final = compute_elo(df)
    # Strongland won everything -> rating above base; Weakland below.
    assert final["Strongland"] > BASE_RATING
    assert final["Weakland"] < BASE_RATING
    assert final["Strongland"] > final["Weakland"]


def test_pre_match_is_leakage_free():
    df = _mini_history()
    pre, final = compute_elo(df)
    # The very first match: both teams start at base (no prior history).
    home, away = elo_diff_for_match(pre, "2018-01-01", "Strongland", "Weakland")
    assert home == BASE_RATING and away == BASE_RATING


def test_unknown_team_falls_back_to_base():
    df = _mini_history()
    pre, final = compute_elo(df)
    home, away = elo_diff_for_match(
        pre, "2030-01-01", "Nowhereland", "Weakland", final_ratings=final)
    assert home == BASE_RATING  # unknown team
    assert away == final["Weakland"]  # known team uses current rating

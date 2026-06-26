"""Tests for the recent-form module."""
import pandas as pd

from ml.form import compute_form, form_for_match, NEUTRAL_FORM


def _history():
    rows = [
        ("2020-01-01", "Hotstreak", "Coldstreak", 2, 0, True, "Friendly"),
        ("2020-02-01", "Hotstreak", "Midteam", 1, 0, True, "Friendly"),
        ("2020-03-01", "Coldstreak", "Midteam", 0, 3, True, "Friendly"),
        ("2020-04-01", "Hotstreak", "Coldstreak", 3, 1, True, "Friendly"),
    ]
    df = pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "neutral", "tournament"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_winning_team_has_higher_form():
    df = _history()
    _, final = compute_form(df)
    assert final["Hotstreak"] > final["Coldstreak"]
    # Form is expressed in [0, 1].
    for v in final.values():
        assert 0.0 <= v <= 1.0


def test_first_match_is_neutral_form():
    df = _history()
    pre, _ = compute_form(df)
    h, a = form_for_match(pre, "2020-01-01", "Hotstreak", "Coldstreak")
    assert h == NEUTRAL_FORM and a == NEUTRAL_FORM


def test_unknown_team_falls_back():
    df = _history()
    pre, final = compute_form(df)
    h, a = form_for_match(
        pre, "2030-01-01", "Nobody", "Hotstreak", final_form=final)
    assert h == NEUTRAL_FORM
    assert a == final["Hotstreak"]

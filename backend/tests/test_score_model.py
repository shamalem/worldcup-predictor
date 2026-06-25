"""Tests for the Poisson + Dixon-Coles scoreline model."""
import numpy as np

from ml.score_model import (
    fit_score_model, predict_score, scoreline_grid, outcome_from_grid,
)


def test_scoreline_grid_sums_to_one():
    grid = scoreline_grid(1.5, 1.2, rho=-0.05)
    assert abs(grid.sum() - 1.0) < 1e-9
    # Outcome split also sums to 1.
    p_a, p_draw, p_b = outcome_from_grid(grid)
    assert abs((p_a + p_draw + p_b) - 1.0) < 1e-9


def test_stronger_attack_raises_win_prob(fake_wc_df):
    model = fit_score_model(fake_wc_df)
    # Manufacture a clearly stronger team by boosting its attack rating.
    teams = model.teams
    strong, weak = teams[0], teams[1]
    model.attack[strong] = 2.0
    model.defence[strong] = 0.5
    model.attack[weak] = 0.6
    model.defence[weak] = 1.6

    pred = predict_score(model, strong, weak)
    op = pred["outcome_probabilities"]
    assert op["team_a"] > op["team_b"]
    # Expected goals for the strong side should exceed the weak side.
    assert pred["expected_goals"]["team_a"] > pred["expected_goals"]["team_b"]


def test_prediction_shape(fake_wc_df):
    model = fit_score_model(fake_wc_df)
    pred = predict_score(model, model.teams[0], model.teams[1])
    assert "most_likely_score" in pred
    assert "top_scorelines" in pred
    assert len(pred["top_scorelines"]) >= 1
    ms = pred["most_likely_score"]
    assert ms["team_a"] >= 0 and ms["team_b"] >= 0
    # Probabilities are valid.
    for s in pred["top_scorelines"]:
        assert 0.0 <= s["probability"] <= 1.0


def test_symmetry_when_swapping_teams(fake_wc_df):
    """Swapping A and B should mirror the outcome probabilities."""
    model = fit_score_model(fake_wc_df)
    a, b = model.teams[0], model.teams[1]
    p1 = predict_score(model, a, b)["outcome_probabilities"]
    p2 = predict_score(model, b, a)["outcome_probabilities"]
    assert abs(p1["team_a"] - p2["team_b"]) < 1e-6
    assert abs(p1["draw"] - p2["draw"]) < 1e-6

"""Exact-scoreline predictor for World Cup matches.

A Poisson goals model with team **attack** and **defence** strengths and a
**Dixon-Coles** low-score correction. It estimates how many goals each team is
expected to score, turns that into a probability for *every* scoreline, and from
that grid derives both the most likely exact score and a win/draw/loss split.

Design choices that keep this dependency-light (numpy only, no scipy):

* Attack/defence strengths are estimated empirically (goals scored / conceded
  relative to the tournament average) with **shrinkage** toward the average so
  teams with only a handful of World Cup matches don't get extreme ratings.
* The Dixon-Coles ``rho`` (the nudge applied to 0-0, 1-0, 0-1, 1-1) is fitted by
  a small 1-D search that maximises log-likelihood on the training scorelines.

Exact-score prediction is genuinely hard — football is low-scoring and noisy, so
even a good model lands the exact score only ~15-25% of the time. The scoreline
*probabilities* are still meaningful (1-0 and 2-1 rank far above 5-4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Prior strength (in "pseudo-matches" at league average) used to shrink the
# attack/defence ratings of teams with few World Cup matches toward 1.0.
SHRINKAGE_PRIOR = 5.0
# Largest goal count considered per team when building the scoreline grid.
MAX_GOALS = 10
# Candidate Dixon-Coles rho values searched during fitting.
_RHO_GRID = np.round(np.arange(-0.20, 0.001, 0.01), 3)


# --------------------------------------------------------------------------- #
# Poisson + Dixon-Coles primitives
# --------------------------------------------------------------------------- #
def _poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam). Implemented without scipy."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles dependence correction for the four low-score cells."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def scoreline_grid(lam_a: float, lam_b: float, rho: float,
                   max_goals: int = MAX_GOALS) -> np.ndarray:
    """Return a (max_goals+1) x (max_goals+1) matrix of scoreline probabilities.

    Entry [x, y] is P(team A scores x, team B scores y). The grid is
    Dixon-Coles corrected and renormalised to sum to 1.
    """
    a_pmf = np.array([_poisson_pmf(i, lam_a) for i in range(max_goals + 1)])
    b_pmf = np.array([_poisson_pmf(j, lam_b) for j in range(max_goals + 1)])
    grid = np.outer(a_pmf, b_pmf)

    # Apply the low-score correction to the four affected cells.
    for x in (0, 1):
        for y in (0, 1):
            grid[x, y] *= _dc_tau(x, y, lam_a, lam_b, rho)

    total = grid.sum()
    if total > 0:
        grid /= total
    return grid


def outcome_from_grid(grid: np.ndarray) -> tuple[float, float, float]:
    """Collapse a scoreline grid into (P(A win), P(draw), P(B win))."""
    p_a = float(np.tril(grid, -1).sum())   # x > y
    p_draw = float(np.trace(grid))         # x == y
    p_b = float(np.triu(grid, 1).sum())    # x < y
    return p_a, p_draw, p_b


# --------------------------------------------------------------------------- #
# Fitting team strengths
# --------------------------------------------------------------------------- #
@dataclass
class ScoreModel:
    """A fitted score model. Pickled and served by the API."""
    league_avg: float
    host_mult: float
    rho: float
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)
    teams: list = field(default_factory=list)
    n_matches: int = 0


def _accumulate_strengths(df: pd.DataFrame, league_avg: float):
    """Empirical, shrunk attack/defence ratings from a set of matches.

    attack[t]  > 1  -> scores more than an average side
    defence[t] > 1  -> concedes more than average (i.e. a *weaker* defence)
    """
    scored: dict[str, float] = {}
    conceded: dict[str, float] = {}
    played: dict[str, int] = {}

    def add(team, gf, ga):
        scored[team] = scored.get(team, 0.0) + gf
        conceded[team] = conceded.get(team, 0.0) + ga
        played[team] = played.get(team, 0) + 1

    for _, r in df.iterrows():
        add(r["home_team"], r["home_score"], r["away_score"])
        add(r["away_team"], r["away_score"], r["home_score"])

    attack, defence = {}, {}
    for team, n in played.items():
        # Shrink the per-team rate toward the league average using a prior of
        # SHRINKAGE_PRIOR pseudo-matches, then express as a ratio of league avg.
        atk_rate = (scored[team] + SHRINKAGE_PRIOR * league_avg) / (n + SHRINKAGE_PRIOR)
        def_rate = (conceded[team] + SHRINKAGE_PRIOR * league_avg) / (n + SHRINKAGE_PRIOR)
        attack[team] = atk_rate / league_avg
        defence[team] = def_rate / league_avg
    return attack, defence


def _fit_rho(df: pd.DataFrame, model: "ScoreModel") -> float:
    """Pick the rho in _RHO_GRID that maximises training log-likelihood."""
    best_rho, best_ll = 0.0, -np.inf
    # Precompute expected goals per match once (rho-independent).
    lams = []
    for _, r in df.iterrows():
        la, lb = expected_goals(model, r["home_team"], r["away_team"])
        lams.append((int(r["home_score"]), int(r["away_score"]), la, lb))

    for rho in _RHO_GRID:
        ll = 0.0
        for x, y, la, lb in lams:
            p = (_poisson_pmf(x, la) * _poisson_pmf(y, lb)
                 * _dc_tau(x, y, la, lb, float(rho)))
            ll += math.log(max(p, 1e-12))
        if ll > best_ll:
            best_ll, best_rho = ll, float(rho)
    return best_rho


def expected_goals(model: "ScoreModel", team_a: str, team_b: str,
                   a_is_host: bool = False, b_is_host: bool = False
                   ) -> tuple[float, float]:
    """Expected goals (lambda_A, lambda_B) for a fixture.

    Unknown teams fall back to league-average strength (ratio 1.0).
    """
    atk_a = model.attack.get(team_a, 1.0)
    def_a = model.defence.get(team_a, 1.0)
    atk_b = model.attack.get(team_b, 1.0)
    def_b = model.defence.get(team_b, 1.0)

    lam_a = model.league_avg * atk_a * def_b
    lam_b = model.league_avg * atk_b * def_a
    if a_is_host:
        lam_a *= model.host_mult
    if b_is_host:
        lam_b *= model.host_mult
    # Guard against pathological values.
    return float(np.clip(lam_a, 0.05, 8.0)), float(np.clip(lam_b, 0.05, 8.0))


def fit_score_model(df: pd.DataFrame) -> ScoreModel:
    """Fit the full score model on a set of World Cup matches."""
    total_goals = float(df["home_score"].sum() + df["away_score"].sum())
    league_avg = total_goals / (2 * len(df)) if len(df) else 1.3

    # Host advantage: how much more do non-neutral home sides score vs the away
    # side in the same matches? Clipped to a sensible range; default if sparse.
    non_neutral = df[~df["neutral"].astype(bool)]
    if len(non_neutral) >= 20:
        h = non_neutral["home_score"].mean()
        a = max(non_neutral["away_score"].mean(), 1e-6)
        host_mult = float(np.clip(h / a, 1.0, 1.8))
    else:
        host_mult = 1.25

    attack, defence = _accumulate_strengths(df, league_avg)
    model = ScoreModel(
        league_avg=league_avg,
        host_mult=host_mult,
        rho=0.0,
        attack=attack,
        defence=defence,
        teams=sorted(attack.keys()),
        n_matches=len(df),
    )
    model.rho = _fit_rho(df, model)
    return model


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_score(model: "ScoreModel", team_a: str, team_b: str,
                  a_is_host: bool = False, b_is_host: bool = False,
                  top_n: int = 5) -> dict:
    """Predict the scoreline distribution for a fixture.

    Returns the most likely exact score, the top-N scorelines with their
    probabilities, expected goals for each side, and a win/draw/loss split
    derived from the full grid.
    """
    lam_a, lam_b = expected_goals(model, team_a, team_b, a_is_host, b_is_host)
    grid = scoreline_grid(lam_a, lam_b, model.rho)

    # Rank all scorelines by probability.
    flat = [
        (i, j, float(grid[i, j]))
        for i in range(grid.shape[0]) for j in range(grid.shape[1])
    ]
    flat.sort(key=lambda t: t[2], reverse=True)
    top = flat[:top_n]
    best_i, best_j, best_p = top[0]

    p_a, p_draw, p_b = outcome_from_grid(grid)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "expected_goals": {
            "team_a": round(lam_a, 2),
            "team_b": round(lam_b, 2),
        },
        "most_likely_score": {
            "team_a": best_i,
            "team_b": best_j,
            "probability": round(best_p, 4),
            "text": f"{team_a} {best_i}-{best_j} {team_b}",
        },
        "top_scorelines": [
            {
                "team_a": i,
                "team_b": j,
                "probability": round(p, 4),
                "label": f"{i}-{j}",
            }
            for (i, j, p) in top
        ],
        "outcome_probabilities": {
            "team_a": round(p_a, 4),
            "draw": round(p_draw, 4),
            "team_b": round(p_b, 4),
        },
        "rho": round(model.rho, 3),
    }


# --------------------------------------------------------------------------- #
# Evaluation (used by the trainer to report metrics)
# --------------------------------------------------------------------------- #
def evaluate(model: "ScoreModel", df: pd.DataFrame) -> dict:
    """Score the model on held-out matches.

    Reports exact-score accuracy, outcome (W/D/L) accuracy, and mean absolute
    error on total goals.
    """
    if len(df) == 0:
        return {"n": 0}

    exact_hits = 0
    outcome_hits = 0
    goal_abs_err = 0.0
    for _, r in df.iterrows():
        pred = predict_score(model, r["home_team"], r["away_team"])
        ms = pred["most_likely_score"]
        if ms["team_a"] == int(r["home_score"]) and ms["team_b"] == int(r["away_score"]):
            exact_hits += 1

        # Actual outcome
        if r["home_score"] > r["away_score"]:
            actual = "A"
        elif r["home_score"] < r["away_score"]:
            actual = "B"
        else:
            actual = "D"
        op = pred["outcome_probabilities"]
        pred_outcome = max(
            (("A", op["team_a"]), ("D", op["draw"]), ("B", op["team_b"])),
            key=lambda t: t[1],
        )[0]
        if pred_outcome == actual:
            outcome_hits += 1

        eg = pred["expected_goals"]
        goal_abs_err += abs(
            (eg["team_a"] + eg["team_b"]) - (r["home_score"] + r["away_score"])
        )

    n = len(df)
    return {
        "n": n,
        "exact_score_accuracy": round(exact_hits / n, 4),
        "outcome_accuracy": round(outcome_hits / n, 4),
        "total_goals_mae": round(goal_abs_err / n, 4),
    }

"""Turn a single prediction into model-based explanations.

We compute per-feature contributions with SHAP when available, otherwise fall
back to a (importance x signed-deviation) proxy. Contributions are then mapped
to readable, template-based sentences -- NO LLM is used anywhere.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import CLASS_LABELS, FEATURE_COLUMNS, FEATURE_DISPLAY

# Feature -> (sentence when it favours team A, sentence when it favours team B).
# {A} and {B} are filled with the team names.
_TEMPLATES = {
    "a_win_rate":        ("{A} has a stronger historical World Cup win rate.",
                          "{B} has a stronger historical World Cup win rate."),
    "b_win_rate":        ("{B} has a weaker historical World Cup win rate.",
                          "{B} has a stronger historical World Cup win rate."),
    "a_goal_diff":       ("{A} has a better average goal difference in WC matches.",
                          "{A} has a poorer average goal difference in WC matches."),
    "b_goal_diff":       ("{B} has a poorer average goal difference in WC matches.",
                          "{B} has a better average goal difference in WC matches."),
    "a_goals_for":       ("{A} scores more goals per World Cup match.",
                          "{A} scores fewer goals per World Cup match."),
    "b_goals_for":       ("{B} scores fewer goals per World Cup match.",
                          "{B} scores more goals per World Cup match."),
    "a_goals_against":   ("{A} concedes fewer goals per World Cup match.",
                          "{A} concedes more goals per World Cup match."),
    "b_goals_against":   ("{B} concedes more goals per World Cup match.",
                          "{B} concedes fewer goals per World Cup match."),
    "a_matches_played":  ("{A} is more experienced at the World Cup.",
                          "{A} is less experienced at the World Cup."),
    "b_matches_played":  ("{B} is less experienced at the World Cup.",
                          "{B} is more experienced at the World Cup."),
    "a_knockout_played": ("{A} has more knockout-stage experience.",
                          "{A} has less knockout-stage experience."),
    "b_knockout_played": ("{B} has weaker performance in similar knockout-stage matches.",
                          "{B} has more knockout-stage experience."),
    "h2h_a_win_rate":    ("{A} has the upper hand in head-to-head World Cup meetings.",
                          "{B} has the upper hand in head-to-head World Cup meetings."),
    "h2h_matches":       ("{A} and {B} have a meaningful head-to-head history.",
                          "{A} and {B} have a meaningful head-to-head history."),
    "elo_diff":          ("{A} is the stronger side by overall Elo rating, across all internationals.",
                          "{B} is the stronger side by overall Elo rating, across all internationals."),
    "elo_expectation":   ("{A}'s Elo rating implies a strong win probability.",
                          "{B}'s Elo rating implies a strong win probability."),
    "a_recent_form":     ("{A} comes in with stronger recent form.",
                          "{A} comes in with weaker recent form."),
    "b_recent_form":     ("{B} comes in with weaker recent form.",
                          "{B} comes in with stronger recent form."),
    "a_is_host":         ("{A} benefits from host-nation advantage.",
                          "{A} does not have host-nation advantage."),
    "b_is_host":         ("{B} does not have host-nation advantage.",
                          "{B} benefits from host-nation advantage."),
    "stage_ordinal":     ("The match stage favours {A}.",
                          "The match stage favours {B}."),
    "is_knockout":       ("Knockout-match dynamics favour {A}.",
                          "Knockout-match dynamics favour {B}."),
    "neutral":           ("The neutral venue favours {A}.",
                          "The neutral venue favours {B}."),
    "year_norm":         ("Era-related trends favour {A}.",
                          "Era-related trends favour {B}."),
}


def _shap_contributions(bundle, x_scaled: np.ndarray, class_idx: int):
    """Return per-feature contributions toward ``class_idx`` for one instance.

    For linear models we use the closed-form SHAP value (exact for a linear
    model with the training mean as baseline): ``coef[class] * x_scaled``. The
    features are already standardized, so the baseline is the zero vector and
    this is both exact and robust. For tree models we use shap.TreeExplainer
    with the model's own expected-value baseline. A signed-importance proxy is
    the last-resort fallback so we never return all-zeros.
    """
    model = bundle["model"]

    # --- Linear models: exact closed-form SHAP (no degenerate background). ---
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:  # binary -> single row; class 1 is positive
            row = coef if class_idx == 1 else -coef
        else:
            row = coef[class_idx]
        return row * np.asarray(x_scaled[0], dtype=float)

    # --- Tree models: TreeExplainer with its own baseline. ---
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x_scaled)
        if isinstance(sv, list):                 # list per class
            vals = np.asarray(sv[class_idx])[0]
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3:                    # (n, features, classes)
                vals = arr[0, :, class_idx]
            else:                                # (n, features)
                vals = arr[0]
        vals = np.asarray(vals, dtype=float)
        if np.any(np.abs(vals) > 1e-12):
            return vals
        raise RuntimeError("degenerate shap values")
    except Exception:
        # Fallback: importance-weighted signed deviation from the mean.
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        else:
            imp = np.ones(len(FEATURE_COLUMNS))
        return imp * np.asarray(x_scaled[0], dtype=float)


def _favours_a(feature: str, contrib: float, predicted_class: int) -> bool:
    """Does this (signed) contribution push toward Team A winning?"""
    # Contribution is toward the predicted class. Convert to "toward A win".
    if predicted_class == 0:        # Team A win
        toward_a = contrib
    elif predicted_class == 2:      # Team B win
        toward_a = -contrib
    else:                           # Draw -> use raw sign, weak signal
        toward_a = contrib
    return toward_a >= 0


def explain(
    bundle,
    feature_row: Dict[str, float],
    x_scaled: np.ndarray,
    probabilities: List[float],
    team_a: str,
    team_b: str,
    top_k: int = 4,
):
    """Return (reasons, contributions) for the predicted outcome."""
    predicted_class = int(np.argmax(probabilities))
    shap_vals = _shap_contributions(bundle, x_scaled, predicted_class)

    contributions = [
        {
            "feature": f,
            "label": FEATURE_DISPLAY.get(f, f),
            "value": round(float(feature_row[f]), 3),
            "contribution": round(float(shap_vals[i]), 4),
        }
        for i, f in enumerate(FEATURE_COLUMNS)
    ]
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    reasons: List[str] = []
    for c in contributions[:top_k]:
        favours_a = _favours_a(c["feature"], c["contribution"], predicted_class)
        tmpl_a, tmpl_b = _TEMPLATES.get(
            c["feature"], ("{A} is favoured by " + c["label"] + ".",
                           "{B} is favoured by " + c["label"] + "."))
        sentence = (tmpl_a if favours_a else tmpl_b).format(A=team_a, B=team_b)
        reasons.append(sentence)

    return reasons, contributions


def confidence_level(probabilities: List[float]) -> str:
    """Map the winning probability to a coarse confidence band."""
    p = max(probabilities)
    if p >= 0.55:
        return "High"
    if p >= 0.42:
        return "Medium"
    return "Low"

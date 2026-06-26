"""Recent-form features computed over the full international record.

Elo captures long-run strength; *form* captures short-term momentum — how a team
has done in its last handful of matches going into the fixture. Like Elo, this is
computed over **all** internationals (not just World Cup games) and snapshotted
**before** each match, so it stays leakage-free.

For each team we track a rolling window of their most recent ``FORM_WINDOW``
results and express form as normalised points-per-game in ``[0, 1]``:

    points = 3 win / 1 draw / 0 loss
    form   = mean(points over last N) / 3      (0.5 used when no history yet)
"""
from __future__ import annotations

from collections import deque, defaultdict
from typing import Deque, Dict, Tuple

import pandas as pd

from .config import RESULTS_CSV
from .elo import load_all_internationals

FORM_WINDOW = 10          # number of recent internationals that define "form"
NEUTRAL_FORM = 0.5        # default for a team with no prior matches


def _form_value(window: Deque[int]) -> float:
    if not window:
        return NEUTRAL_FORM
    return round(sum(window) / (3.0 * len(window)), 4)


def compute_form(all_df: pd.DataFrame):
    """Replay all internationals; return (pre_match_form, final_form).

    ``pre_match_form[(date, home, away)] = (home_form, away_form)``
    ``final_form[team] = current form`` (most recent window, for inference)
    """
    windows: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    pre_match: Dict[Tuple[str, str, str], Tuple[float, float]] = {}

    for _, m in all_df.iterrows():
        home, away = str(m["home_team"]), str(m["away_team"])
        hs, as_ = int(m["home_score"]), int(m["away_score"])

        d = pd.Timestamp(m["date"]).date().isoformat()
        pre_match[(d, home, away)] = (
            _form_value(windows[home]), _form_value(windows[away]))

        if hs > as_:
            windows[home].append(3); windows[away].append(0)
        elif hs < as_:
            windows[home].append(0); windows[away].append(3)
        else:
            windows[home].append(1); windows[away].append(1)

    final_form = {t: _form_value(w) for t, w in windows.items()}
    return pre_match, final_form


def form_for_match(pre_match, date, home: str, away: str,
                   final_form=None) -> Tuple[float, float]:
    """Return (home_form, away_form) before a match, with sensible fallbacks."""
    d = pd.Timestamp(date).date().isoformat()
    key = (d, str(home), str(away))
    if key in pre_match:
        return pre_match[key]
    if final_form is not None:
        return (final_form.get(home, NEUTRAL_FORM),
                final_form.get(away, NEUTRAL_FORM))
    return (NEUTRAL_FORM, NEUTRAL_FORM)


def load_and_compute_form(csv_path=RESULTS_CSV):
    """Convenience: load all internationals and compute form in one call."""
    return compute_form(load_all_internationals(csv_path))

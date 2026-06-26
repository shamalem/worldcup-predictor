"""Shared constants for the World Cup ML pipeline."""
from pathlib import Path

# Repo paths (this file lives in backend/ml/)
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
DATA_DIR = REPO_DIR / "data"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

RESULTS_CSV = DATA_DIR / "results.csv"

# Saved training artifacts
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
WC_MATCHES_PATH = ARTIFACTS_DIR / "wc_matches.parquet"
# CSV fallback used automatically when pyarrow/fastparquet is unavailable.
WC_MATCHES_CSV_PATH = ARTIFACTS_DIR / "wc_matches.csv"

# Exact-scoreline (Poisson + Dixon-Coles) model artifacts.
SCORE_MODEL_PATH = ARTIFACTS_DIR / "score_model.pkl"
SCORE_METADATA_PATH = ARTIFACTS_DIR / "score_metadata.json"

# Elo team-strength ratings (computed over ALL internationals).
ELO_RATINGS_PATH = ARTIFACTS_DIR / "elo_ratings.json"


def save_wc_matches(df):
    """Persist the cleaned WC matches frame.

    Prefers Parquet (compact, typed) but transparently falls back to CSV when
    no parquet engine (pyarrow/fastparquet) is installed, so training never
    fails just because of an optional dependency.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(WC_MATCHES_PATH, index=False)
        # Remove a stale CSV so readers don't pick an older copy.
        if WC_MATCHES_CSV_PATH.exists():
            WC_MATCHES_CSV_PATH.unlink()
        return WC_MATCHES_PATH
    except (ImportError, ValueError):
        df.to_csv(WC_MATCHES_CSV_PATH, index=False)
        return WC_MATCHES_CSV_PATH


def wc_matches_available() -> bool:
    """True if either the parquet or CSV cache exists."""
    return WC_MATCHES_PATH.exists() or WC_MATCHES_CSV_PATH.exists()


def load_wc_matches():
    """Load the cleaned WC matches frame from whichever cache exists.

    Returns a DataFrame with the ``date`` column parsed back to datetime.
    Raises FileNotFoundError if neither cache is present.
    """
    import pandas as pd

    if WC_MATCHES_PATH.exists():
        df = pd.read_parquet(WC_MATCHES_PATH)
    elif WC_MATCHES_CSV_PATH.exists():
        df = pd.read_csv(WC_MATCHES_CSV_PATH)
    else:
        raise FileNotFoundError(
            "No WC matches cache found. Run `python -m ml.train` first."
        )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# Only the World Cup finals tournament (NOT qualification).
WORLD_CUP_TOURNAMENT = "FIFA World Cup"

# Match stages the user can pick. Maps a label -> ordinal "lateness" used as a
# feature (higher = later/more important stage).
STAGE_ORDER = {
    "Group Stage": 0,
    "Round of 16": 1,
    "Quarterfinal": 2,
    "Semifinal": 3,
    "Final": 4,
}
KNOCKOUT_STAGES = {"Round of 16", "Quarterfinal", "Semifinal", "Final"}

# Target classes (from the perspective of team A / home slot)
CLASS_LABELS = ["Team A win", "Draw", "Team B win"]

# The ordered list of model feature columns. Keep this in ONE place so training
# and inference build identical vectors.
FEATURE_COLUMNS = [
    "a_win_rate",
    "b_win_rate",
    "a_goals_for",
    "b_goals_for",
    "a_goals_against",
    "b_goals_against",
    "a_goal_diff",
    "b_goal_diff",
    "a_matches_played",
    "b_matches_played",
    "a_knockout_played",
    "b_knockout_played",
    "h2h_a_win_rate",
    "h2h_matches",
    "a_is_host",
    "b_is_host",
    "stage_ordinal",
    "is_knockout",
    "neutral",
    "year_norm",
    "elo_diff",
]

# Human-readable labels for explanations / charts.
FEATURE_DISPLAY = {
    "a_win_rate": "Team A historical WC win rate",
    "b_win_rate": "Team B historical WC win rate",
    "a_goals_for": "Team A goals scored per WC match",
    "b_goals_for": "Team B goals scored per WC match",
    "a_goals_against": "Team A goals conceded per WC match",
    "b_goals_against": "Team B goals conceded per WC match",
    "a_goal_diff": "Team A average WC goal difference",
    "b_goal_diff": "Team B average WC goal difference",
    "a_matches_played": "Team A WC match experience",
    "b_matches_played": "Team B WC match experience",
    "a_knockout_played": "Team A knockout-stage experience",
    "b_knockout_played": "Team B knockout-stage experience",
    "h2h_a_win_rate": "Head-to-head WC record (A vs B)",
    "h2h_matches": "Head-to-head WC meetings",
    "a_is_host": "Team A host-nation advantage",
    "b_is_host": "Team B host-nation advantage",
    "stage_ordinal": "Match stage importance",
    "is_knockout": "Knockout match",
    "neutral": "Neutral venue",
    "year_norm": "Tournament era",
    "elo_diff": "Team strength gap (all-matches Elo)",
}

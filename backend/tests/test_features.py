"""Feature engineering correctness + leakage checks."""
from ml.config import FEATURE_COLUMNS
from ml.features import build_training_frame, build_feature_vector


def test_training_frame_shape(fake_wc_df):
    frame = build_training_frame(fake_wc_df)
    # Two rows (orientations) per match.
    assert len(frame) == 2 * len(fake_wc_df)
    for col in FEATURE_COLUMNS:
        assert col in frame.columns
    assert set(frame["target"].unique()).issubset({0, 1, 2})


def test_no_leakage_first_match_is_blank(fake_wc_df):
    """The earliest match has no prior history, so experience must be zero."""
    frame = build_training_frame(fake_wc_df)
    first = frame.iloc[0]
    assert first["a_matches_played"] == 0
    assert first["b_matches_played"] == 0


def test_feature_vector_has_all_columns(fake_wc_df):
    row = build_feature_vector(
        fake_wc_df, "Brazil", "Germany", 2014, "Final", neutral=True)
    assert set(row.keys()) == set(FEATURE_COLUMNS)
    # Brazil has played WC matches before 2014 -> experience > 0
    assert row["a_matches_played"] > 0

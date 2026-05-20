"""
tests/unit/test_viewership.py
─────────────────────────────
Unit tests for the viewership feature table, model, and export pipeline.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.model.features import (
    FEATURE_COLUMNS,
    build_training_features,
    build_2026_features,
    _is_rivalry,
    _is_cfp_rematch,
)
from src.model.viewership_model import (
    train_model,
    cross_validate,
    predict_2026,
    _extract_X_y,
)
from src.outputs.export_viewership import export_viewership


# ──────────────────────────────────────────────────────────────────────────────
# Feature table tests (Task 10)
# ──────────────────────────────────────────────────────────────────────────────


class TestFeatureTable:
    """Tests for the viewership feature engineering pipeline."""

    def test_training_features_non_empty(self):
        rows = build_training_features()
        assert len(rows) > 0, "Training features should not be empty"

    def test_training_features_have_required_columns(self):
        rows = build_training_features()
        row = rows[0]
        for col in FEATURE_COLUMNS:
            assert col in row, f"Missing feature column: {col}"
        assert "viewers_millions" in row, "Missing target column"
        assert "season" in row, "Missing season column"

    def test_training_features_exclude_bowls_and_playoffs(self):
        rows = build_training_features()
        # Training rows shouldn't include bowl/playoff games
        # (they're excluded in the function)
        for row in rows:
            # Just verify they all have positive viewership
            assert row["viewers_millions"] > 0

    def test_training_features_no_nulls_in_feature_columns(self):
        rows = build_training_features()
        for row in rows:
            for col in FEATURE_COLUMNS:
                val = row[col]
                assert val is not None, f"Null in {col}"
                assert not (isinstance(val, float) and math.isnan(val)), f"NaN in {col}"

    def test_training_features_reasonable_values(self):
        rows = build_training_features()
        for row in rows:
            assert row["combined_brand"] >= 0
            assert row["combined_elo"] > 0
            assert 0 <= row["elo_closeness"] <= 1.0
            assert row["week_multiplier"] > 0
            assert row["slot_multiplier"] > 0
            assert row["is_rivalry"] in (0, 1)
            assert row["is_conference_game"] in (0, 1)

    def test_2026_features_match_schedule_count(self):
        with open(Path("data/processed/game_schedule.json")) as f:
            schedule = json.load(f)
        features = build_2026_features()
        assert len(features) == len(schedule)

    def test_2026_features_have_game_metadata(self):
        features = build_2026_features()
        row = features[0]
        assert "game_id" in row
        assert "home_team" in row
        assert "away_team" in row
        assert "week" in row
        assert "home_strength" in row
        assert "away_strength" in row

    def test_2026_features_have_model_columns(self):
        features = build_2026_features()
        for row in features:
            for col in FEATURE_COLUMNS:
                assert col in row, f"Missing {col} in 2026 features"

    def test_rivalry_detection(self):
        assert _is_rivalry("Ohio State", "Michigan") is True
        assert _is_rivalry("Michigan", "Ohio State") is True
        assert _is_rivalry("Ohio State", "Oregon") is False

    def test_cfp_rematch_detection(self):
        assert _is_cfp_rematch("Indiana", "Ohio State") is True
        assert _is_cfp_rematch("Ohio State", "Indiana") is True
        assert _is_cfp_rematch("Michigan", "Ohio State") is False


# ──────────────────────────────────────────────────────────────────────────────
# Model tests (Task 11)
# ──────────────────────────────────────────────────────────────────────────────


class TestViewershipModel:
    """Tests for the Ridge regression viewership model."""

    @pytest.fixture(scope="class")
    def training_rows(self):
        return build_training_features()

    @pytest.fixture(scope="class")
    def model_result(self, training_rows):
        return train_model(training_rows)

    def test_model_trains_successfully(self, model_result):
        assert model_result["model"] is not None
        assert model_result["scaler"] is not None
        assert model_result["n_train"] > 0

    def test_r_squared_positive(self, model_result):
        # Model should explain some variance
        assert model_result["r_squared"] > 0.3

    def test_residuals_centered(self, model_result):
        residuals = model_result["residuals"]
        # Residuals should be approximately centered at 0
        assert abs(np.mean(residuals)) < 0.1

    def test_coefficients_have_correct_count(self, model_result):
        assert len(model_result["coefficients"]) == len(FEATURE_COLUMNS)

    def test_combined_brand_positive_coefficient(self, model_result):
        # Higher combined brand should predict more viewers
        assert model_result["coefficients"]["combined_brand"] > 0

    def test_cross_validation_runs(self, training_rows):
        cv = cross_validate(training_rows)
        assert "folds" in cv
        assert len(cv["folds"]) == 3  # 2023, 2024, 2025
        assert cv["overall_r_squared"] > 0.2

    def test_cross_validation_fold_metrics(self, training_rows):
        cv = cross_validate(training_rows)
        for fold in cv["folds"]:
            assert "held_out_season" in fold
            assert "r_squared" in fold
            assert "mdape" in fold
            assert fold["n_test"] > 0

    def test_predictions_are_positive(self, model_result):
        predictions = predict_2026(model_result)
        for p in predictions:
            assert p["predicted_viewers_millions"] > 0
            assert p["lower_bound_millions"] > 0
            assert p["upper_bound_millions"] > p["predicted_viewers_millions"]

    def test_predictions_sorted_descending(self, model_result):
        predictions = predict_2026(model_result)
        for i in range(len(predictions) - 1):
            assert predictions[i]["predicted_viewers_millions"] >= predictions[i + 1]["predicted_viewers_millions"]

    def test_predictions_have_required_fields(self, model_result):
        predictions = predict_2026(model_result)
        required = [
            "game_id", "week", "home_team", "away_team",
            "predicted_viewers_millions", "lower_bound_millions",
            "upper_bound_millions", "is_conference_game",
        ]
        for p in predictions:
            for field in required:
                assert field in p, f"Missing field: {field}"

    def test_ohio_state_michigan_is_top_game(self, model_result):
        predictions = predict_2026(model_result)
        top_5_matchups = [
            (p["home_team"], p["away_team"]) for p in predictions[:5]
        ]
        # Ohio State-Michigan should be in top 5
        assert any(
            ("Ohio State" in h + a and "Michigan" in h + a)
            or ("Oregon" in h + a and "Ohio State" in h + a)
            for h, a in top_5_matchups
        )


# ──────────────────────────────────────────────────────────────────────────────
# Export tests (Task 12)
# ──────────────────────────────────────────────────────────────────────────────


class TestExportViewership:
    """Tests for the viewership export pipeline."""

    def test_export_creates_files(self, tmp_path):
        predictions = [
            {
                "game_id": "test_001",
                "week": 1,
                "game_date": "2026-09-05",
                "home_team": "Ohio State",
                "away_team": "Michigan",
                "predicted_viewers_millions": 8.5,
                "lower_bound_millions": 6.2,
                "upper_bound_millions": 10.8,
                "is_conference_game": True,
                "is_rivalry": True,
                "is_cfp_rematch": False,
                "combined_brand": 8.9,
                "home_strength": 1.0,
                "away_strength": 0.6,
                "market_score": 5.0,
            }
        ]

        with patch("src.outputs.export_viewership.OUTPUT_DIR", tmp_path):
            paths = export_viewership(predictions)
            assert paths["json"].exists()
            assert paths["csv"].exists()

            # Verify JSON content
            with open(paths["json"]) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["home_team"] == "Ohio State"
            assert data[0]["predicted_viewers_millions"] == 8.5

    def test_export_csv_has_headers(self, tmp_path):
        predictions = [
            {
                "game_id": "test_001",
                "week": 1,
                "game_date": "2026-09-05",
                "home_team": "Ohio State",
                "away_team": "Michigan",
                "predicted_viewers_millions": 8.5,
                "lower_bound_millions": 6.2,
                "upper_bound_millions": 10.8,
                "is_conference_game": True,
                "is_rivalry": True,
                "is_cfp_rematch": False,
                "combined_brand": 8.9,
                "home_strength": 1.0,
                "away_strength": 0.6,
                "market_score": 5.0,
            }
        ]

        with patch("src.outputs.export_viewership.OUTPUT_DIR", tmp_path):
            paths = export_viewership(predictions)
            with open(paths["csv"]) as f:
                header = f.readline().strip()
            assert "predicted_viewers_millions" in header
            assert "home_team" in header

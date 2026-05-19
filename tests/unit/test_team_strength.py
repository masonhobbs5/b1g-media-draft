"""
tests/unit/test_team_strength.py
────────────────────────────────
Unit tests for src/model/team_strength.py
"""

import pytest

from src.model.team_strength import (
    _normalize_values,
    calibrate_home_field,
    compute_composite_ratings,
    extract_elo_2025,
    DEFAULT_WEIGHTS,
    DEFAULT_HOME_FIELD_ELO,
    NEW_COACH_PENALTY_ELO,
)


class TestNormalizeValues:
    def test_basic_normalization(self):
        result = _normalize_values({"a": 10, "b": 20, "c": 30})
        assert result["a"] == 0.0
        assert result["c"] == 1.0
        assert result["b"] == pytest.approx(0.5)

    def test_single_value(self):
        result = _normalize_values({"a": 5})
        assert result["a"] == 0.5

    def test_all_same(self):
        result = _normalize_values({"a": 10, "b": 10, "c": 10})
        assert all(v == 0.5 for v in result.values())

    def test_empty(self):
        assert _normalize_values({}) == {}

    def test_negative_values(self):
        result = _normalize_values({"a": -10, "b": 0, "c": 10})
        assert result["a"] == 0.0
        assert result["b"] == pytest.approx(0.5)
        assert result["c"] == 1.0


class TestExtractElo2025:
    def test_returns_all_teams(self):
        elo = extract_elo_2025()
        from config.constants import BIG_TEN_TEAMS
        # Should have Elo for most/all B1G teams
        assert len(elo) >= 17  # at least 17 of 19
        for team in elo:
            assert team in BIG_TEN_TEAMS

    def test_reasonable_elo_range(self):
        elo = extract_elo_2025()
        for team, rating in elo.items():
            assert 1000 <= rating <= 2500, f"{team} has unreasonable Elo: {rating}"


class TestCalibrateHomeField:
    def test_returns_expected_keys(self):
        result = calibrate_home_field()
        assert "home_win_pct" in result
        assert "games_counted" in result
        assert "home_wins" in result
        assert "away_wins" in result
        assert "elo_advantage" in result

    def test_home_advantage_positive(self):
        result = calibrate_home_field()
        assert result["home_win_pct"] > 0.5, "Home teams should win >50%"
        assert result["elo_advantage"] > 0

    def test_games_counted_reasonable(self):
        result = calibrate_home_field()
        # 3 years × ~140 games/year = ~420 games expected
        assert result["games_counted"] >= 300

    def test_wins_sum_to_total(self):
        result = calibrate_home_field()
        assert result["home_wins"] + result["away_wins"] == result["games_counted"]


class TestCompositeRatings:
    def test_returns_all_teams(self):
        ratings = compute_composite_ratings()
        from config.constants import BIG_TEN_TEAMS
        teams = {r["team"] for r in ratings}
        assert teams == BIG_TEN_TEAMS

    def test_scores_in_range(self):
        ratings = compute_composite_ratings()
        for r in ratings:
            assert 0.0 <= r["composite_score"] <= 1.0
            assert 0.0 <= r["odds_normalized"] <= 1.0
            assert 0.0 <= r["sp_plus_normalized"] <= 1.0
            assert 0.0 <= r["elo_normalized"] <= 1.0

    def test_has_max_and_min(self):
        ratings = compute_composite_ratings()
        scores = [r["composite_score"] for r in ratings]
        assert max(scores) == 1.0
        assert min(scores) == 0.0

    def test_coach_penalty_applied(self):
        # With penalty
        with_penalty = compute_composite_ratings(apply_coach_penalty=True)
        # Without penalty
        no_penalty = compute_composite_ratings(apply_coach_penalty=False)

        # New coaches should have lower raw scores with penalty
        coach_teams = [r["team"] for r in with_penalty if r["new_coach_2026"]]
        assert len(coach_teams) >= 1

        for team in coach_teams:
            wp = next(r for r in with_penalty if r["team"] == team)
            np = next(r for r in no_penalty if r["team"] == team)
            assert wp["composite_raw"] < np["composite_raw"]

    def test_custom_weights(self):
        # All weight on odds
        ratings_odds = compute_composite_ratings(
            weights={"odds": 1.0, "sp_plus": 0.0, "elo": 0.0},
            apply_coach_penalty=False,
        )
        # The team with highest odds should be #1
        top = max(ratings_odds, key=lambda r: r["composite_score"])
        assert top["odds_normalized"] == 1.0

    def test_required_fields_present(self):
        ratings = compute_composite_ratings()
        expected_keys = {
            "team", "odds_normalized", "sp_plus_normalized", "elo_normalized",
            "composite_raw", "composite_score", "new_coach_2026",
            "elo_2025_raw", "sp_plus_2025_raw",
        }
        for r in ratings:
            assert set(r.keys()) == expected_keys


class TestDefaultWeights:
    def test_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_positive(self):
        assert all(v > 0 for v in DEFAULT_WEIGHTS.values())

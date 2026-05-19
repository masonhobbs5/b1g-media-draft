"""
tests/unit/test_win_probability.py
──────────────────────────────────
Unit tests for src/model/win_probability.py
"""

import pytest

from src.model.win_probability import (
    win_probability,
    compute_all_game_probabilities,
    PROB_FLOOR,
    PROB_CEIL,
    SCALE,
)


class TestWinProbability:
    """Core win_probability function tests."""

    def test_equal_teams_neutral_site(self):
        """Equal teams on neutral site should be 50/50."""
        p = win_probability(0.5, 0.5, neutral_site=True, hfa=0.1)
        assert p == pytest.approx(0.5, abs=0.001)

    def test_equal_teams_home_advantage(self):
        """Equal teams at home should favor home team."""
        p = win_probability(0.5, 0.5, neutral_site=False, hfa=0.1)
        assert p > 0.5

    def test_symmetry(self):
        """P(A beats B at neutral) = 1 - P(B beats A at neutral)."""
        p1 = win_probability(0.8, 0.3, neutral_site=True, hfa=0.1)
        p2 = win_probability(0.3, 0.8, neutral_site=True, hfa=0.1)
        assert p1 + p2 == pytest.approx(1.0, abs=0.001)

    def test_monotonicity_home_strength(self):
        """Higher home strength → higher home win prob."""
        p_low = win_probability(0.3, 0.5, neutral_site=True, hfa=0.0)
        p_mid = win_probability(0.5, 0.5, neutral_site=True, hfa=0.0)
        p_high = win_probability(0.7, 0.5, neutral_site=True, hfa=0.0)
        assert p_low < p_mid < p_high

    def test_monotonicity_away_strength(self):
        """Higher away strength → lower home win prob."""
        p_low = win_probability(0.5, 0.3, neutral_site=True, hfa=0.0)
        p_mid = win_probability(0.5, 0.5, neutral_site=True, hfa=0.0)
        p_high = win_probability(0.5, 0.7, neutral_site=True, hfa=0.0)
        assert p_low > p_mid > p_high

    def test_clamped_floor(self):
        """Massive away advantage should still be >= PROB_FLOOR."""
        p = win_probability(0.0, 1.0, neutral_site=True, hfa=0.0)
        assert p >= PROB_FLOOR

    def test_clamped_ceil(self):
        """Massive home advantage should still be <= PROB_CEIL."""
        p = win_probability(1.0, 0.0, neutral_site=True, hfa=0.0)
        assert p <= PROB_CEIL

    def test_neutral_site_removes_hfa(self):
        """Neutral site should give same result regardless of hfa value."""
        p1 = win_probability(0.6, 0.4, neutral_site=True, hfa=0.0)
        p2 = win_probability(0.6, 0.4, neutral_site=True, hfa=0.2)
        assert p1 == pytest.approx(p2, abs=0.001)

    def test_hfa_increases_home_prob(self):
        """Larger HFA → higher home win prob."""
        p_no_hfa = win_probability(0.5, 0.5, neutral_site=False, hfa=0.0)
        p_with_hfa = win_probability(0.5, 0.5, neutral_site=False, hfa=0.15)
        assert p_with_hfa > p_no_hfa

    def test_calibration_strong_vs_weak(self):
        """Top team (1.0) vs bottom team (0.0) at home should be very high."""
        p = win_probability(1.0, 0.0, neutral_site=False, hfa=0.1)
        assert p > 0.95

    def test_calibration_close_matchup(self):
        """Teams 0.1 apart should be a competitive game (55-75% range)."""
        p = win_probability(0.55, 0.45, neutral_site=True, hfa=0.0)
        assert 0.55 < p < 0.75


class TestComputeAllGameProbabilities:
    """Integration tests using real processed data."""

    def test_returns_all_games(self):
        probs = compute_all_game_probabilities()
        assert len(probs) >= 130  # Should be ~135 games

    def test_required_fields(self):
        probs = compute_all_game_probabilities()
        expected_keys = {
            "game_id", "week", "home_team", "away_team",
            "home_win_prob", "away_win_prob",
            "home_strength", "away_strength",
            "neutral_site", "is_conference_game",
        }
        for p in probs:
            assert set(p.keys()) == expected_keys

    def test_probabilities_sum_to_one(self):
        probs = compute_all_game_probabilities()
        for p in probs:
            assert p["home_win_prob"] + p["away_win_prob"] == pytest.approx(1.0, abs=0.001)

    def test_probabilities_in_valid_range(self):
        probs = compute_all_game_probabilities()
        for p in probs:
            assert PROB_FLOOR <= p["home_win_prob"] <= PROB_CEIL
            assert PROB_FLOOR <= p["away_win_prob"] <= PROB_CEIL

    def test_ohio_state_favored_at_home_vs_weak(self):
        """Ohio State at home vs a non-conference opponent should be heavily favored."""
        probs = compute_all_game_probabilities()
        osu_home = [p for p in probs if p["home_team"] == "Ohio State" and not p["is_conference_game"]]
        if osu_home:
            assert osu_home[0]["home_win_prob"] > 0.85

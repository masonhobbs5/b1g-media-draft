"""
tests/unit/test_season_simulator.py
───────────────────────────────────
Unit tests for src/simulation/season_simulator.py
"""

import pytest
import numpy as np

from src.simulation.season_simulator import (
    simulate_season,
    summarize_results,
)
from config.constants import BIG_TEN_TEAMS


class TestSimulateSeason:
    """Tests for the core simulation engine."""

    def test_deterministic_with_seed(self):
        """Same seed produces identical results."""
        r1 = simulate_season(n_iterations=100, seed=123)
        r2 = simulate_season(n_iterations=100, seed=123)
        for team in BIG_TEN_TEAMS:
            np.testing.assert_array_equal(
                r1[team]["total_wins"], r2[team]["total_wins"]
            )

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        r1 = simulate_season(n_iterations=1000, seed=1)
        r2 = simulate_season(n_iterations=1000, seed=2)
        # At least one team should differ
        any_diff = any(
            not np.array_equal(r1[t]["total_wins"], r2[t]["total_wins"])
            for t in BIG_TEN_TEAMS
        )
        assert any_diff

    def test_all_teams_present(self):
        """All B1G teams appear in results."""
        results = simulate_season(n_iterations=100, seed=42)
        assert set(results.keys()) == BIG_TEN_TEAMS

    def test_wins_within_bounds(self):
        """No team can have more wins than games played."""
        results = simulate_season(n_iterations=100, seed=42)
        for team, data in results.items():
            max_possible = data["games_played"]
            assert np.all(data["total_wins"] <= max_possible)
            assert np.all(data["total_wins"] >= 0)

    def test_conf_wins_within_bounds(self):
        """Conference wins cannot exceed conference games."""
        results = simulate_season(n_iterations=100, seed=42)
        for team, data in results.items():
            max_conf = data["conf_games_played"]
            assert np.all(data["conf_wins"] <= max_conf)
            assert np.all(data["conf_wins"] >= 0)

    def test_iteration_count_correct(self):
        """Output arrays have the requested number of iterations."""
        n = 500
        results = simulate_season(n_iterations=n, seed=42)
        for team, data in results.items():
            assert len(data["total_wins"]) == n
            assert len(data["conf_wins"]) == n

    def test_toy_schedule(self):
        """Simulate with a controlled 2-game schedule to verify correctness."""
        toy_probs = [
            {
                "game_id": "1", "week": 1,
                "home_team": "Ohio State", "away_team": "Michigan",
                "home_win_prob": 1.0, "away_win_prob": 0.0,
                "home_strength": 1.0, "away_strength": 0.5,
                "neutral_site": False, "is_conference_game": True,
            },
            {
                "game_id": "2", "week": 2,
                "home_team": "Ohio State", "away_team": "Indiana",
                "home_win_prob": 0.0, "away_win_prob": 1.0,
                "home_strength": 1.0, "away_strength": 1.0,
                "neutral_site": False, "is_conference_game": True,
            },
        ]
        results = simulate_season(n_iterations=100, seed=42, win_probs=toy_probs)
        # Ohio State: always wins game 1, always loses game 2 → 1 win
        assert np.all(results["Ohio State"]["total_wins"] == 1)
        assert np.all(results["Ohio State"]["conf_wins"] == 1)
        # Michigan: always loses game 1 → 0 wins
        assert np.all(results["Michigan"]["total_wins"] == 0)
        # Indiana: always wins game 2 → 1 win
        assert np.all(results["Indiana"]["total_wins"] == 1)


class TestSummarizeResults:
    """Tests for the summary/aggregation layer."""

    def test_returns_all_teams(self):
        results = simulate_season(n_iterations=100, seed=42)
        summaries = summarize_results(results)
        teams = {s["team"] for s in summaries}
        assert teams == BIG_TEN_TEAMS

    def test_sorted_by_mean_wins(self):
        results = simulate_season(n_iterations=1000, seed=42)
        summaries = summarize_results(results)
        wins = [s["mean_wins"] for s in summaries]
        assert wins == sorted(wins, reverse=True)

    def test_win_distribution_sums_to_one(self):
        results = simulate_season(n_iterations=1000, seed=42)
        summaries = summarize_results(results)
        for s in summaries:
            total = sum(s["win_distribution"].values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_threshold_probabilities_monotonic(self):
        """P(11+ wins) <= P(10+ wins) <= P(8+ wins)."""
        results = simulate_season(n_iterations=1000, seed=42)
        summaries = summarize_results(results)
        for s in summaries:
            assert s["p_11_plus_wins"] <= s["p_10_plus_wins"] <= s["p_8_plus_wins"]

    def test_mean_wins_reasonable(self):
        """Mean wins should be between 0 and games_played."""
        results = simulate_season(n_iterations=1000, seed=42)
        summaries = summarize_results(results)
        for s in summaries:
            assert 0 <= s["mean_wins"] <= s["games_played"]

    def test_required_fields_present(self):
        results = simulate_season(n_iterations=100, seed=42)
        summaries = summarize_results(results)
        expected = {
            "team", "games_played", "conf_games_played",
            "mean_wins", "mean_conf_wins", "median_wins", "std_wins",
            "win_distribution", "conf_win_distribution",
            "p_8_plus_wins", "p_10_plus_wins", "p_11_plus_wins",
            "p_undefeated_conf",
        }
        for s in summaries:
            assert set(s.keys()) == expected

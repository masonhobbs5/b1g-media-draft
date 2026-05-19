"""
tests/integration/test_pipeline.py
───────────────────────────────────
End-to-end integration tests for the B1G simulation pipeline.

These tests run the full pipeline on real processed data with minimal
iterations, verifying all artifacts are produced with correct structure.
No external API calls are made — only processed JSON files (committed to repo)
are consumed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import settings
from src.model.win_probability import build_win_probabilities, win_probability
from src.simulation.season_simulator import run_simulation, simulate_season
from src.outputs.export_records import export_records
from src.model.viewership_model import build_viewership_predictions
from src.outputs.export_viewership import export_viewership
from src.simulation.draft_simulator import build_draft_results
from src.outputs.export_draft import export_draft


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_ITERS = 100  # Fast iterations for integration tests
SEED = 99


@pytest.fixture(scope="module")
def win_probs():
    """Compute win probabilities once for the module."""
    return build_win_probabilities()


@pytest.fixture(scope="module")
def simulation_results():
    """Run Monte Carlo simulation with minimal iterations."""
    return run_simulation(n_iterations=N_ITERS, seed=SEED)


@pytest.fixture(scope="module")
def viewership_result():
    """Run viewership model pipeline."""
    return build_viewership_predictions()


@pytest.fixture(scope="module")
def draft_results():
    """Run draft simulation with minimal iterations."""
    return build_draft_results(n_iterations=N_ITERS, seed=SEED)


# ── Test: Full pipeline produces all 7 output files ──────────────────────────

class TestFullPipeline:
    """Verify the end-to-end pipeline produces valid artifact files."""

    def test_export_records_creates_files(self, simulation_results, tmp_path):
        paths = export_records(simulation_results, output_dir=tmp_path)
        assert paths["json"].exists()
        assert paths["csv"].exists()

        with open(paths["json"]) as f:
            data = json.load(f)
        assert len(data) > 0
        assert "team" in data[0]
        assert "mean_wins" in data[0]

    def test_export_viewership_creates_files(self, viewership_result, tmp_path):
        paths = export_viewership(viewership_result["predictions"], output_dir=tmp_path)
        assert paths["json"].exists()
        assert paths["csv"].exists()

        with open(paths["json"]) as f:
            data = json.load(f)
        assert len(data) > 0
        assert "predicted_viewers_millions" in data[0]

    def test_export_draft_creates_files(self, draft_results, tmp_path):
        paths = export_draft(draft_results, output_dir=tmp_path)
        assert paths["json"].exists()
        assert paths["csv"].exists()
        assert paths["weekly"].exists()

        with open(paths["json"]) as f:
            assignments = json.load(f)
        assert len(assignments) > 0
        assert "fox_prob" in assignments[0]

        with open(paths["weekly"]) as f:
            weekly = json.load(f)
        assert "season_totals" in weekly
        assert "FOX" in weekly["season_totals"]


# ── Test: Win probability model correctness ───────────────────────────────────

class TestWinProbabilities:
    """Verify win probability pipeline produces sane game-level outputs."""

    def test_all_games_scored(self, win_probs):
        """Every game in schedule gets a probability."""
        with open(settings.PROCESSED_DIR / "game_schedule.json") as f:
            schedule = json.load(f)
        assert len(win_probs) == len(schedule)

    def test_probabilities_sum_to_one(self, win_probs):
        for game in win_probs:
            total = game["home_win_prob"] + game["away_win_prob"]
            assert abs(total - 1.0) < 1e-6, f"Game {game['game_id']} probs sum to {total}"

    def test_no_extreme_probabilities(self, win_probs):
        """No game should be exactly 0 or 1 (clipping to floor/ceil)."""
        for game in win_probs:
            assert 0.01 <= game["home_win_prob"] <= 0.99


# ── Test: Season simulation structure ─────────────────────────────────────────

class TestSeasonSimulation:
    """Verify simulation produces correct structure and reasonable values."""

    def test_all_big_ten_teams_present(self, simulation_results):
        from config.constants import BIG_TEN_TEAMS
        result_teams = {s["team"] for s in simulation_results}
        for team in BIG_TEN_TEAMS:
            assert team in result_teams, f"{team} missing from simulation results"

    def test_wins_bounded(self, simulation_results):
        """Mean wins should be within realistic bounds."""
        for s in simulation_results:
            assert 0 <= s["mean_wins"] <= s["games_played"]
            assert 0 <= s["mean_conf_wins"] <= s["conf_games_played"]

    def test_probabilities_valid(self, simulation_results):
        """Threshold probabilities should be in [0, 1]."""
        for s in simulation_results:
            assert 0 <= s["p_8_plus_wins"] <= 1
            assert 0 <= s["p_10_plus_wins"] <= 1
            assert 0 <= s["p_11_plus_wins"] <= 1

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        r1 = run_simulation(n_iterations=50, seed=77)
        r2 = run_simulation(n_iterations=50, seed=77)
        for s1, s2 in zip(r1, r2):
            assert s1["mean_wins"] == s2["mean_wins"]
            assert s1["team"] == s2["team"]


# ── Test: Viewership model outputs ───────────────────────────────────────────

class TestViewershipModel:
    """Verify viewership predictions have correct structure."""

    def test_predictions_cover_all_games(self, viewership_result):
        with open(settings.PROCESSED_DIR / "game_schedule.json") as f:
            schedule = json.load(f)
        assert len(viewership_result["predictions"]) == len(schedule)

    def test_predictions_positive(self, viewership_result):
        for p in viewership_result["predictions"]:
            assert p["predicted_viewers_millions"] > 0
            assert p["lower_bound_millions"] >= 0
            assert p["upper_bound_millions"] >= p["predicted_viewers_millions"]

    def test_cv_diagnostics_present(self, viewership_result):
        """Model should report cross-validation metrics."""
        assert "diagnostics" in viewership_result
        diag = viewership_result["diagnostics"]
        assert "cv_results" in diag
        cv = diag["cv_results"]
        assert "overall_r_squared" in cv
        assert cv["overall_r_squared"] > 0  # Model should explain some variance


# ── Test: Draft simulation constraints ────────────────────────────────────────

class TestDraftSimulation:
    """Verify draft simulation respects constraints end-to-end."""

    def test_no_notre_dame_games(self, draft_results):
        """Notre Dame games must be excluded from draft pool."""
        for g in draft_results["enriched_assignments"]:
            assert g["home_team"] != "Notre Dame"
            assert g["away_team"] != "Notre Dame"

    def test_probabilities_valid(self, draft_results):
        for g in draft_results["enriched_assignments"]:
            total = g["fox_prob"] + g["cbs_prob"] + g["nbc_prob"] + g["undrafted_prob"]
            assert abs(total - 1.0) < 0.01, f"Probs sum to {total} for {g['game_id']}"

    def test_nbc_zero_in_nd_weeks(self, draft_results):
        """NBC should never pick in pre-committed Notre Dame weeks."""
        nd_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)
        for g in draft_results["enriched_assignments"]:
            if g["week"] in nd_weeks:
                assert g["nbc_prob"] == 0.0

    def test_weekly_viewers_all_networks(self, draft_results):
        weekly = draft_results["avg_weekly_viewers"]
        for net in ("FOX", "CBS", "NBC"):
            assert net in weekly
            assert len(weekly[net]) > 0

    def test_fox_dominates_total_viewers(self, draft_results):
        """FOX has more picks and should have highest season total."""
        weekly = draft_results["avg_weekly_viewers"]
        fox_total = sum(weekly["FOX"].values())
        cbs_total = sum(weekly["CBS"].values())
        nbc_total = sum(weekly["NBC"].values())
        assert fox_total > cbs_total
        assert fox_total > nbc_total


# ── Test: Toy data pipeline (no file dependencies) ───────────────────────────

class TestToyPipeline:
    """Run simulation on synthetic win probabilities to prove the engine works
    independently of real data files."""

    def test_toy_season_simulation(self):
        """4-team round-robin with known probabilities using real B1G teams."""
        toy_probs = [
            {"game_id": "g1", "week": 1, "home_team": "Ohio State", "away_team": "Michigan",
             "home_win_prob": 0.7, "away_win_prob": 0.3, "is_conference_game": True},
            {"game_id": "g2", "week": 1, "home_team": "Oregon", "away_team": "USC",
             "home_win_prob": 0.6, "away_win_prob": 0.4, "is_conference_game": True},
            {"game_id": "g3", "week": 2, "home_team": "Ohio State", "away_team": "Oregon",
             "home_win_prob": 0.55, "away_win_prob": 0.45, "is_conference_game": True},
            {"game_id": "g4", "week": 2, "home_team": "Michigan", "away_team": "USC",
             "home_win_prob": 0.5, "away_win_prob": 0.5, "is_conference_game": True},
            {"game_id": "g5", "week": 3, "home_team": "Ohio State", "away_team": "USC",
             "home_win_prob": 0.8, "away_win_prob": 0.2, "is_conference_game": True},
            {"game_id": "g6", "week": 3, "home_team": "Michigan", "away_team": "Oregon",
             "home_win_prob": 0.45, "away_win_prob": 0.55, "is_conference_game": True},
        ]

        results = simulate_season(n_iterations=5000, seed=42, win_probs=toy_probs)

        # All 4 teams should appear in results (among all B1G teams)
        for team in ("Ohio State", "Michigan", "Oregon", "USC"):
            assert team in results
            assert results[team]["games_played"] == 3
            assert results[team]["conf_games_played"] == 3

        # Ohio State is strongest (0.7, 0.55, 0.8 home probs) — should avg ~2 wins
        osu_mean = results["Ohio State"]["total_wins"].mean()
        assert 1.8 < osu_mean < 2.4

        # USC is weakest — should average ~1 win
        usc_mean = results["USC"]["total_wins"].mean()
        assert 0.6 < usc_mean < 1.5

        # Teams not in toy schedule should have 0 games
        assert results["Illinois"]["games_played"] == 0

    def test_toy_win_probability_symmetry(self):
        """Verify win probability function respects symmetry."""
        p_home = win_probability(0.8, 0.5, neutral_site=True)
        p_reversed = win_probability(0.5, 0.8, neutral_site=True)
        assert abs(p_home + p_reversed - 1.0) < 1e-6

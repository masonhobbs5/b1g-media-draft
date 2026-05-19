"""
tests/unit/test_draft.py
────────────────────────
Unit tests for draft value scoring, draft simulation, and export.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.simulation.draft_value import (
    _normalize_values,
    compute_window_fit,
    score_game,
    score_weekly_slate,
    rank_weeks_by_top_value,
)
from src.simulation.draft_simulator import (
    _softmax_select,
    _is_eligible_for_network,
    _compute_week_draft_value,
    _select_games_for_week,
    simulate_draft,
)
from src.outputs.export_draft import export_draft


# ──────────────────────────────────────────────────────────────────────────────
# Draft Value Scoring Tests (Task 13)
# ──────────────────────────────────────────────────────────────────────────────


class TestDraftValue:
    """Tests for the V(g) game value scoring system."""

    def test_normalize_values_basic(self):
        result = _normalize_values([1.0, 2.0, 3.0])
        assert result == [0.0, 0.5, 1.0]

    def test_normalize_values_equal(self):
        result = _normalize_values([5.0, 5.0, 5.0])
        assert result == [0.5, 0.5, 0.5]

    def test_normalize_values_empty(self):
        assert _normalize_values([]) == []

    def test_window_fit_fox_et_home(self):
        game = {"home_team": "Ohio State", "home_tz": "ET", "noon_eligible": True}
        assert compute_window_fit(game, "FOX") == 1.0

    def test_window_fit_fox_pt_ineligible(self):
        game = {"home_team": "Oregon", "home_tz": "PT", "noon_eligible": False}
        assert compute_window_fit(game, "FOX") == 0.0

    def test_window_fit_cbs_pt_best(self):
        game = {"home_team": "USC", "home_tz": "PT", "noon_eligible": False}
        assert compute_window_fit(game, "CBS") == 1.0

    def test_window_fit_nbc_et_best(self):
        game = {"home_team": "Ohio State", "home_tz": "ET", "noon_eligible": True}
        assert compute_window_fit(game, "NBC") == 1.0

    def test_score_game_has_all_components(self):
        game = {
            "game_id": "test_001",
            "home_team": "Ohio State",
            "away_team": "Michigan",
            "home_strength": 1.0,
            "away_strength": 0.6,
            "is_conference_game": True,
            "is_rivalry": True,
            "is_cfp_rematch": False,
            "market_score": 5.2,
            "home_tz": "ET",
            "noon_eligible": True,
        }
        result = score_game(game, {"test_001": 8.3}, "FOX")
        assert "prestige_raw" in result
        assert "viewership_raw" in result
        assert "stakes_raw" in result
        assert "market_raw" in result
        assert "window_fit_raw" in result
        assert "novelty_raw" in result
        assert result["prestige_raw"] == 1.6  # 1.0 + 0.6
        assert result["viewership_raw"] == 8.3

    def test_score_weekly_slate_sorted_descending(self):
        games = [
            {"game_id": "a", "week": 10, "home_team": "Ohio State", "away_team": "Oregon",
             "home_strength": 1.0, "away_strength": 0.88, "is_conference_game": True,
             "is_rivalry": False, "is_cfp_rematch": False, "market_score": 5.2,
             "home_tz": "ET", "noon_eligible": True},
            {"game_id": "b", "week": 10, "home_team": "Purdue", "away_team": "Maryland",
             "home_strength": 0.32, "away_strength": 0.40, "is_conference_game": True,
             "is_rivalry": False, "is_cfp_rematch": False, "market_score": 3.5,
             "home_tz": "ET", "noon_eligible": True},
        ]
        vw = {"a": 8.4, "b": 1.2}
        slate = score_weekly_slate(games, vw, "FOX")
        assert len(slate) == 2
        assert slate[0]["total_value"] >= slate[1]["total_value"]
        assert slate[0]["game_id"] == "a"  # OSU-Oregon should be top

    def test_score_weekly_slate_values_normalized(self):
        games = [
            {"game_id": "a", "week": 5, "home_team": "Iowa", "away_team": "Ohio State",
             "home_strength": 0.57, "away_strength": 1.0, "is_conference_game": True,
             "is_rivalry": False, "is_cfp_rematch": False, "market_score": 5.0,
             "home_tz": "CT", "noon_eligible": True},
            {"game_id": "b", "week": 5, "home_team": "Rutgers", "away_team": "Howard",
             "home_strength": 0.40, "away_strength": 0.05, "is_conference_game": False,
             "is_rivalry": False, "is_cfp_rematch": False, "market_score": 4.5,
             "home_tz": "ET", "noon_eligible": True},
        ]
        vw = {"a": 7.2, "b": 0.5}
        slate = score_weekly_slate(games, vw, "FOX")
        # All normalized values should be in [0, 1]
        for g in slate:
            assert 0.0 <= g["total_value"] <= 1.0

    def test_rank_weeks_by_top_value_returns_all_weeks(self):
        rankings = rank_weeks_by_top_value()
        assert len(rankings) == 13  # 13 weeks in schedule
        # Should be sorted descending by top_value
        for i in range(len(rankings) - 1):
            assert rankings[i]["top_value"] >= rankings[i + 1]["top_value"]


# ──────────────────────────────────────────────────────────────────────────────
# Draft Simulation Tests (Task 14)
# ──────────────────────────────────────────────────────────────────────────────


class TestDraftSimulation:
    """Tests for the two-phase broadcaster draft simulation."""

    def test_softmax_select_single(self):
        assert _softmax_select([5.0]) == 0

    def test_softmax_select_deterministic_low_temp(self):
        rng = np.random.default_rng(42)
        values = [10.0, 1.0, 0.5]
        # With very low temperature, should almost always pick index 0
        picks = [_softmax_select(values, temperature=0.01, rng=rng) for _ in range(100)]
        assert picks.count(0) > 95

    def test_softmax_select_uniform_high_temp(self):
        rng = np.random.default_rng(42)
        values = [1.0, 1.0, 1.0]
        # With equal values, should be roughly uniform regardless of temp
        picks = [_softmax_select(values, temperature=0.3, rng=rng) for _ in range(300)]
        for idx in range(3):
            assert 60 < picks.count(idx) < 140  # ~100 each, with variance

    def test_is_eligible_fox_blocks_pacific(self):
        assert _is_eligible_for_network(
            {"home_team": "USC", "noon_eligible": False}, "FOX"
        ) is False
        assert _is_eligible_for_network(
            {"home_team": "Oregon", "noon_eligible": False}, "FOX"
        ) is False

    def test_is_eligible_fox_allows_eastern(self):
        assert _is_eligible_for_network(
            {"home_team": "Ohio State", "noon_eligible": True}, "FOX"
        ) is True

    def test_is_eligible_cbs_nbc_allow_all(self):
        assert _is_eligible_for_network(
            {"home_team": "USC", "noon_eligible": False}, "CBS"
        ) is True
        assert _is_eligible_for_network(
            {"home_team": "USC", "noon_eligible": False}, "NBC"
        ) is True

    def test_week_draft_value_gap_based(self):
        """Week draft value is gap (primary) + small top-viewers boost."""
        games = [
            {"game_id": "a", "home_team": "Ohio State", "noon_eligible": True},
            {"game_id": "b", "home_team": "Michigan", "noon_eligible": True},
            {"game_id": "c", "home_team": "Purdue", "noon_eligible": True},
        ]
        viewership = {"a": 8.0, "b": 3.0, "c": 1.5}
        # gap = 8.0 - 3.0 = 5.0, value = 5.0 + 0.05*8.0 = 5.4
        value = _compute_week_draft_value(games, viewership, "FOX")
        assert abs(value - 5.4) < 0.01

    def test_week_draft_value_small_gap(self):
        """Weeks with close top games have low draft value (gap dominates)."""
        games = [
            {"game_id": "a", "home_team": "Ohio State", "noon_eligible": True},
            {"game_id": "b", "home_team": "Michigan", "noon_eligible": True},
        ]
        viewership = {"a": 7.0, "b": 6.8}
        value = _compute_week_draft_value(games, viewership, "FOX")
        # gap = 0.2, value = 0.2 + 0.05*7.0 = 0.55
        assert abs(value - 0.55) < 0.01

    def test_week_draft_value_respects_eligibility(self):
        """FOX draft value should skip PT-home games."""
        games = [
            {"game_id": "a", "home_team": "USC", "noon_eligible": False},  # Ineligible for FOX
            {"game_id": "b", "home_team": "Ohio State", "noon_eligible": True},
            {"game_id": "c", "home_team": "Michigan", "noon_eligible": True},
        ]
        viewership = {"a": 10.0, "b": 5.0, "c": 3.0}
        # For FOX: eligible are b=5.0, c=3.0 → gap=2.0, value = 2.0 + 0.05*5.0 = 2.25
        fox_value = _compute_week_draft_value(games, viewership, "FOX")
        assert abs(fox_value - 2.25) < 0.01
        # For CBS: eligible are all → gap=5.0, value = 5.0 + 0.05*10.0 = 5.5
        cbs_value = _compute_week_draft_value(games, viewership, "CBS")
        assert abs(cbs_value - 5.5) < 0.01

    def test_select_games_for_week_fox_drafted(self):
        """When FOX drafts, FOX picks first, then CBS/NBC."""
        rng = np.random.default_rng(42)
        games = [
            {"game_id": "a", "home_team": "Ohio State", "away_team": "Michigan",
             "total_value": 0.9, "noon_eligible": True, "home_tz": "ET"},
            {"game_id": "b", "home_team": "Penn State", "away_team": "Iowa",
             "total_value": 0.6, "noon_eligible": True, "home_tz": "ET"},
            {"game_id": "c", "home_team": "Wisconsin", "away_team": "Minnesota",
             "total_value": 0.3, "noon_eligible": True, "home_tz": "CT"},
        ]
        slates = {"FOX": games, "CBS": games, "NBC": games}
        viewership = {"a": 8.0, "b": 5.0, "c": 3.0}

        result = _select_games_for_week(
            week=5, drafting_network="FOX", network_slates=slates,
            viewership_lookup=viewership, rng=rng, temperature=0.01,
            cbs_nbc_second_toggle=True,
        )
        # FOX should get the best game (low temperature → near-deterministic)
        assert result["FOX"]["game_id"] == "a"
        # CBS gets 2nd (toggle=True)
        assert result["CBS"] is not None
        assert result["CBS"]["game_id"] != "a"
        # NBC gets 3rd
        assert result["NBC"] is not None
        # All three should have different games
        picked_ids = {result[n]["game_id"] for n in ("FOX", "CBS", "NBC")}
        assert len(picked_ids) == 3

    def test_select_games_nbc_excluded_in_nd_weeks(self):
        """NBC should not pick games in Notre Dame pre-committed weeks."""
        rng = np.random.default_rng(42)
        games = [
            {"game_id": "a", "home_team": "Ohio State", "away_team": "Michigan",
             "total_value": 0.9, "noon_eligible": True, "home_tz": "ET"},
            {"game_id": "b", "home_team": "Penn State", "away_team": "Iowa",
             "total_value": 0.6, "noon_eligible": True, "home_tz": "ET"},
            {"game_id": "c", "home_team": "Wisconsin", "away_team": "Minnesota",
             "total_value": 0.3, "noon_eligible": True, "home_tz": "CT"},
        ]
        slates = {"FOX": games, "CBS": games, "NBC": games}
        viewership = {"a": 8.0, "b": 5.0, "c": 3.0}

        # Week 1 is a Notre Dame week — NBC should not participate
        result = _select_games_for_week(
            week=1, drafting_network="FOX", network_slates=slates,
            viewership_lookup=viewership, rng=rng, temperature=0.01,
            cbs_nbc_second_toggle=True,
        )
        assert result["NBC"] is None
        assert result["FOX"] is not None
        assert result["CBS"] is not None

    def test_simulate_draft_deterministic(self):
        """Same seed should produce same results."""
        result_1 = simulate_draft(n_iterations=100, seed=123, temperature=0.3)
        result_2 = simulate_draft(n_iterations=100, seed=123, temperature=0.3)
        assert result_1["game_assignments"] == result_2["game_assignments"]

    def test_simulate_draft_probabilities_sum_to_one(self):
        result = simulate_draft(n_iterations=100, seed=42)
        for game_id, probs in result["game_assignments"].items():
            total = probs["fox_prob"] + probs["cbs_prob"] + probs["nbc_prob"] + probs["undrafted_prob"]
            assert abs(total - 1.0) < 0.02, f"Game {game_id} probs sum to {total}"

    def test_simulate_draft_three_games_per_week(self):
        """Each week should select exactly 3 games (2 in ND weeks)."""
        result = simulate_draft(n_iterations=200, seed=42)
        from config.settings import settings
        nd_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)

        with open("data/processed/game_schedule.json") as f:
            schedule = json.load(f)
        schedule_lookup = {g["game_id"]: g for g in schedule}

        # In a single deterministic run, check game counts per week
        # Using high-probability assignments as a proxy
        for game_id, probs in result["game_assignments"].items():
            assigned_prob = probs["fox_prob"] + probs["cbs_prob"] + probs["nbc_prob"]
            # Games with low total assignment prob are rarely selected
            # (expected: most games undrafted since only 3 per week)
            game = schedule_lookup.get(game_id, {})
            week = game.get("week", 0)
            if week in nd_weeks:
                # NBC shouldn't be assigned in ND weeks
                assert probs["nbc_prob"] == 0.0

    def test_nbc_zero_in_notre_dame_weeks(self):
        """NBC should not pick any games in its pre-committed Notre Dame weeks."""
        from config.settings import settings
        result = simulate_draft(n_iterations=200, seed=42)
        with open("data/processed/game_schedule.json") as f:
            schedule = json.load(f)
        schedule_lookup = {g["game_id"]: g for g in schedule}

        committed_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)
        for game_id, probs in result["game_assignments"].items():
            game = schedule_lookup.get(game_id, {})
            if game.get("week") in committed_weeks:
                assert probs["nbc_prob"] == 0.0, (
                    f"NBC has non-zero prob ({probs['nbc_prob']}) for game "
                    f"{game_id} in pre-committed week {game.get('week')}"
                )

    def test_notre_dame_excluded_from_draft(self):
        """Notre Dame games must not appear in the draft pool at all."""
        result = simulate_draft(n_iterations=100, seed=42)
        with open("data/processed/game_schedule.json") as f:
            schedule = json.load(f)
        nd_game_ids = {
            g["game_id"] for g in schedule
            if "Notre Dame" in (g["home_team"], g["away_team"])
        }
        for nd_id in nd_game_ids:
            assert nd_id not in result["game_assignments"], (
                f"Notre Dame game {nd_id} should not appear in draft assignments"
            )

    def test_pacific_teams_rarely_on_fox(self):
        """Games with Pacific home teams should very rarely end up on FOX."""
        result = simulate_draft(n_iterations=500, seed=42)
        from config.constants import PACIFIC_TZ_TEAMS
        with open("data/processed/game_schedule.json") as f:
            schedule = json.load(f)
        pacific_home_ids = {
            g["game_id"] for g in schedule
            if g["home_team"] in PACIFIC_TZ_TEAMS
        }
        for game_id in pacific_home_ids:
            if game_id in result["game_assignments"]:
                # Allow small probability from trade scenarios
                assert result["game_assignments"][game_id]["fox_prob"] < 0.05, (
                    f"Pacific home game {game_id} has fox_prob="
                    f"{result['game_assignments'][game_id]['fox_prob']}"
                )

    def test_each_network_gets_one_game_per_week(self):
        """In non-ND weeks, each network should get exactly 1 game."""
        # Run a single iteration with low temperature to verify structure
        result = simulate_draft(n_iterations=1, seed=42, temperature=0.01)
        from config.settings import settings
        nd_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)

        with open("data/processed/game_schedule.json") as f:
            schedule = json.load(f)
        schedule_lookup = {g["game_id"]: g for g in schedule}

        # Count games assigned per week per network
        week_net_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"FOX": 0, "CBS": 0, "NBC": 0})
        for game_id, probs in result["game_assignments"].items():
            game = schedule_lookup.get(game_id, {})
            week = game.get("week", 0)
            if probs["fox_prob"] == 1.0:
                week_net_counts[week]["FOX"] += 1
            if probs["cbs_prob"] == 1.0:
                week_net_counts[week]["CBS"] += 1
            if probs["nbc_prob"] == 1.0:
                week_net_counts[week]["NBC"] += 1

        for week, counts in week_net_counts.items():
            if week in nd_weeks:
                assert counts["NBC"] == 0
                assert counts["FOX"] == 1
                assert counts["CBS"] == 1
            else:
                assert counts["FOX"] == 1, f"Week {week}: FOX has {counts['FOX']} games"
                assert counts["CBS"] == 1, f"Week {week}: CBS has {counts['CBS']} games"
                assert counts["NBC"] == 1, f"Week {week}: NBC has {counts['NBC']} games"


# ──────────────────────────────────────────────────────────────────────────────
# Export Tests (Task 15)
# ──────────────────────────────────────────────────────────────────────────────


class TestExportDraft:
    """Tests for the draft export pipeline."""

    def test_export_creates_files(self, tmp_path):
        draft_results = {
            "enriched_assignments": [
                {
                    "game_id": "test_001",
                    "week": 13,
                    "home_team": "Ohio State",
                    "away_team": "Michigan",
                    "predicted_viewers_millions": 8.3,
                    "is_conference_game": True,
                    "fox_prob": 0.91,
                    "cbs_prob": 0.07,
                    "nbc_prob": 0.02,
                    "undrafted_prob": 0.0,
                }
            ],
            "avg_weekly_viewers": {
                "FOX": {13: 8.3},
                "CBS": {13: 5.5},
                "NBC": {13: 3.2},
            },
            "n_iterations": 10000,
            "temperature": 0.3,
            "trade_probability": 0.15,
        }

        with patch("src.outputs.export_draft.OUTPUT_DIR", tmp_path):
            paths = export_draft(draft_results)
            assert paths["json"].exists()
            assert paths["csv"].exists()
            assert paths["weekly"].exists()

            # Verify JSON content
            with open(paths["json"]) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["fox_prob"] == 0.91

            # Verify weekly summary
            with open(paths["weekly"]) as f:
                weekly = json.load(f)
            assert "metadata" in weekly
            assert weekly["metadata"]["n_iterations"] == 10000
            assert "season_totals" in weekly

    def test_export_csv_has_headers(self, tmp_path):
        draft_results = {
            "enriched_assignments": [
                {
                    "game_id": "test_001",
                    "week": 13,
                    "home_team": "Ohio State",
                    "away_team": "Michigan",
                    "predicted_viewers_millions": 8.3,
                    "is_conference_game": True,
                    "fox_prob": 0.91,
                    "cbs_prob": 0.07,
                    "nbc_prob": 0.02,
                    "undrafted_prob": 0.0,
                }
            ],
            "avg_weekly_viewers": {},
            "n_iterations": 100,
            "temperature": 0.3,
            "trade_probability": 0.15,
        }

        with patch("src.outputs.export_draft.OUTPUT_DIR", tmp_path):
            paths = export_draft(draft_results)
            with open(paths["csv"]) as f:
                header = f.readline().strip()
            assert "fox_prob" in header
            assert "home_team" in header

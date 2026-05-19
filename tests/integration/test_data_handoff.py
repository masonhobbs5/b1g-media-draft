"""
tests/integration/test_data_handoff.py
──────────────────────────────────────
Integration tests for the data-loading → model handoff chain.

Verifies that data loaders produce outputs in the schema expected by
downstream model and simulation code, without external API dependence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import settings
from config.constants import BIG_TEN_TEAMS
from src.utils.data_loaders import load_schedule, load_prestige, load_viewership


# ── Test: Data loaders return valid objects ───────────────────────────────────

class TestDataLoaders:
    """Verify data loaders parse processed files correctly."""

    def test_schedule_loads(self):
        games = load_schedule()
        assert len(games) > 0
        # Verify key fields exist on first game
        g = games[0]
        assert hasattr(g, "game_id")
        assert hasattr(g, "week")
        assert hasattr(g, "home_team")
        assert hasattr(g, "away_team")
        assert hasattr(g, "is_conference_game")

    def test_schedule_has_big_ten_teams(self):
        games = load_schedule()
        teams_in_schedule = set()
        for g in games:
            teams_in_schedule.add(g.home_team)
            teams_in_schedule.add(g.away_team)
        for team in BIG_TEN_TEAMS:
            assert team in teams_in_schedule, f"{team} not in schedule"

    def test_prestige_loads(self):
        prestige = load_prestige()
        assert len(prestige) > 0
        p = prestige[0]
        assert hasattr(p, "team")
        assert hasattr(p, "prestige_score")
        assert 0 <= p.prestige_score <= 1

    def test_viewership_loads(self):
        records = load_viewership()
        assert len(records) > 0
        r = records[0]
        assert hasattr(r, "season")
        assert hasattr(r, "viewers_millions")
        assert r.viewers_millions > 0


# ── Test: Data flows between pipeline stages ─────────────────────────────────

class TestPipelineHandoff:
    """Verify schema compatibility between pipeline stages."""

    def test_strength_json_schema(self):
        """fbs_strength.json has fields expected by win_probability module."""
        path = settings.PROCESSED_DIR / "fbs_strength.json"
        if not path.exists():
            pytest.skip("fbs_strength.json not yet generated")
        with open(path) as f:
            data = json.load(f)
        assert len(data) > 0
        entry = data[0]
        assert "team" in entry
        assert "composite_score" in entry
        assert isinstance(entry["composite_score"], (int, float))
        assert 0 <= entry["composite_score"] <= 1

    def test_win_probs_json_schema(self):
        """win_probabilities.json has fields expected by season_simulator."""
        path = settings.PROCESSED_DIR / "win_probabilities.json"
        if not path.exists():
            pytest.skip("win_probabilities.json not yet generated")
        with open(path) as f:
            data = json.load(f)
        required_keys = {"game_id", "week", "home_team", "away_team",
                         "home_win_prob", "away_win_prob", "is_conference_game"}
        for game in data:
            missing = required_keys - set(game.keys())
            assert not missing, f"Game {game.get('game_id')} missing: {missing}"
            assert 0 < game["home_win_prob"] < 1
            assert 0 < game["away_win_prob"] < 1

    def test_viewership_features_schema(self):
        """viewership_features.json has fields expected by viewership model."""
        path = settings.PROCESSED_DIR / "viewership_features.json"
        if not path.exists():
            pytest.skip("viewership_features.json not yet generated")
        with open(path) as f:
            data = json.load(f)
        assert "team_brands" in data
        assert "week_multipliers" in data
        assert "slot_multipliers" in data
        # Team brands should include top B1G teams
        assert "Ohio State" in data["team_brands"]
        assert "Michigan" in data["team_brands"]
        brand = data["team_brands"]["Ohio State"]
        assert "avg_viewers_millions" in brand
        assert brand["avg_viewers_millions"] > 0

    def test_schedule_to_viewership_game_ids_match(self):
        """Viewership output game_ids are a subset of schedule game_ids."""
        vw_path = settings.OUTPUT_DIR / "expected_viewership.json"
        if not vw_path.exists():
            pytest.skip("expected_viewership.json not yet generated")
        with open(settings.PROCESSED_DIR / "game_schedule.json") as f:
            schedule = json.load(f)
        with open(vw_path) as f:
            viewership = json.load(f)
        sched_ids = {g["game_id"] for g in schedule}
        vw_ids = {g["game_id"] for g in viewership}
        assert vw_ids == sched_ids

    def test_draft_game_ids_subset_of_schedule(self):
        """Draft output game_ids are a subset of schedule game_ids."""
        draft_path = settings.OUTPUT_DIR / "draft_assignments.json"
        if not draft_path.exists():
            pytest.skip("draft_assignments.json not yet generated")
        with open(settings.PROCESSED_DIR / "game_schedule.json") as f:
            schedule = json.load(f)
        with open(draft_path) as f:
            draft = json.load(f)
        sched_ids = {g["game_id"] for g in schedule}
        draft_ids = {g["game_id"] for g in draft}
        assert draft_ids.issubset(sched_ids)

    def test_draft_excludes_non_home_b1g(self):
        """Draft only includes games where a B1G team is home."""
        draft_path = settings.OUTPUT_DIR / "draft_assignments.json"
        if not draft_path.exists():
            pytest.skip("draft_assignments.json not yet generated")
        with open(draft_path) as f:
            draft = json.load(f)
        for g in draft:
            assert g["home_team"] in BIG_TEN_TEAMS, (
                f"Non-B1G home team {g['home_team']} in draft"
            )

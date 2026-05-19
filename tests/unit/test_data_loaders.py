"""
tests/unit/test_data_loaders.py
───────────────────────────────
Unit tests for src/utils/data_loaders.py
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.utils.data_loaders import (
    Game,
    TeamPrestige,
    ViewershipRecord,
    load_schedule,
    load_prestige,
    load_viewership,
    load_prestige_lookup,
    _validate_game,
    _validate_prestige,
    _validate_viewership,
)


# ===========================================================================
# Fixtures: minimal valid records
# ===========================================================================

@pytest.fixture
def valid_game_dict():
    return {
        "game_id": "12345",
        "week": 5,
        "game_date": "2026-10-03",
        "home_team": "Ohio State",
        "away_team": "Michigan",
        "is_conference_game": True,
        "home_tz": "ET",
        "home_dma_rank": 32,
        "market_score": 4.123,
        "noon_eligible": True,
        "is_rivalry_game": True,
        "is_cfp_rematch": False,
        "post_dst": False,
        "neutral_site": False,
        "source": "cfbd_api",
    }


@pytest.fixture
def valid_prestige_dict():
    return {
        "team": "Ohio State",
        "championship_odds_normalized": 0.978,
        "sp_plus_2025": 30.5,
        "sp_plus_normalized": 1.0,
        "prestige_score": 0.978,
        "dma_rank": 32,
        "market_score": 4.123,
        "timezone": "ET",
        "noon_eligible": True,
        "new_coach_2026": False,
    }


@pytest.fixture
def valid_viewership_dict():
    return {
        "season": 2024,
        "week": 12,
        "team_a": "Michigan",
        "team_b": "Ohio State",
        "viewers_millions": 17.5,
        "network": "FOX",
        "time_slot": "noon",
        "home_team": "Ohio State",
        "away_team": "Michigan",
        "home_points": 13,
        "away_points": 10,
        "home_pregame_elo": 2100,
        "away_pregame_elo": 1950,
        "is_conference_game": True,
        "is_bowl_game": False,
        "is_playoff_game": False,
        "source": "vision",
    }


# ===========================================================================
# Validation tests — valid records
# ===========================================================================

class TestValidateGame:
    def test_valid(self, valid_game_dict):
        game = _validate_game(valid_game_dict, 0)
        assert isinstance(game, Game)
        assert game.home_team == "Ohio State"
        assert game.week == 5

    def test_missing_field(self, valid_game_dict):
        del valid_game_dict["week"]
        with pytest.raises(ValueError, match="missing fields"):
            _validate_game(valid_game_dict, 0)

    def test_empty_game_id(self, valid_game_dict):
        valid_game_dict["game_id"] = ""
        with pytest.raises(ValueError, match="empty game_id"):
            _validate_game(valid_game_dict, 0)

    def test_invalid_week(self, valid_game_dict):
        valid_game_dict["week"] = -1
        with pytest.raises(ValueError, match="invalid week"):
            _validate_game(valid_game_dict, 0)

    def test_dma_rank_out_of_range(self, valid_game_dict):
        valid_game_dict["home_dma_rank"] = 0
        with pytest.raises(ValueError, match="invalid home_dma_rank"):
            _validate_game(valid_game_dict, 0)

    def test_negative_market_score(self, valid_game_dict):
        valid_game_dict["market_score"] = -0.5
        with pytest.raises(ValueError, match="negative market_score"):
            _validate_game(valid_game_dict, 0)


class TestValidatePrestige:
    def test_valid(self, valid_prestige_dict):
        p = _validate_prestige(valid_prestige_dict, 0)
        assert isinstance(p, TeamPrestige)
        assert p.team == "Ohio State"

    def test_missing_field(self, valid_prestige_dict):
        del valid_prestige_dict["prestige_score"]
        with pytest.raises(ValueError, match="missing fields"):
            _validate_prestige(valid_prestige_dict, 0)

    def test_prestige_out_of_range(self, valid_prestige_dict):
        valid_prestige_dict["prestige_score"] = 1.5
        with pytest.raises(ValueError, match="prestige_score out of"):
            _validate_prestige(valid_prestige_dict, 0)

    def test_odds_out_of_range(self, valid_prestige_dict):
        valid_prestige_dict["championship_odds_normalized"] = -0.1
        with pytest.raises(ValueError, match="championship_odds_normalized out of"):
            _validate_prestige(valid_prestige_dict, 0)

    def test_invalid_dma_rank(self, valid_prestige_dict):
        valid_prestige_dict["dma_rank"] = 250
        with pytest.raises(ValueError, match="invalid dma_rank"):
            _validate_prestige(valid_prestige_dict, 0)


class TestValidateViewership:
    def test_valid(self, valid_viewership_dict):
        v = _validate_viewership(valid_viewership_dict, 0)
        assert isinstance(v, ViewershipRecord)
        assert v.viewers_millions == 17.5

    def test_missing_field(self, valid_viewership_dict):
        del valid_viewership_dict["network"]
        with pytest.raises(ValueError, match="missing fields"):
            _validate_viewership(valid_viewership_dict, 0)

    def test_invalid_season(self, valid_viewership_dict):
        valid_viewership_dict["season"] = 1900
        with pytest.raises(ValueError, match="invalid season"):
            _validate_viewership(valid_viewership_dict, 0)

    def test_zero_viewers(self, valid_viewership_dict):
        valid_viewership_dict["viewers_millions"] = 0
        with pytest.raises(ValueError, match="non-positive viewers"):
            _validate_viewership(valid_viewership_dict, 0)

    def test_empty_team(self, valid_viewership_dict):
        valid_viewership_dict["team_a"] = ""
        with pytest.raises(ValueError, match="empty team name"):
            _validate_viewership(valid_viewership_dict, 0)


# ===========================================================================
# Loader tests — file I/O
# ===========================================================================

class TestLoadSchedule:
    def test_loads_real_data(self):
        games = load_schedule()
        assert len(games) >= 100
        assert all(isinstance(g, Game) for g in games)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("[]")
        with pytest.raises(ValueError, match="empty or not a list"):
            load_schedule(p)

    def test_not_a_list_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"game_id": "1"}')
        with pytest.raises(ValueError, match="empty or not a list"):
            load_schedule(p)


class TestLoadPrestige:
    def test_loads_real_data(self):
        teams = load_prestige()
        assert len(teams) == 19
        assert all(isinstance(t, TeamPrestige) for t in teams)

    def test_lookup(self):
        lookup = load_prestige_lookup()
        assert "Ohio State" in lookup
        assert lookup["Ohio State"].prestige_score > 0.9


class TestLoadViewership:
    def test_loads_real_data(self):
        records = load_viewership()
        assert len(records) >= 400
        assert all(isinstance(r, ViewershipRecord) for r in records)
        # Verify bowl/playoff flags exist
        bowls = [r for r in records if r.is_bowl_game]
        playoffs = [r for r in records if r.is_playoff_game]
        assert len(bowls) > 0
        assert len(playoffs) > 0

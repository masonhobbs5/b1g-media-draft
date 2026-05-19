"""
tests/unit/test_acquisition.py
─────────────────────────────
Unit tests for the data acquisition layer.
All HTTP calls are mocked — these tests run offline with no API keys needed.
Run: pytest tests/unit/test_acquisition.py -v
"""

import json
import math
from pathlib import Path

import pytest
import responses as responses_lib

from config.constants import BIG_TEN_TEAMS, TEAM_META, PRESTIGE_FALLBACK
from src.acquisition.odds_client import (
    _american_to_prob,
    _normalize,
    fetch_prestige_scores,
)
from src.acquisition.smw_scraper import _fuzzy_match_team


# ── odds_client tests ─────────────────────────────────────────────────────────

class TestAmericanToProb:
    def test_positive_odds(self):
        # +650 → 100/(650+100) ≈ 13.33%
        assert abs(_american_to_prob(650) - 0.1333) < 0.001

    def test_negative_odds(self):
        # -200 → 200/(200+100) = 66.67%
        assert abs(_american_to_prob(-200) - 0.6667) < 0.001

    def test_even_money(self):
        # +100 → 100/200 = 50%
        assert abs(_american_to_prob(100) - 0.5) < 0.001


class TestNormalize:
    def test_top_team_is_one(self):
        raw = {"Ohio State": 0.133, "Michigan": 0.050, "Indiana": 0.125}
        result = _normalize(raw)
        assert result["Ohio State"] == pytest.approx(1.0)

    def test_all_teams_present(self):
        """Normalize should return a value for every Big Ten team."""
        result = _normalize(PRESTIGE_FALLBACK)
        assert set(result.keys()) == BIG_TEN_TEAMS

    def test_scores_between_zero_and_one(self):
        result = _normalize(PRESTIGE_FALLBACK)
        for team, score in result.items():
            assert 0.0 <= score <= 1.0, f"{team} score {score} out of range"


class TestFallback:
    def test_fallback_returns_all_teams(self):
        """fetch_prestige_scores always uses PRESTIGE_FALLBACK and returns all 18 teams."""
        result = fetch_prestige_scores()
        assert set(result.keys()) == BIG_TEN_TEAMS


# ── smw_scraper tests ─────────────────────────────────────────────────────────

class TestFuzzyMatchTeam:
    @pytest.mark.parametrize("raw, expected", [
        ("Ohio State",       "Ohio State"),
        ("Buckeyes",         "Ohio State"),
        ("michigan",         "Michigan"),
        ("Wolverines",       "Michigan"),
        ("Penn State",       "Penn State"),
        ("Nittany Lions",    "Penn State"),
        ("USC Trojans",      "USC"),
        ("Notre Dame",       "Notre Dame"),  # Notre Dame joined B1G
        ("Alabama",          None),
        ("Hawkeyes",         "Iowa"),
        ("Cornhuskers",      "Nebraska"),
    ])
    def test_team_matching(self, raw, expected):
        assert _fuzzy_match_team(raw) == expected


# ── cfbd_client tests ─────────────────────────────────────────────────────────

class TestNormalizeGame:
    def test_conference_game_flag(self):
        from src.acquisition.cfbd_client import _normalize_game
        g = {
            "id": "1", "week": 5, "startDate": "2026-10-17T00:00:00",
            "homeTeam": "Indiana", "awayTeam": "Ohio State",
            "neutralSite": False,
        }
        result = _normalize_game(g)
        assert result["is_conference_game"] is True
        assert result["is_cfp_rematch"] is True

    def test_pacific_noon_ineligible(self):
        from src.acquisition.cfbd_client import _normalize_game
        g = {
            "id": "2", "week": 7, "startDate": "2026-10-31T00:00:00",
            "homeTeam": "Oregon", "awayTeam": "Indiana",
            "neutralSite": False,
        }
        result = _normalize_game(g)
        assert result["noon_eligible"] is False

    def test_post_dst_flag(self):
        from src.acquisition.cfbd_client import _normalize_game
        g = {
            "id": "3", "week": 11, "startDate": "2026-11-14T00:00:00",
            "homeTeam": "Ohio State", "awayTeam": "Michigan State",
            "neutralSite": False,
        }
        result = _normalize_game(g)
        assert result["post_dst"] is True

    def test_rivalry_flag(self):
        from src.acquisition.cfbd_client import _normalize_game
        g = {
            "id": "4", "week": 14, "startDate": "2026-11-28T00:00:00",
            "homeTeam": "Ohio State", "awayTeam": "Michigan",
            "neutralSite": False,
        }
        result = _normalize_game(g)
        assert result["is_rivalry_game"] is True

    def test_market_score_formula(self):
        from src.acquisition.cfbd_client import _normalize_game
        g = {
            "id": "5", "week": 3, "startDate": "2026-09-19T00:00:00",
            "homeTeam": "Rutgers", "awayTeam": "Illinois",
            "neutralSite": False,
        }
        result = _normalize_game(g)
        expected = round(math.log(1 + (211 - TEAM_META["Rutgers"]["dma_rank"])), 4)
        assert result["market_score"] == pytest.approx(expected, abs=0.001)

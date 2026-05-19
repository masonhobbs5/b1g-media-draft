"""
src/utils/data_loaders.py
─────────────────────────
Canonical typed loaders for the three processed datasets.
Each loader validates required fields and value ranges, raising ValueError
on malformed data so downstream code can rely on a stable schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.settings import settings

PROCESSED_DIR: Path = settings.PROCESSED_DIR


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass(frozen=True, slots=True)
class Game:
    game_id: str
    week: int
    game_date: Optional[str]
    home_team: str
    away_team: str
    is_conference_game: bool
    home_tz: str
    home_dma_rank: int
    market_score: float
    noon_eligible: bool
    is_rivalry_game: bool
    is_cfp_rematch: bool
    post_dst: bool
    neutral_site: bool
    source: str


@dataclass(frozen=True, slots=True)
class TeamPrestige:
    team: str
    championship_odds_normalized: float
    sp_plus_2025: float
    sp_plus_normalized: float
    prestige_score: float
    dma_rank: int
    market_score: float
    timezone: str
    noon_eligible: bool
    new_coach_2026: bool


@dataclass(frozen=True, slots=True)
class ViewershipRecord:
    season: int
    week: Optional[int]
    team_a: str
    team_b: str
    viewers_millions: float
    network: str
    time_slot: str
    home_team: Optional[str]
    away_team: Optional[str]
    home_points: Optional[int]
    away_points: Optional[int]
    home_pregame_elo: Optional[int]
    away_pregame_elo: Optional[int]
    is_conference_game: bool
    is_bowl_game: bool
    is_playoff_game: bool
    source: str


# ===========================================================================
# Validation helpers
# ===========================================================================

_GAME_REQUIRED = frozenset(Game.__dataclass_fields__.keys())
_PRESTIGE_REQUIRED = frozenset(TeamPrestige.__dataclass_fields__.keys())
_VIEWERSHIP_REQUIRED = frozenset(ViewershipRecord.__dataclass_fields__.keys())


def _validate_game(raw: dict, idx: int) -> Game:
    missing = _GAME_REQUIRED - raw.keys()
    if missing:
        raise ValueError(f"Game record {idx} missing fields: {missing}")
    if not raw["game_id"]:
        raise ValueError(f"Game record {idx} has empty game_id")
    if not isinstance(raw["week"], int) or raw["week"] < 0:
        raise ValueError(f"Game record {idx} has invalid week: {raw['week']}")
    if not raw["home_team"] or not raw["away_team"]:
        raise ValueError(f"Game record {idx} has empty team name")
    if raw["home_dma_rank"] < 1 or raw["home_dma_rank"] > 220:
        raise ValueError(f"Game record {idx} has invalid home_dma_rank: {raw['home_dma_rank']}")
    if raw["market_score"] < 0:
        raise ValueError(f"Game record {idx} has negative market_score")
    return Game(**{k: raw[k] for k in Game.__dataclass_fields__})


def _validate_prestige(raw: dict, idx: int) -> TeamPrestige:
    missing = _PRESTIGE_REQUIRED - raw.keys()
    if missing:
        raise ValueError(f"Prestige record {idx} missing fields: {missing}")
    if not raw["team"]:
        raise ValueError(f"Prestige record {idx} has empty team name")
    if not (0.0 <= raw["prestige_score"] <= 1.0):
        raise ValueError(f"Prestige record {idx} has prestige_score out of [0,1]: {raw['prestige_score']}")
    if not (0.0 <= raw["championship_odds_normalized"] <= 1.0):
        raise ValueError(f"Prestige record {idx} has championship_odds_normalized out of [0,1]")
    if not (0.0 <= raw["sp_plus_normalized"] <= 1.0):
        raise ValueError(f"Prestige record {idx} has sp_plus_normalized out of [0,1]")
    if raw["dma_rank"] < 1 or raw["dma_rank"] > 220:
        raise ValueError(f"Prestige record {idx} has invalid dma_rank: {raw['dma_rank']}")
    return TeamPrestige(**{k: raw[k] for k in TeamPrestige.__dataclass_fields__})


def _validate_viewership(raw: dict, idx: int) -> ViewershipRecord:
    missing = _VIEWERSHIP_REQUIRED - raw.keys()
    if missing:
        raise ValueError(f"Viewership record {idx} missing fields: {missing}")
    if raw["season"] < 2000 or raw["season"] > 2100:
        raise ValueError(f"Viewership record {idx} has invalid season: {raw['season']}")
    if not raw["team_a"] or not raw["team_b"]:
        raise ValueError(f"Viewership record {idx} has empty team name")
    if raw["viewers_millions"] <= 0:
        raise ValueError(f"Viewership record {idx} has non-positive viewers: {raw['viewers_millions']}")
    if raw["time_slot"] not in ("noon", "afternoon", "primetime", "unknown"):
        raise ValueError(f"Viewership record {idx} has invalid time_slot: {raw['time_slot']}")
    return ViewershipRecord(**{k: raw[k] for k in ViewershipRecord.__dataclass_fields__})


# ===========================================================================
# Public loaders
# ===========================================================================

def load_schedule(path: Path | None = None) -> list[Game]:
    """Load and validate game_schedule.json. Returns list of Game dataclasses."""
    path = path or PROCESSED_DIR / "game_schedule.json"
    with open(path) as f:
        raw_list = json.load(f)
    if not isinstance(raw_list, list) or len(raw_list) == 0:
        raise ValueError(f"Schedule file {path} is empty or not a list")
    return [_validate_game(r, i) for i, r in enumerate(raw_list)]


def load_prestige(path: Path | None = None) -> list[TeamPrestige]:
    """Load and validate team_prestige.json. Returns list of TeamPrestige dataclasses."""
    path = path or PROCESSED_DIR / "team_prestige.json"
    with open(path) as f:
        raw_list = json.load(f)
    if not isinstance(raw_list, list) or len(raw_list) == 0:
        raise ValueError(f"Prestige file {path} is empty or not a list")
    return [_validate_prestige(r, i) for i, r in enumerate(raw_list)]


def load_viewership(path: Path | None = None) -> list[ViewershipRecord]:
    """Load and validate viewership_pairs.json. Returns list of ViewershipRecord dataclasses."""
    path = path or PROCESSED_DIR / "viewership_pairs.json"
    with open(path) as f:
        raw_list = json.load(f)
    if not isinstance(raw_list, list) or len(raw_list) == 0:
        raise ValueError(f"Viewership file {path} is empty or not a list")
    return [_validate_viewership(r, i) for i, r in enumerate(raw_list)]


# ===========================================================================
# Convenience dict-based lookups
# ===========================================================================

def load_prestige_lookup(path: Path | None = None) -> dict[str, TeamPrestige]:
    """Load prestige as a dict keyed by team name."""
    return {p.team: p for p in load_prestige(path)}

"""
src/acquisition/cfbd_client.py
──────────────────────────────
Thin wrapper around the cfbd SDK.
Handles auth, caching raw responses to data/raw/schedule/, and returning
normalized dicts that map directly to the game_schedule JSON schema.
"""

from __future__ import annotations

import json
import math
import logging
from datetime import datetime
from pathlib import Path

import certifi
import cfbd
import time
import urllib3

from config.settings import settings
from config.constants import BIG_TEN_TEAMS, TEAM_META, RIVALRY_PAIRS, CFP_REMATCH_PAIRS, PACIFIC_TZ_TEAMS

logger = logging.getLogger(__name__)

_DST_END = datetime.fromisoformat(settings.DST_END)


def _build_api_client() -> cfbd.ApiClient:
    config = cfbd.Configuration()
    config.access_token = settings.CFBD_API_KEY
    config.ssl_ca_cert = certifi.where()
    return cfbd.ApiClient(config)


def fetch_schedule(force_refresh: bool = False) -> list[dict]:
    """
    Pull the full 2026 Big Ten schedule from CFBD.
    Results are cached to data/raw/schedule/cfbd_2026_raw.json.
    Set force_refresh=True to re-hit the API.

    Returns a list of normalized game dicts matching game_schedule schema.
    """
    cache_path = settings.RAW_DIR / "schedule" / "cfbd_2026_raw.json"

    if cache_path.exists() and not force_refresh:
        logger.info("Loading schedule from cache: %s", cache_path)
        with open(cache_path) as f:
            raw_games = json.load(f)
    else:
        logger.info("Fetching schedule from CFBD API...")
        client = _build_api_client()
        games_api = cfbd.GamesApi(client)

        # One call for all conference games
        conf_games = games_api.get_games(
            year=settings.SEASON, conference=settings.CONFERENCE, season_type="regular"
        )

        # Per-team calls for non-conference games
        teams_api = cfbd.TeamsApi(client)
        b1g_teams = [t.school for t in teams_api.get_teams(conference=settings.CONFERENCE)]

        all_games, seen_ids = list(conf_games), {g.id for g in conf_games}
        for team in b1g_teams:
            for g in games_api.get_games(year=settings.SEASON, team=team, season_type="regular"):
                if g.id not in seen_ids:
                    all_games.append(g)
                    seen_ids.add(g.id)

        # Serialize raw SDK objects as dicts for caching
        raw_games = [g.to_dict() for g in all_games]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(raw_games, f, indent=2, default=str)
        logger.info("Cached %d games to %s", len(raw_games), cache_path)

    return [_normalize_game(g) for g in raw_games]


def fetch_postseason_lookup(
    years: list[int] = [2023, 2024, 2025],
    force_refresh: bool = False,
) -> dict[tuple, dict]:
    """
    Returns a lookup keyed by (year, frozenset({home_team, away_team})) for every
    postseason game involving at least one B1G team.  Each value has:
        is_bowl_game, is_playoff_game, notes
    Cache per year: data/raw/schedule/cfbd_{year}_postseason.json
    """
    lookup: dict[tuple, dict] = {}
    client = None

    for year in years:
        cache_path = settings.RAW_DIR / "schedule" / f"cfbd_{year}_postseason.json"
        if cache_path.exists() and not force_refresh:
            logger.info("Loading %d postseason data from cache: %s", year, cache_path)
            with open(cache_path) as f:
                raw_games = json.load(f)
        else:
            logger.info("Fetching %d postseason data from CFBD API...", year)
            if client is None:
                client = _build_api_client()
            api = cfbd.GamesApi(client)
            games_by_id: dict[int, dict] = {}
            for team in BIG_TEN_TEAMS:
                try:
                    for g in api.get_games(year=year, team=team, season_type="postseason"):
                        games_by_id[g.id] = g.to_dict()
                    time.sleep(0.15)  # stay well under rate limit
                except Exception as exc:
                    logger.warning("CFBD postseason fetch failed for %s %d: %s", team, year, exc)
                    time.sleep(1.0)
            raw_games = list(games_by_id.values())
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(raw_games, f, indent=2, default=str)
            logger.info("Cached %d postseason games for %d to %s", len(raw_games), year, cache_path)

        for g in raw_games:
            home  = g.get("homeTeam", "") or ""
            away  = g.get("awayTeam", "") or ""
            notes = g.get("notes", "") or ""
            is_playoff = "College Football Playoff" in notes
            is_bowl    = bool(notes) and not is_playoff
            lookup[(year, frozenset({home, away}))] = {
                "is_bowl_game":    is_bowl,
                "is_playoff_game": is_playoff,
                "notes":           notes,
            }

    return lookup


def fetch_historical_games(
    years: list[int] = [2023, 2024, 2025],
    force_refresh: bool = False,
) -> dict[tuple, dict]:
    """
    Fetch all games involving at least one B1G team for given years.
    Returns a lookup keyed by (year, frozenset({team_a, team_b})) with:
        week, start_date, start_hour_et, home_team, away_team,
        home_points, away_points, home_pregame_elo, away_pregame_elo,
        attendance, venue, neutral_site, conference_game
    Cache per year: data/raw/schedule/cfbd_{year}_all_games.json
    """
    from datetime import datetime, timedelta, timezone

    ET_OFFSET = timedelta(hours=-5)  # approximate; sufficient for slot bucketing

    lookup: dict[tuple, dict] = {}
    client = None

    for year in years:
        cache_path = settings.RAW_DIR / "schedule" / f"cfbd_{year}_all_games.json"
        if cache_path.exists() and not force_refresh:
            logger.info("Loading %d historical games from cache: %s", year, cache_path)
            with open(cache_path) as f:
                raw_games = json.load(f)
        else:
            logger.info("Fetching %d historical games from CFBD API...", year)
            if client is None:
                client = _build_api_client()
            api = cfbd.GamesApi(client)
            games_by_id: dict[int, dict] = {}

            # Conference games in one call
            try:
                for g in api.get_games(year=year, conference="B1G", season_type="both"):
                    games_by_id[g.id] = g.to_dict()
            except Exception as exc:
                logger.warning("CFBD conference fetch failed for %d: %s", year, exc)

            time.sleep(0.3)

            # Per-team calls for non-conference games
            for team in BIG_TEN_TEAMS:
                try:
                    for g in api.get_games(year=year, team=team, season_type="both"):
                        if g.id not in games_by_id:
                            games_by_id[g.id] = g.to_dict()
                    time.sleep(0.15)
                except Exception as exc:
                    logger.warning("CFBD fetch failed for %s %d: %s", team, year, exc)
                    time.sleep(1.0)

            raw_games = list(games_by_id.values())
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(raw_games, f, indent=2, default=str)
            logger.info("Cached %d games for %d to %s", len(raw_games), year, cache_path)

        for g in raw_games:
            home = g.get("homeTeam", "") or ""
            away = g.get("awayTeam", "") or ""
            start_raw = g.get("startDate", "") or ""

            # Parse start time to ET hour for slot bucketing
            start_hour_et = None
            if start_raw:
                try:
                    dt = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
                    dt_et = dt.astimezone(timezone(ET_OFFSET))
                    start_hour_et = dt_et.hour
                except (ValueError, TypeError):
                    pass

            key = (year, frozenset({home, away}))
            # If multiple games between same teams (rare), keep later one
            lookup[key] = {
                "week":              g.get("week"),
                "start_date":        str(start_raw)[:10] if start_raw else None,
                "start_hour_et":     start_hour_et,
                "home_team":         home,
                "away_team":         away,
                "home_points":       g.get("homePoints"),
                "away_points":       g.get("awayPoints"),
                "home_pregame_elo":  g.get("homePregameElo"),
                "away_pregame_elo":  g.get("awayPregameElo"),
                "attendance":        g.get("attendance"),
                "venue":             g.get("venue"),
                "neutral_site":      g.get("neutralSite", False),
                "conference_game":   g.get("conferenceGame", False),
            }

    return lookup


def fetch_fbs_ratings(year: int = 2025, force_refresh: bool = False) -> dict:
    """
    Fetch SP+ and Elo ratings for all FBS teams from CFBD.

    Returns dict with:
        sp_plus: {team_name: rating, ...}
        elo: {team_name: elo_value, ...}

    Caches to:
        data/raw/ratings/cfbd_{year}_sp_plus.json
        data/raw/ratings/cfbd_{year}_elo.json
    """
    ratings_dir = settings.RAW_DIR / "ratings"
    ratings_dir.mkdir(parents=True, exist_ok=True)

    sp_path = ratings_dir / f"cfbd_{year}_sp_plus.json"
    elo_path = ratings_dir / f"cfbd_{year}_elo.json"

    # ── SP+ ───────────────────────────────────────────────────────────────────
    if sp_path.exists() and not force_refresh:
        logger.info("Loading SP+ from cache: %s", sp_path)
        with open(sp_path) as f:
            sp_list = json.load(f)
    else:
        logger.info("Fetching SP+ ratings from CFBD API (year=%d)...", year)
        client = _build_api_client()
        api = cfbd.RatingsApi(client)
        all_sp = api.get_sp(year=year)
        sp_list = [
            {"team": t.team, "conference": t.conference, "rating": t.rating}
            for t in all_sp if t.rating is not None
        ]
        sp_list.sort(key=lambda x: -(x["rating"] or 0))
        with open(sp_path, "w") as f:
            json.dump(sp_list, f, indent=2)
        logger.info("Cached %d SP+ ratings to %s", len(sp_list), sp_path)

    # ── Elo ───────────────────────────────────────────────────────────────────
    if elo_path.exists() and not force_refresh:
        logger.info("Loading Elo from cache: %s", elo_path)
        with open(elo_path) as f:
            elo_list = json.load(f)
    else:
        logger.info("Fetching Elo ratings from CFBD API (year=%d)...", year)
        client = _build_api_client()
        api = cfbd.RatingsApi(client)
        all_elo = api.get_elo(year=year)
        elo_list = [
            {"team": e.team, "conference": e.conference, "elo": e.elo}
            for e in all_elo if e.elo is not None
        ]
        elo_list.sort(key=lambda x: -x["elo"])
        with open(elo_path, "w") as f:
            json.dump(elo_list, f, indent=2)
        logger.info("Cached %d Elo ratings to %s", len(elo_list), elo_path)

    sp_lookup = {t["team"]: t["rating"] for t in sp_list}
    elo_lookup = {t["team"]: t["elo"] for t in elo_list}

    return {"sp_plus": sp_lookup, "elo": elo_lookup}


def _normalize_game(g: dict) -> dict:
    """Transform a raw CFBD 5.x game dict into the project's game_schedule schema."""
    home = g.get("homeTeam", "")
    away = g.get("awayTeam", "")
    home_meta = TEAM_META.get(home, {"dma_rank": 200, "tz": "ET"})
    raw_date = g.get("startDate", "")
    game_date = datetime.fromisoformat(str(raw_date)[:10]) if raw_date else None

    return {
        "game_id":           str(g.get("id", "")),
        "week":              g.get("week"),
        "game_date":         str(raw_date)[:10] if raw_date else None,
        "home_team":         home,
        "away_team":         away,
        "is_conference_game": home in BIG_TEN_TEAMS and away in BIG_TEN_TEAMS,
        "home_tz":           home_meta.get("tz", "ET"),
        "home_dma_rank":     home_meta.get("dma_rank", 200),
        "market_score":      round(math.log(1 + (211 - home_meta.get("dma_rank", 200))), 4),
        "noon_eligible":     home not in PACIFIC_TZ_TEAMS,
        "is_rivalry_game":   frozenset({home, away}) in RIVALRY_PAIRS,
        "is_cfp_rematch":    frozenset({home, away}) in CFP_REMATCH_PAIRS,
        "post_dst":          game_date > _DST_END if game_date else False,
        "neutral_site":      g.get("neutralSite", False),
        "source":            "cfbd_api",
    }

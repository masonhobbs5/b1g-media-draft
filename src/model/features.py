"""
src/model/features.py
─────────────────────
Derived features from historical viewership data:
  - Team brand ratings (avg audience when a team appears)
  - Week-of-season multipliers
  - Time-slot multipliers
  - Full feature table for viewership model training and 2026 scoring

All computations use the enriched viewership_pairs.json produced by the
acquisition pipeline (which includes week, time_slot, scores, Elo, etc.).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from config.constants import BIG_TEN_TEAMS, RIVALRY_PAIRS, CFP_REMATCH_PAIRS
from config.settings import settings
from src.utils.data_loaders import load_viewership, ViewershipRecord

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = settings.PROCESSED_DIR


# ===========================================================================
# Team brand ratings
# ===========================================================================

def compute_team_brand_ratings(
    records: list[ViewershipRecord] | None = None,
) -> dict[str, dict]:
    """
    Compute average viewership when each B1G team appears (home or away).
    Returns dict keyed by team name with:
        avg_viewers_millions, appearances, total_viewers_millions
    """
    if records is None:
        records = load_viewership()

    team_viewers: dict[str, list[float]] = defaultdict(list)

    for r in records:
        # Skip bowl/playoff (outlier viewership distorts brand signal)
        if r.is_bowl_game or r.is_playoff_game:
            continue
        if r.team_a in BIG_TEN_TEAMS:
            team_viewers[r.team_a].append(r.viewers_millions)
        if r.team_b in BIG_TEN_TEAMS:
            team_viewers[r.team_b].append(r.viewers_millions)

    brands: dict[str, dict] = {}
    for team in sorted(BIG_TEN_TEAMS):
        viewers = team_viewers.get(team, [])
        if viewers:
            brands[team] = {
                "team": team,
                "avg_viewers_millions": round(sum(viewers) / len(viewers), 3),
                "appearances": len(viewers),
                "total_viewers_millions": round(sum(viewers), 3),
            }
        else:
            brands[team] = {
                "team": team,
                "avg_viewers_millions": 0.0,
                "appearances": 0,
                "total_viewers_millions": 0.0,
            }

    return brands


# ===========================================================================
# Week-of-season multipliers
# ===========================================================================

def compute_week_multipliers(
    records: list[ViewershipRecord] | None = None,
) -> dict[int, dict]:
    """
    Compute average viewership by week number.
    Returns dict keyed by week with:
        avg_viewers_millions, game_count, multiplier (relative to overall avg)
    """
    if records is None:
        records = load_viewership()

    week_viewers: dict[int, list[float]] = defaultdict(list)
    all_viewers: list[float] = []

    for r in records:
        if r.week is None or r.is_bowl_game or r.is_playoff_game:
            continue
        week_viewers[r.week].append(r.viewers_millions)
        all_viewers.append(r.viewers_millions)

    if not all_viewers:
        return {}

    overall_avg = sum(all_viewers) / len(all_viewers)

    multipliers: dict[int, dict] = {}
    for week in sorted(week_viewers.keys()):
        viewers = week_viewers[week]
        week_avg = sum(viewers) / len(viewers)
        multipliers[week] = {
            "week": week,
            "avg_viewers_millions": round(week_avg, 3),
            "game_count": len(viewers),
            "multiplier": round(week_avg / overall_avg, 3),
        }

    return multipliers


# ===========================================================================
# Time-slot multipliers
# ===========================================================================

def compute_slot_multipliers(
    records: list[ViewershipRecord] | None = None,
) -> dict[str, dict]:
    """
    Compute average viewership by broadcast window (noon, afternoon, primetime).
    Returns dict keyed by slot with:
        avg_viewers_millions, game_count, multiplier (relative to overall avg)
    """
    if records is None:
        records = load_viewership()

    slot_viewers: dict[str, list[float]] = defaultdict(list)
    all_viewers: list[float] = []

    for r in records:
        if r.time_slot == "unknown" or r.is_bowl_game or r.is_playoff_game:
            continue
        slot_viewers[r.time_slot].append(r.viewers_millions)
        all_viewers.append(r.viewers_millions)

    if not all_viewers:
        return {}

    overall_avg = sum(all_viewers) / len(all_viewers)

    multipliers: dict[str, dict] = {}
    for slot in ("noon", "afternoon", "primetime"):
        viewers = slot_viewers.get(slot, [])
        if viewers:
            slot_avg = sum(viewers) / len(viewers)
            multipliers[slot] = {
                "slot": slot,
                "avg_viewers_millions": round(slot_avg, 3),
                "game_count": len(viewers),
                "multiplier": round(slot_avg / overall_avg, 3),
            }

    return multipliers


# ===========================================================================
# Persist all derived features
# ===========================================================================

def build_derived_features(
    records: list[ViewershipRecord] | None = None,
) -> dict[str, dict]:
    """Compute and persist all derived features to data/processed/."""
    if records is None:
        records = load_viewership()

    brands = compute_team_brand_ratings(records)
    week_mults = compute_week_multipliers(records)
    slot_mults = compute_slot_multipliers(records)

    output = {
        "team_brands": brands,
        "week_multipliers": week_mults,
        "slot_multipliers": slot_mults,
    }

    out_path = PROCESSED_DIR / "viewership_features.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote derived viewership features to %s", out_path)

    return output


# ===========================================================================
# Feature table for viewership model
# ===========================================================================

def _is_rivalry(team_a: str, team_b: str) -> bool:
    """Check if a matchup is a rivalry game."""
    pair = frozenset({team_a, team_b})
    return pair in RIVALRY_PAIRS


def _is_cfp_rematch(team_a: str, team_b: str) -> bool:
    """Check if a matchup is a CFP rematch."""
    pair = frozenset({team_a, team_b})
    return pair in CFP_REMATCH_PAIRS


def _load_features_json() -> dict:
    """Load precomputed viewership_features.json."""
    path = PROCESSED_DIR / "viewership_features.json"
    if not path.exists():
        return {"team_brands": {}, "week_multipliers": {}, "slot_multipliers": {}}
    with open(path) as f:
        return json.load(f)


def build_training_features(
    records: list[ViewershipRecord] | None = None,
) -> list[dict]:
    """
    Build a feature table from historical viewership records for model training.

    Each row represents one historical game with features and the target variable
    (viewers_millions). Only regular-season games with known week and time_slot
    are included.

    Returns list of dicts with feature columns and target.
    """
    if records is None:
        records = load_viewership()

    features_data = _load_features_json()
    brands = features_data.get("team_brands", {})
    week_mults = features_data.get("week_multipliers", {})
    slot_mults = features_data.get("slot_multipliers", {})

    rows = []
    for r in records:
        # Skip bowl/playoff games and those missing key fields
        if r.is_bowl_game or r.is_playoff_game:
            continue
        if r.week is None or r.time_slot == "unknown":
            continue

        team_a = r.team_a
        team_b = r.team_b

        # Team brand ratings (millions)
        brand_a = brands.get(team_a, {}).get("avg_viewers_millions", 1.5)
        brand_b = brands.get(team_b, {}).get("avg_viewers_millions", 1.5)
        # Use average of brand for non-B1G teams (opponent brand signal)
        combined_brand = brand_a + brand_b
        max_brand = max(brand_a, brand_b)

        # Elo-based strength features
        elo_a = r.home_pregame_elo or 1500
        elo_b = r.away_pregame_elo or 1500
        combined_elo = elo_a + elo_b
        elo_diff = abs(elo_a - elo_b)  # closeness signal
        avg_elo = combined_elo / 2.0

        # Week and slot multipliers
        wk_mult = week_mults.get(str(r.week), {}).get("multiplier", 1.0)
        sl_mult = slot_mults.get(r.time_slot, {}).get("multiplier", 1.0)

        # Game context flags
        is_rivalry = _is_rivalry(team_a, team_b)
        is_conf = r.is_conference_game

        # Network tier (broadcast reach signal)
        net_tier = NETWORK_TIERS.get(r.network, 0)

        # Late-season flag (weeks 10+ have conference championship implications)
        late_season = int((r.week or 0) >= 10)

        # Brand × Elo interaction (top brands with high Elo draw outsized audiences)
        brand_x_elo = combined_brand * avg_elo / 1500.0

        row = {
            "season": r.season,
            "week": r.week,
            "team_a": team_a,
            "team_b": team_b,
            "time_slot": r.time_slot,
            "network": r.network,
            # ── Features ──
            "combined_brand": combined_brand,
            "max_brand": max_brand,
            "combined_elo": combined_elo,
            "elo_closeness": 1.0 / (1.0 + elo_diff / 200.0),  # 0-1, higher = closer
            "avg_elo": avg_elo,
            "week_multiplier": wk_mult,
            "slot_multiplier": sl_mult,
            "is_rivalry": int(is_rivalry),
            "is_conference_game": int(is_conf),
            "network_tier": net_tier,
            "late_season": late_season,
            "brand_x_elo": brand_x_elo,
            # ── Target ──
            "viewers_millions": r.viewers_millions,
        }
        rows.append(row)

    logger.info("Built training feature table: %d rows", len(rows))
    return rows


def build_2026_features(
    schedule: list[dict] | None = None,
    strength_lookup: dict[str, float] | None = None,
) -> list[dict]:
    """
    Build feature vectors for each 2026 game for viewership prediction.

    Uses same feature schema as training data but derives values from
    the 2026 schedule and team strength model instead of historical Elo.

    Args:
        schedule: 2026 game schedule (auto-loaded if None)
        strength_lookup: team → composite_score (auto-loaded if None)

    Returns list of dicts with features for each 2026 game.
    """
    if schedule is None:
        with open(PROCESSED_DIR / "game_schedule.json") as f:
            schedule = json.load(f)

    if strength_lookup is None:
        fbs_path = PROCESSED_DIR / "fbs_strength.json"
        if fbs_path.exists():
            with open(fbs_path) as f:
                teams = json.load(f)
            strength_lookup = {t["team"]: t["composite_score"] for t in teams}
        else:
            strength_lookup = {}

    # Load FBS Elo for Elo-based features
    fbs_path = PROCESSED_DIR / "fbs_strength.json"
    elo_lookup: dict[str, int] = {}
    if fbs_path.exists():
        with open(fbs_path) as f:
            teams = json.load(f)
        elo_lookup = {t["team"]: t.get("elo_2025", 1500) for t in teams}

    features_data = _load_features_json()
    brands = features_data.get("team_brands", {})
    week_mults = features_data.get("week_multipliers", {})
    slot_mults = features_data.get("slot_multipliers", {})

    # Default slot multiplier (use noon as default for unscheduled games)
    default_slot_mult = slot_mults.get("noon", {}).get("multiplier", 1.0)

    rows = []
    for game in schedule:
        home = game["home_team"]
        away = game["away_team"]
        week = game["week"]

        # Team brand ratings
        brand_home = brands.get(home, {}).get("avg_viewers_millions", 1.5)
        brand_away = brands.get(away, {}).get("avg_viewers_millions", 1.5)
        combined_brand = brand_home + brand_away
        max_brand = max(brand_home, brand_away)

        # Elo-based features (from 2025 end-of-season Elo as proxy)
        elo_home = elo_lookup.get(home, 1500)
        elo_away = elo_lookup.get(away, 1500)
        combined_elo = elo_home + elo_away
        elo_diff = abs(elo_home - elo_away)
        avg_elo = combined_elo / 2.0

        # Week and slot multipliers
        wk_mult = week_mults.get(str(week), {}).get("multiplier", 1.0)
        # 2026 games don't have assigned time slots yet; use default
        sl_mult = default_slot_mult

        # Game context flags
        is_rivalry = _is_rivalry(home, away)
        is_cfp_rematch = _is_cfp_rematch(home, away)
        is_conf = game.get("is_conference_game", False)

        # Network tier: 2026 games are predicted for the draft pool (FOX/CBS/NBC)
        # so we assume major-network broadcast reach (tier 3).
        net_tier = 3

        # Late-season flag
        late_season = int(week >= 10)

        # Brand × Elo interaction
        brand_x_elo = combined_brand * avg_elo / 1500.0

        row = {
            "game_id": game["game_id"],
            "week": week,
            "game_date": game.get("game_date"),
            "home_team": home,
            "away_team": away,
            # ── Features (same schema as training) ──
            "combined_brand": combined_brand,
            "max_brand": max_brand,
            "combined_elo": combined_elo,
            "elo_closeness": 1.0 / (1.0 + elo_diff / 200.0),
            "avg_elo": avg_elo,
            "week_multiplier": wk_mult,
            "slot_multiplier": sl_mult,
            "is_rivalry": int(is_rivalry),
            "is_conference_game": int(is_conf),
            "network_tier": net_tier,
            "late_season": late_season,
            "brand_x_elo": brand_x_elo,
            # ── Extra context for downstream use ──
            "is_cfp_rematch": int(is_cfp_rematch),
            "home_strength": strength_lookup.get(home, 0.05),
            "away_strength": strength_lookup.get(away, 0.05),
            "neutral_site": game.get("neutral_site", False),
            "home_dma_rank": game.get("home_dma_rank", 100),
            "market_score": game.get("market_score", 0.0),
        }
        rows.append(row)

    logger.info("Built 2026 feature table: %d games", len(rows))
    return rows


# Network tier mapping: broadcast reach correlates with viewership
NETWORK_TIERS: dict[str, int] = {
    "FOX": 3, "CBS": 3, "NBC": 3,
    "ABC": 2, "ESPN": 2,
    "FS1": 1,
    "BTN": 0, "ESPNU": 0, "ESPN2": 0, "Peacock": 0, "CW": 0,
}

# Column names used as model inputs (shared between training and scoring)
FEATURE_COLUMNS = [
    "combined_brand",
    "max_brand",
    "combined_elo",
    "elo_closeness",
    "week_multiplier",
    "slot_multiplier",
    "is_rivalry",
    "is_conference_game",
    "network_tier",
    "late_season",
    "brand_x_elo",
]

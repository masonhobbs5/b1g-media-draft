"""
src/simulation/draft_value.py
─────────────────────────────
Broadcaster game value scoring: V(g) for the weekly draft.

Computes a composite score for each eligible game using:
  - Prestige (0.30): combined team strength
  - Viewership (0.25): predicted audience from model
  - Stakes (0.20): conference game, rivalry, CFP rematch signals
  - Market (0.10): DMA/market score of the home team
  - Window fit (0.10): how well the game fits each network's slot
  - Novelty (0.05): rivalry and rematch boost

Each component is normalized to [0, 1] within the weekly slate before
weighting, so games are ranked relative to their weekly competition.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.constants import BIG_TEN_TEAMS, PACIFIC_TZ_TEAMS
from config.settings import settings

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = settings.PROCESSED_DIR


def _normalize_values(values: list[float]) -> list[float]:
    """Normalize a list of values to [0, 1] using min-max scaling."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-9:
        return [0.5] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def compute_window_fit(game: dict, network: str) -> float:
    """
    Score how well a game fits a network's broadcast window.

    FOX (noon ET): Prefers ET/CT teams at home, penalizes PT home teams.
    CBS (3:30 ET): Good for all, slight preference for PT games.
    NBC (primetime 7:30 ET): Prefers premium matchups, no timezone penalty.

    Returns a score in [0, 1].
    """
    home_team = game["home_team"]
    home_tz = game.get("home_tz", "ET")
    noon_eligible = game.get("noon_eligible", True)

    if network == "FOX":
        # FOX broadcasts at noon ET
        if not noon_eligible or home_team in PACIFIC_TZ_TEAMS:
            return 0.0  # Cannot air PT home game at noon ET
        if home_tz == "ET":
            return 1.0
        if home_tz == "CT":
            return 0.9
        return 0.5  # Shouldn't happen but fallback

    elif network == "CBS":
        # CBS at 3:30 ET — works for all timezones
        if home_tz == "PT":
            return 1.0  # 12:30 local — ideal for PT
        if home_tz == "CT":
            return 0.9  # 2:30 local
        return 0.8  # ET 3:30 local — fine

    elif network == "NBC":
        # NBC primetime 7:30 ET — works for all
        if home_tz == "ET":
            return 1.0  # 7:30 local — prime
        if home_tz == "CT":
            return 0.9  # 6:30 local — still good
        if home_tz == "PT":
            return 0.7  # 4:30 local — early for "primetime" feel
        return 0.8

    return 0.5


def score_game(
    game: dict,
    viewership_lookup: dict[str, float],
    network: str = "FOX",
) -> dict:
    """
    Compute V(g) for a single game from a specific network's perspective.

    Args:
        game: Schedule dict with game metadata.
        viewership_lookup: game_id → predicted_viewers_millions.
        network: Which network is evaluating ("FOX", "CBS", "NBC").

    Returns:
        Dict with component scores and total V(g).
    """
    game_id = game["game_id"]

    # ── Component: Prestige (combined strength) ───────────────────────────────
    home_str = game.get("home_strength", 0.5)
    away_str = game.get("away_strength", 0.5)
    prestige_raw = home_str + away_str

    # ── Component: Viewership (predicted audience) ────────────────────────────
    viewership_raw = viewership_lookup.get(game_id, 2.0)

    # ── Component: Stakes (conference, rivalry, rematch) ──────────────────────
    stakes_raw = 0.0
    if game.get("is_conference_game", False):
        stakes_raw += 0.5
    if game.get("is_rivalry", False) or game.get("is_rivalry_game", False):
        stakes_raw += 0.3
    if game.get("is_cfp_rematch", False):
        stakes_raw += 0.2

    # ── Component: Market (DMA/market score) ──────────────────────────────────
    market_raw = game.get("market_score", 3.0)

    # ── Component: Window fit ─────────────────────────────────────────────────
    window_fit_raw = compute_window_fit(game, network)

    # ── Component: Novelty (rivalry + rematch) ────────────────────────────────
    novelty_raw = 0.0
    if game.get("is_rivalry", False) or game.get("is_rivalry_game", False):
        novelty_raw += 0.6
    if game.get("is_cfp_rematch", False):
        novelty_raw += 0.4

    return {
        "game_id": game_id,
        "prestige_raw": prestige_raw,
        "viewership_raw": viewership_raw,
        "stakes_raw": stakes_raw,
        "market_raw": market_raw,
        "window_fit_raw": window_fit_raw,
        "novelty_raw": novelty_raw,
    }


def score_weekly_slate(
    games: list[dict],
    viewership_lookup: dict[str, float],
    network: str = "FOX",
) -> list[dict]:
    """
    Score and rank a weekly slate of games for a specific network.

    Normalizes component scores within the slate and applies V(g) weights.

    Args:
        games: List of schedule game dicts for one week (B1G home games only).
        viewership_lookup: game_id → predicted_viewers_millions.
        network: Which network is evaluating.

    Returns:
        List of game dicts with raw/normalized scores and final value,
        sorted descending by total_value.
    """
    if not games:
        return []

    # Score each game
    raw_scores = [score_game(g, viewership_lookup, network) for g in games]

    # Normalize each component within the slate
    prestige_norms = _normalize_values([s["prestige_raw"] for s in raw_scores])
    viewership_norms = _normalize_values([s["viewership_raw"] for s in raw_scores])
    stakes_norms = _normalize_values([s["stakes_raw"] for s in raw_scores])
    market_norms = _normalize_values([s["market_raw"] for s in raw_scores])
    # Window fit is already 0-1, but re-normalize within slate for fairness
    window_norms = _normalize_values([s["window_fit_raw"] for s in raw_scores])
    novelty_norms = _normalize_values([s["novelty_raw"] for s in raw_scores])

    results = []
    for i, (game, raw) in enumerate(zip(games, raw_scores)):
        total_value = (
            settings.WEIGHT_PRESTIGE * prestige_norms[i]
            + settings.WEIGHT_VIEWERSHIP * viewership_norms[i]
            + settings.WEIGHT_STAKES * stakes_norms[i]
            + settings.WEIGHT_MARKET * market_norms[i]
            + settings.WEIGHT_WINDOW_FIT * window_norms[i]
            + settings.WEIGHT_NOVELTY * novelty_norms[i]
        )

        results.append({
            "game_id": raw["game_id"],
            "week": game["week"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "network": network,
            "total_value": round(total_value, 4),
            "prestige_norm": round(prestige_norms[i], 4),
            "viewership_norm": round(viewership_norms[i], 4),
            "stakes_norm": round(stakes_norms[i], 4),
            "market_norm": round(market_norms[i], 4),
            "window_fit_norm": round(window_norms[i], 4),
            "novelty_norm": round(novelty_norms[i], 4),
            "is_conference_game": game.get("is_conference_game", False),
            "noon_eligible": game.get("noon_eligible", True),
            "home_tz": game.get("home_tz", "ET"),
        })

    results.sort(key=lambda x: -x["total_value"])
    return results


def build_all_game_values() -> dict[int, dict[str, list[dict]]]:
    """
    Score all eligible games across all weeks for each network.

    Returns:
        Dict keyed by week, each containing a dict keyed by network
        with the scored/ranked slate.
    """
    # Load schedule
    with open(PROCESSED_DIR / "game_schedule.json") as f:
        schedule = json.load(f)

    # Load viewership predictions
    viewership_path = settings.OUTPUT_DIR / "expected_viewership.json"
    if viewership_path.exists():
        with open(viewership_path) as f:
            vw_list = json.load(f)
        viewership_lookup = {g["game_id"]: g["predicted_viewers_millions"] for g in vw_list}
    else:
        viewership_lookup = {}

    # Load FBS strength for prestige component
    fbs_path = PROCESSED_DIR / "fbs_strength.json"
    strength_lookup = {}
    if fbs_path.exists():
        with open(fbs_path) as f:
            teams = json.load(f)
        strength_lookup = {t["team"]: t["composite_score"] for t in teams}

    # Enrich schedule games with strength data
    for game in schedule:
        game["home_strength"] = strength_lookup.get(game["home_team"], 0.05)
        game["away_strength"] = strength_lookup.get(game["away_team"], 0.05)

    # Filter to eligible games:
    #   - Home team must be a Big Ten member
    #   - Neither team may be Notre Dame (NBC holds exclusive rights to all
    #     Notre Dame games; those are handled outside the weekly draft)
    eligible = [
        g for g in schedule
        if g["home_team"] in BIG_TEN_TEAMS
        and "Notre Dame" not in (g["home_team"], g["away_team"])
    ]

    # Group by week
    weeks: dict[int, list[dict]] = {}
    for g in eligible:
        weeks.setdefault(g["week"], []).append(g)

    # Score for each network
    all_values: dict[int, dict[str, list[dict]]] = {}
    for week in sorted(weeks.keys()):
        week_games = weeks[week]
        all_values[week] = {}
        for network in ("FOX", "CBS", "NBC"):
            scored = score_weekly_slate(week_games, viewership_lookup, network)
            all_values[week][network] = scored

    logger.info(
        "Scored %d weeks, %d total eligible games",
        len(all_values), len(eligible),
    )

    return all_values


def rank_weeks_by_top_value() -> list[dict]:
    """
    Rank weeks by the value of their best available game (from FOX perspective).

    Returns list of week summaries sorted by top game value descending.
    """
    all_values = build_all_game_values()
    week_rankings = []

    for week, network_slates in sorted(all_values.items()):
        fox_slate = network_slates.get("FOX", [])
        if not fox_slate:
            continue
        top_game = fox_slate[0]
        week_rankings.append({
            "week": week,
            "top_game": f"{top_game['home_team']} vs {top_game['away_team']}",
            "top_value": top_game["total_value"],
            "n_eligible_games": len(fox_slate),
        })

    week_rankings.sort(key=lambda x: -x["top_value"])
    return week_rankings

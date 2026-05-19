"""
src/model/win_probability.py
────────────────────────────
Converts team-strength ratings into game-level win probabilities.

Uses a Bradley-Terry / Elo-style formulation:
    P(home wins) = 1 / (1 + 10^((away_strength - home_strength - hfa) / scale))

Where:
    - strength values are composite_score from team_strength.json (0-1 scale)
    - hfa is the home-field advantage expressed on the same scale
    - scale controls the sensitivity (calibrated so that a 1.0 gap ≈ 99% prob)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = settings.PROCESSED_DIR

# ── Model parameters ──────────────────────────────────────────────────────────
# Scale factor: maps composite_score difference to win probability.
# With scale=0.6, a 0.3 difference ≈ 76% win prob (roughly 7-point spread).
SCALE = 0.6

# Clamp probabilities to avoid 0% or 100%
PROB_FLOOR = 0.01
PROB_CEIL = 0.99

# Default strength for FCS teams not in any rating database
FCS_DEFAULT_STRENGTH = 0.05


def _load_full_strength_lookup() -> dict[str, float]:
    """
    Load composite_score for all teams (B1G + non-B1G FBS).

    Prefers the unified fbs_strength.json. Falls back to team_strength.json
    (B1G only) if the FBS file doesn't exist yet.
    """
    fbs_path = PROCESSED_DIR / "fbs_strength.json"
    if fbs_path.exists():
        with open(fbs_path) as f:
            teams = json.load(f)
        return {t["team"]: t["composite_score"] for t in teams}

    # Fallback: B1G only
    ts_path = PROCESSED_DIR / "team_strength.json"
    if ts_path.exists():
        with open(ts_path) as f:
            teams = json.load(f)
        return {t["team"]: t["composite_score"] for t in teams}

    return {}


def _load_hfa_on_composite_scale() -> float:
    """
    Convert the calibrated home-field Elo advantage to composite_score scale.

    Uses the full FBS Elo range for the conversion since composite_score
    is now normalized across all FBS teams.
    """
    hfa_path = PROCESSED_DIR / "home_field_advantage.json"
    if not hfa_path.exists():
        return 0.10  # sensible default

    with open(hfa_path) as f:
        hfa_data = json.load(f)

    # Use the Elo advantage and convert to composite scale
    elo_adv = hfa_data.get("elo_advantage", 55.0)

    # Load FBS strength to get the actual Elo range across all teams
    fbs_path = PROCESSED_DIR / "fbs_strength.json"
    ts_path = PROCESSED_DIR / "team_strength.json"

    if fbs_path.exists():
        with open(fbs_path) as f:
            teams = json.load(f)
        elos = [t["elo_2025"] for t in teams if t.get("elo_2025")]
    elif ts_path.exists():
        with open(ts_path) as f:
            teams = json.load(f)
        elos = [t["elo_2025_raw"] for t in teams]
    else:
        return 0.10

    if elos:
        elo_range = max(elos) - min(elos)
        if elo_range > 0:
            return elo_adv / elo_range
    return 0.10


def win_probability(
    home_strength: float,
    away_strength: float,
    neutral_site: bool = False,
    hfa: float | None = None,
) -> float:
    """
    Compute P(home team wins) using Bradley-Terry formulation.

    Args:
        home_strength: composite_score for home team [0, 1]
        away_strength: composite_score for away team [0, 1]
        neutral_site: if True, no home-field advantage applied
        hfa: home-field advantage on composite scale (auto-loaded if None)

    Returns:
        Probability of home team winning, clamped to [PROB_FLOOR, PROB_CEIL]
    """
    if hfa is None:
        hfa = _load_hfa_on_composite_scale()

    advantage = hfa if not neutral_site else 0.0
    diff = away_strength - home_strength - advantage

    # Bradley-Terry: P = 1 / (1 + 10^(diff / scale))
    exponent = diff / SCALE
    # Guard against overflow
    if exponent > 10:
        prob = PROB_FLOOR
    elif exponent < -10:
        prob = PROB_CEIL
    else:
        prob = 1.0 / (1.0 + 10.0 ** exponent)

    return max(PROB_FLOOR, min(PROB_CEIL, prob))


def compute_all_game_probabilities(
    schedule: list[dict] | None = None,
    strength_lookup: dict[str, float] | None = None,
) -> list[dict]:
    """
    Compute win probabilities for every game on the 2026 schedule.

    Uses all-FBS strength ratings for non-B1G opponents when available,
    falling back to a low default only for FCS teams.

    Returns list of dicts with:
        game_id, week, home_team, away_team, home_win_prob, away_win_prob,
        home_strength, away_strength, neutral_site, is_conference_game
    """
    if schedule is None:
        with open(PROCESSED_DIR / "game_schedule.json") as f:
            schedule = json.load(f)

    if strength_lookup is None:
        strength_lookup = _load_full_strength_lookup()

    hfa = _load_hfa_on_composite_scale()
    results = []

    for game in schedule:
        home = game["home_team"]
        away = game["away_team"]
        # FCS teams without any data get a very low default
        home_str = strength_lookup.get(home, FCS_DEFAULT_STRENGTH)
        away_str = strength_lookup.get(away, FCS_DEFAULT_STRENGTH)

        home_wp = win_probability(
            home_str, away_str,
            neutral_site=game.get("neutral_site", False),
            hfa=hfa,
        )

        results.append({
            "game_id": game["game_id"],
            "week": game["week"],
            "home_team": home,
            "away_team": away,
            "home_win_prob": round(home_wp, 4),
            "away_win_prob": round(1.0 - home_wp, 4),
            "home_strength": round(home_str, 4),
            "away_strength": round(away_str, 4),
            "neutral_site": game.get("neutral_site", False),
            "is_conference_game": game.get("is_conference_game", False),
        })

    return results


def build_win_probabilities() -> list[dict]:
    """Full pipeline: compute and persist game win probabilities."""
    probs = compute_all_game_probabilities()

    out_path = PROCESSED_DIR / "win_probabilities.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(probs, f, indent=2)
    logger.info("Wrote win_probabilities.json (%d games)", len(probs))

    return probs

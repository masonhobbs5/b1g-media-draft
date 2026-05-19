"""
src/model/team_strength.py
──────────────────────────
Composite team-strength ratings for the 2026 Big Ten season.

Produces a single pregame strength rating per team by blending:
  - Championship odds (implied probability, normalized) — B1G only
  - 2025 SP+ (end-of-season rating, normalized)
  - 2025 end-of-season Elo (normalized)

Also produces all-FBS strength ratings for non-conference opponents
using SP+ and Elo (since odds aren't available for all teams).

Also provides home-field advantage calibration from historical results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.constants import BIG_TEN_TEAMS, TEAM_META, SP_PLUS_2026, CHAMPIONSHIP_ODDS_2026, POWER_4_CONFERENCES
from config.settings import settings

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = settings.PROCESSED_DIR

# ── Default blending weights (sum to 1.0) ─────────────────────────────────────
DEFAULT_WEIGHTS = {
    "odds": 0.40,
    "sp_plus": 0.35,
    "elo": 0.25,
}

# ── Home-field advantage (in Elo points, calibrated from historical data) ─────
# This gets overwritten by calibrate_home_field() but provides a sensible default
DEFAULT_HOME_FIELD_ELO = 55.0  # ~3.5 point spread equivalent

# ── New coach penalty (Elo points deducted) ───────────────────────────────────
NEW_COACH_PENALTY_ELO = 30.0  # ~2 point spread


def extract_elo_2025() -> dict[str, int]:
    """
    Extract end-of-2025 season Elo for each B1G team from cached game results.
    Uses the latest postgame Elo from cfbd_2025_all_games.json.
    """
    cache_path = settings.RAW_DIR / "schedule" / "cfbd_2025_all_games.json"
    if not cache_path.exists():
        logger.warning("No 2025 game cache found at %s, returning empty Elo", cache_path)
        return {}

    with open(cache_path) as f:
        games = json.load(f)

    team_elo: dict[str, int] = {}
    for g in sorted(games, key=lambda x: x.get("week", 0)):
        home = g.get("homeTeam", "")
        away = g.get("awayTeam", "")
        if home in BIG_TEN_TEAMS and g.get("homePostgameElo"):
            team_elo[home] = g["homePostgameElo"]
        if away in BIG_TEN_TEAMS and g.get("awayPostgameElo"):
            team_elo[away] = g["awayPostgameElo"]

    return team_elo


def calibrate_home_field(years: list[int] = [2023, 2024, 2025]) -> dict:
    """
    Compute generic home-field advantage from historical B1G game results.

    Returns dict with:
        home_win_pct, games_counted, home_wins, away_wins,
        elo_advantage (estimated Elo points for home team)
    """
    home_wins = 0
    away_wins = 0

    for year in years:
        cache_path = settings.RAW_DIR / "schedule" / f"cfbd_{year}_all_games.json"
        if not cache_path.exists():
            continue
        with open(cache_path) as f:
            games = json.load(f)

        for g in games:
            home = g.get("homeTeam", "")
            away = g.get("awayTeam", "")
            hp = g.get("homePoints")
            ap = g.get("awayPoints")
            neutral = g.get("neutralSite", False)

            # Only count games with at least one B1G team, non-neutral, completed
            if not (home in BIG_TEN_TEAMS or away in BIG_TEN_TEAMS):
                continue
            if neutral or hp is None or ap is None:
                continue
            if hp == ap:
                continue  # ties shouldn't exist but skip just in case

            if hp > ap:
                home_wins += 1
            else:
                away_wins += 1

    total = home_wins + away_wins
    if total == 0:
        return {
            "home_win_pct": 0.5,
            "games_counted": 0,
            "home_wins": 0,
            "away_wins": 0,
            "elo_advantage": DEFAULT_HOME_FIELD_ELO,
        }

    home_pct = home_wins / total
    # Convert win% to Elo advantage: Elo diff = 400 * log10(p / (1-p))
    import math
    if 0 < home_pct < 1:
        elo_adv = 400 * math.log10(home_pct / (1 - home_pct))
    else:
        elo_adv = DEFAULT_HOME_FIELD_ELO

    return {
        "home_win_pct": round(home_pct, 4),
        "games_counted": total,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "elo_advantage": round(elo_adv, 1),
    }


def _normalize_values(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a dict of values to [0, 1]."""
    if not values:
        return {}
    v_min = min(values.values())
    v_max = max(values.values())
    if v_max == v_min:
        return {k: 0.5 for k in values}
    return {k: (v - v_min) / (v_max - v_min) for k, v in values.items()}


def compute_composite_ratings(
    weights: dict[str, float] | None = None,
    apply_coach_penalty: bool = True,
) -> list[dict]:
    """
    Compute composite team strength for all B1G teams.

    Blends normalized odds, SP+, and Elo signals using configurable weights.
    Optionally applies a new-coach penalty.

    Returns list of dicts with:
        team, odds_normalized, sp_plus_normalized, elo_normalized,
        composite_raw, composite_score (after optional penalty, re-normalized to [0,1]),
        new_coach_2026, elo_2025_raw, sp_plus_2025_raw
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Load existing prestige data
    prestige_path = PROCESSED_DIR / "team_prestige.json"
    with open(prestige_path) as f:
        prestige_list = json.load(f)
    prestige_lookup = {p["team"]: p for p in prestige_list}

    # Get Elo data
    elo_raw = extract_elo_2025()

    # Normalize each signal
    odds_values = {t: prestige_lookup[t]["championship_odds_normalized"] for t in BIG_TEN_TEAMS if t in prestige_lookup}
    sp_values = {t: prestige_lookup[t]["sp_plus_2025"] for t in BIG_TEN_TEAMS if t in prestige_lookup}
    elo_values = {t: float(elo_raw.get(t, 1500)) for t in BIG_TEN_TEAMS}

    odds_norm = _normalize_values(odds_values)
    sp_norm = _normalize_values(sp_values)
    elo_norm = _normalize_values(elo_values)

    # Blend
    results = []
    for team in sorted(BIG_TEN_TEAMS):
        o = odds_norm.get(team, 0.5)
        s = sp_norm.get(team, 0.5)
        e = elo_norm.get(team, 0.5)
        composite = weights["odds"] * o + weights["sp_plus"] * s + weights["elo"] * e

        results.append({
            "team": team,
            "odds_normalized": round(o, 4),
            "sp_plus_normalized": round(s, 4),
            "elo_normalized": round(e, 4),
            "composite_raw": round(composite, 4),
            "new_coach_2026": TEAM_META[team].get("new_coach", False),
            "elo_2025_raw": elo_raw.get(team, 1500),
            "sp_plus_2025_raw": prestige_lookup.get(team, {}).get("sp_plus_2025", 0.0),
        })

    # Apply coach penalty (subtract from composite_raw before final normalization)
    if apply_coach_penalty:
        # Convert penalty from Elo scale to normalized scale
        elo_range = max(elo_values.values()) - min(elo_values.values())
        if elo_range > 0:
            penalty_norm = (NEW_COACH_PENALTY_ELO / elo_range) * weights["elo"]
        else:
            penalty_norm = 0.02
        for r in results:
            if r["new_coach_2026"]:
                r["composite_raw"] = round(r["composite_raw"] - penalty_norm, 4)

    # Final normalization to [0, 1]
    raw_values = {r["team"]: r["composite_raw"] for r in results}
    final_norm = _normalize_values(raw_values)
    for r in results:
        r["composite_score"] = round(final_norm[r["team"]], 4)

    return results


def build_team_strength(
    weights: dict[str, float] | None = None,
    apply_coach_penalty: bool = True,
) -> list[dict]:
    """
    Full pipeline: compute ratings, calibrate home-field, persist both.
    Returns the team strength list.
    """
    ratings = compute_composite_ratings(weights=weights, apply_coach_penalty=apply_coach_penalty)
    hfa = calibrate_home_field()

    # Persist team strength
    out_path = PROCESSED_DIR / "team_strength.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ratings, f, indent=2)
    logger.info("Wrote team_strength.json (%d teams)", len(ratings))

    # Persist home-field calibration
    hfa_path = PROCESSED_DIR / "home_field_advantage.json"
    with open(hfa_path, "w") as f:
        json.dump(hfa, f, indent=2)
    logger.info("Wrote home_field_advantage.json: %.1f%% home win rate, %.1f Elo pts",
                hfa["home_win_pct"] * 100, hfa["elo_advantage"])

    # Build and persist all-FBS strength
    fbs_ratings = compute_fbs_strength()

    return ratings


# ── FBS-wide strength (for non-conference opponents) ──────────────────────────

# Default strength for FCS teams not in SP+/Elo databases
FCS_DEFAULT_STRENGTH = 0.05

# Weights for the unified FBS model:
# 2026 SP+ is the strongest forward-looking signal (accounts for roster changes,
# coaching changes, and returning production).
# 2026 championship odds capture market consensus for top teams.
# 2025 Elo provides a backward-looking baseline.
FBS_WEIGHTS = {
    "sp_plus_2026": 0.50,   # 2026 preseason SP+ (most forward-looking)
    "odds_2026": 0.25,      # 2026-27 championship odds (market consensus)
    "elo_2025": 0.25,       # 2025 end-of-season Elo (recent baseline)
}

# Default odds for teams without explicit championship odds
DEFAULT_ODDS_POWER4 = 15000     # +15000 for unlisted Power 4 teams
DEFAULT_ODDS_NON_POWER4 = 100000  # +100000 for non-Power 4 teams


def _odds_to_implied_prob(american_odds: int) -> float:
    """Convert American moneyline odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def compute_fbs_strength(b1g_ratings: list[dict] | None = None) -> list[dict]:
    """
    Compute composite strength for ALL FBS teams on a unified scale.

    Uses three signals with forward-looking emphasis:
    1. 2026 preseason SP+ (50%) — accounts for transfers, coaching changes, returning production
    2. 2026-27 championship odds (25%) — market consensus on who can win it all
    3. 2025 end-of-season Elo (25%) — recent performance baseline

    For teams without explicit championship odds:
    - Power 4 teams get +15000 (0.66% implied)
    - Non-Power 4 teams get +100000 (0.10% implied)

    FCS teams not in the database get FCS_DEFAULT_STRENGTH.

    Returns list of dicts: {team, composite_score, sp_plus_2026, elo_2025, odds_2026, source}
    Persists to data/processed/fbs_strength.json
    """
    from src.acquisition.cfbd_client import fetch_fbs_ratings

    # Load 2025 Elo from CFBD (cached)
    fbs_data = fetch_fbs_ratings(year=2025)
    elo_lookup = fbs_data["elo"]       # {team: elo_value}

    # 2026 SP+ from constants (scraped from ESPN)
    sp26_lookup = dict(SP_PLUS_2026)

    # 2026 Championship odds from constants
    odds_lookup = dict(CHAMPIONSHIP_ODDS_2026)

    # Determine conference for each team (for default odds assignment)
    sp25_path = settings.RAW_DIR / "ratings" / "cfbd_2025_sp_plus.json"
    conference_lookup: dict[str, str] = {}
    if sp25_path.exists():
        with open(sp25_path) as f:
            sp25_data = json.load(f)
        for t in sp25_data:
            conference_lookup[t["team"]] = t.get("conference", "")

    # All FBS teams: union of 2026 SP+, 2025 Elo, and 2025 SP+ sources
    all_fbs_teams = sorted(set(sp26_lookup.keys()) | set(elo_lookup.keys()))

    # Assign championship odds (implied probability) to every team
    odds_implied: dict[str, float] = {}
    for team in all_fbs_teams:
        if team in odds_lookup:
            odds_implied[team] = _odds_to_implied_prob(odds_lookup[team])
        else:
            conf = conference_lookup.get(team, "")
            if conf in POWER_4_CONFERENCES:
                odds_implied[team] = _odds_to_implied_prob(DEFAULT_ODDS_POWER4)
            else:
                odds_implied[team] = _odds_to_implied_prob(DEFAULT_ODDS_NON_POWER4)

    # Normalize each signal across all FBS teams to [0, 1]
    sp26_all = {t: sp26_lookup[t] for t in all_fbs_teams if t in sp26_lookup}
    elo_all = {t: float(elo_lookup[t]) for t in all_fbs_teams if t in elo_lookup}
    odds_all = {t: odds_implied[t] for t in all_fbs_teams if t in odds_implied}

    sp26_norm = _normalize_values(sp26_all)
    elo_norm = _normalize_values(elo_all)
    odds_norm = _normalize_values(odds_all)

    # Compute weighted composite for every team
    all_results = []
    for team in all_fbs_teams:
        s = sp26_norm.get(team)
        e = elo_norm.get(team)
        o = odds_norm.get(team, 0.0)

        # Build composite based on available signals
        if s is not None and e is not None:
            base = (FBS_WEIGHTS["sp_plus_2026"] * s +
                    FBS_WEIGHTS["odds_2026"] * o +
                    FBS_WEIGHTS["elo_2025"] * e)
        elif s is not None:
            # No Elo — use SP+ and odds only, reweighted
            base = 0.67 * s + 0.33 * o
        elif e is not None:
            # No 2026 SP+ — use Elo and odds only
            base = 0.67 * e + 0.33 * o
        else:
            base = 0.0

        all_results.append({
            "team": team,
            "composite_raw": base,
            "sp_plus_2026": sp26_lookup.get(team),
            "elo_2025": elo_lookup.get(team),
            "odds_2026": odds_lookup.get(team),
            "source": "fbs_2026_model",
        })

    # Apply new-coach penalty for B1G teams
    elo_range = max(elo_all.values()) - min(elo_all.values()) if elo_all else 1000
    penalty_on_scale = (NEW_COACH_PENALTY_ELO / elo_range) * FBS_WEIGHTS["elo_2025"]
    for r in all_results:
        if r["team"] in TEAM_META and TEAM_META[r["team"]].get("new_coach", False):
            r["composite_raw"] -= penalty_on_scale

    # Final normalization to [0, 1] across ALL teams
    raw_values = {r["team"]: r["composite_raw"] for r in all_results}
    final_norm = _normalize_values(raw_values)
    for r in all_results:
        r["composite_score"] = round(final_norm[r["team"]], 4)

    # Remove intermediate field
    for r in all_results:
        del r["composite_raw"]

    all_results.sort(key=lambda x: -x["composite_score"])

    # Persist
    out_path = PROCESSED_DIR / "fbs_strength.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Wrote fbs_strength.json (%d teams)", len(all_results))

    return all_results

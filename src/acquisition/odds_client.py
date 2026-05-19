"""
src/acquisition/odds_client.py
──────────────────────────────
Team prestige scores derived from PRESTIGE_FALLBACK in config/constants.py.
Uses hand-encoded championship futures odds (CBS Sports Jan 2026).
Update PRESTIGE_FALLBACK in config/constants.py at Media Days each year.
"""

from __future__ import annotations

import logging

from config.constants import BIG_TEN_TEAMS, PRESTIGE_FALLBACK

logger = logging.getLogger(__name__)


def fetch_prestige_scores() -> dict[str, float]:
    """
    Returns a dict mapping Big Ten team name → normalized prestige score (0–1).
    Scores are derived from hand-encoded PRESTIGE_FALLBACK championship futures odds.
    Update PRESTIGE_FALLBACK in config/constants.py to refresh these values.
    """
    logger.info("Loading prestige scores from PRESTIGE_FALLBACK (constants.py).")
    return _normalize(PRESTIGE_FALLBACK)


def _american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    """Scale raw probabilities so the top Big Ten team = 1.0."""
    b1g_probs = {t: raw.get(t, 0.001) for t in BIG_TEN_TEAMS}
    max_p = max(b1g_probs.values()) or 1.0
    return {t: round(p / max_p, 4) for t, p in b1g_probs.items()}

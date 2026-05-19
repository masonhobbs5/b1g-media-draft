"""
config/settings.py
─────────────────
Single source of truth for all project settings.
Loaded from environment variables (set in .env locally, CI secrets in GitHub Actions).
Import this anywhere: `from config.settings import settings`
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


class Settings:
    # ── API Keys ──────────────────────────────────────────────────────────────
    CFBD_API_KEY:      str = os.environ.get("CFBD_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Paths ─────────────────────────────────────────────────────────────────
    ROOT_DIR: Path = _ROOT
    DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(_ROOT / "data")))
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CACHE_DIR: Path = DATA_DIR / "cache"
    OUTPUT_DIR: Path = Path(os.environ.get("OUTPUT_DIR", str(DATA_DIR / "outputs")))

    # ── Season constants ──────────────────────────────────────────────────────
    SEASON: int = 2026
    CONFERENCE: str = "B1G"
    CHAMPIONSHIP_DATE: str = "2026-12-05"
    DST_END: str = "2026-11-01"  # First Sunday of November — post-DST flag

    # ── Simulation defaults ───────────────────────────────────────────────────
    SIM_ITERATIONS: int = int(os.environ.get("SIM_ITERATIONS", 10_000))
    SIM_SEED: int | None = (
        int(os.environ.get("SIM_SEED")) if os.environ.get("SIM_SEED") else 42
    )
    SIM_WORKERS: int = int(os.environ.get("SIM_WORKERS", 4))

    # ── Network draft structure (hard-coded per contract) ─────────────────────
    # Pick order: [1=Fox, 2=Fox, 3=Fox, 4=CBS, 5=Fox, 6=NBC, 7=CBS, 8=Fox, 9=NBC, 10-NBC, 11=CBS, 12=Fox, 13=NBC]
    # CBS holds pick 4 in 2026 (alternates with NBC each year; NBC had it in 2025)
    DRAFT_ORDER: list[str] = ["FOX","FOX","FOX","CBS","FOX","NBC","CBS","FOX","NBC","NBC","CBS","FOX","NBC"]
    TRADE_PROBABILITY: float = 0.15   # P(Fox trades a pick) per pick — calibrate from 2023-25
    SOFTMAX_TEMPERATURE: float = 0.3  # Controls randomness of within-week game selection

    # ── NBC pre-committed Notre Dame primetime weeks ──────────────────────────
    # NBC has contractually slated Notre Dame as its primetime game in these
    # weeks. NBC will not spend a draft pick in these weeks.
    NBC_NOTRE_DAME_WEEKS: list[int] = [1, 3, 10, 12]

    # ── V(g) scoring weights (must sum to 1.0) ────────────────────────────────
    WEIGHT_PRESTIGE: float = 0.30
    WEIGHT_VIEWERSHIP: float = 0.25
    WEIGHT_MARKET: float = 0.10
    WEIGHT_STAKES: float = 0.20
    WEIGHT_WINDOW_FIT: float = 0.10
    WEIGHT_NOVELTY: float = 0.05


settings = Settings()


def validate_settings() -> list[str]:
    """Return a list of warnings for missing or suspicious config values."""
    warnings = []
    if not settings.CFBD_API_KEY:
        warnings.append("CFBD_API_KEY is not set — acquisition pipeline will fail.")
    if not settings.ANTHROPIC_API_KEY:
        warnings.append("ANTHROPIC_API_KEY is not set — SMW vision extraction will fail.")
    weight_sum = (
        settings.WEIGHT_PRESTIGE + settings.WEIGHT_VIEWERSHIP + settings.WEIGHT_MARKET
        + settings.WEIGHT_STAKES + settings.WEIGHT_WINDOW_FIT + settings.WEIGHT_NOVELTY
    )
    if abs(weight_sum - 1.0) > 0.001:
        warnings.append(f"V(g) weights sum to {weight_sum:.3f}, not 1.0 — check pyproject.toml.")
    return warnings

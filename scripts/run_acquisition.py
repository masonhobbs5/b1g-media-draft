#!/usr/bin/env python3
"""
scripts/run_acquisition.py
──────────────────────────
Run the full data acquisition pipeline in one command.

Usage:
    python scripts/run_acquisition.py           # uses cached data where available
    python scripts/run_acquisition.py --refresh # force re-fetch from all sources
    python scripts/run_acquisition.py --check   # validate keys and print status only

Output files:
    data/raw/schedule/cfbd_2026_raw.json
    data/raw/viewership/smw_{2023,2024,2025}.json
    data/processed/game_schedule.json
    data/processed/team_prestige.json
    data/processed/viewership_pairs.json
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings, validate_settings
from config.constants import TEAM_META, BIG_TEN_TEAMS
from src.acquisition.cfbd_client import fetch_schedule
from src.acquisition.odds_client import fetch_prestige_scores
from src.acquisition.smw_scraper import build_viewership_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("acquisition")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B1G Monte Carlo — data acquisition pipeline")
    p.add_argument("--refresh", action="store_true", help="Force re-fetch from all APIs")
    p.add_argument("--check",   action="store_true", help="Validate config only, don't fetch")
    p.add_argument("--skip-smw", action="store_true", help="Skip Sports Media Watch scrape")
    p.add_argument("--discover-images", action="store_true",
                   help="List all SMW image URLs without calling the vision API, then exit")
    return p.parse_args()


def _discover_smw_images() -> None:
    """Fetch each SMW season page and print discovered image URLs. No API calls made."""
    import time
    import requests
    from bs4 import BeautifulSoup
    from src.acquisition.smw_scraper import SMW_URLS, _HEADERS_HTTP, _find_content_images

    total = 0
    for year in sorted(SMW_URLS):
        url = SMW_URLS[year]
        print(f"\n=== {year} — {url} ===")
        try:
            resp = requests.get(url, headers=_HEADERS_HTTP, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  ERROR fetching page: {exc}")
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        root = (
            soup.find("div", class_="entry-content")
            or soup.find("article")
            or soup.find("div", class_="post-content")
            or soup.body
        )
        imgs = _find_content_images(root, url)
        print(f"  {len(imgs)} image(s) found:")
        for img_url in imgs:
            print(f"    {img_url}")
        total += len(imgs)
        time.sleep(2)

    print(f"\nTotal: {total} image(s) across {len(SMW_URLS)} season pages.")
    print("If this looks correct, add Anthropic credits and run:")
    print("  rm data/raw/viewership/smw_*.json")
    print("  .venv/bin/python scripts/run_acquisition.py")


def main() -> None:
    args = parse_args()

    # ── Validate config ────────────────────────────────────────────────────────
    warnings = validate_settings()
    for w in warnings:
        logger.warning(w)
    if args.check:
        logger.info("Config check complete. %d warning(s).", len(warnings))
        return

    if args.discover_images:
        _discover_smw_images()
        return

    # ── Dataset 1: Schedule ────────────────────────────────────────────────────
    logger.info("=== DATASET 1: 2026 Big Ten Schedule ===")
    schedule = fetch_schedule(force_refresh=args.refresh)
    logger.info("Schedule: %d total games (%d conference, %d non-conference)",
        len(schedule),
        sum(1 for g in schedule if g["is_conference_game"]),
        sum(1 for g in schedule if not g["is_conference_game"]),
    )

    out_path = settings.PROCESSED_DIR / "game_schedule.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(schedule, f, indent=2)
    logger.info("Wrote game_schedule.json (%d games)", len(schedule))

    # ── Dataset 2a: Team prestige from odds ───────────────────────────────────
    logger.info("=== DATASET 2a: Team Prestige Scores ===")
    prestige_scores = fetch_prestige_scores()

    # ── Dataset 2b: SP+ from CFBD (requires Tier 1+ or use 2025 ratings) ─────
    # SP+ ratings pulled separately here to keep odds_client.py focused
    try:
        import cfbd
        import certifi
        from config.settings import settings as s
        cfg = cfbd.Configuration()
        cfg.access_token = s.CFBD_API_KEY
        cfg.ssl_ca_cert = certifi.where()
        api = cfbd.RatingsApi(cfbd.ApiClient(cfg))
        sp_raw = api.get_sp(year=2025)
        sp_lookup = {r.team: r.rating for r in sp_raw if r.team in BIG_TEN_TEAMS}
        sp_min, sp_max = min(sp_lookup.values()), max(sp_lookup.values())
        sp_normalized = {
            t: round((sp_lookup.get(t, sp_min) - sp_min) / (sp_max - sp_min), 4)
            for t in BIG_TEN_TEAMS
        }
    except Exception as e:
        logger.warning("Could not fetch SP+ ratings: %s. Using zeros.", e)
        sp_normalized = {t: 0.5 for t in BIG_TEN_TEAMS}
        sp_lookup = {}

    # ── Merge into team_prestige.json ─────────────────────────────────────────
    team_prestige = []
    for team in sorted(BIG_TEN_TEAMS):
        meta = TEAM_META[team]
        dma = meta["dma_rank"]
        odds_norm = prestige_scores.get(team, 0.01)
        sp_norm   = sp_normalized.get(team, 0.5)
        composite = round(0.6 * odds_norm + 0.4 * sp_norm, 4)

        team_prestige.append({
            "team":                        team,
            "championship_odds_normalized": odds_norm,
            "sp_plus_2025":                round(sp_lookup.get(team, 0.0), 2),
            "sp_plus_normalized":          sp_norm,
            "prestige_score":              composite,
            "dma_rank":                    dma,
            "market_score":                round(math.log(1 + (211 - dma)), 4),
            "timezone":                    meta["tz"],
            "noon_eligible":               meta["tz"] != "PT",
            "new_coach_2026":              meta.get("new_coach", False),
        })

    out_path = settings.PROCESSED_DIR / "team_prestige.json"
    with open(out_path, "w") as f:
        json.dump(team_prestige, f, indent=2)
    logger.info("Wrote team_prestige.json (%d teams)", len(team_prestige))

    # ── Dataset 2c: Viewership pairs ──────────────────────────────────────────
    if not args.skip_smw:
        logger.info("=== DATASET 2c: Historical Viewership Pairs ===")
        pairs = build_viewership_pairs(
            seasons=[2023, 2024, 2025],
            force_refresh=args.refresh,
            api_key=settings.ANTHROPIC_API_KEY or None,
        )
        logger.info("Built viewership_pairs.json (%d game records)", len(pairs))
    else:
        logger.info("Skipping SMW scrape (--skip-smw flag set).")

    logger.info("=== Acquisition complete. Run tests/unit/ to validate output. ===")


if __name__ == "__main__":
    main()

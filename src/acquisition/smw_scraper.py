"""
smw_scraper.py
==============
Scrapes Big Ten game viewership from Sports Media Watch.
 
WHY VISION API
--------------
SMW embeds its ratings data as screenshot images inside articles rather than
HTML text.  A pure text/regex scraper only catches the 1-2 records that appear
in prose captions — hence the original "only 3 records" result.
 
This version:
  1. Fetches the season page and finds all <img> tags in the article body.
  2. Downloads each image and base64-encodes it.
  3. Sends each image to Claude's vision API with a structured-extraction prompt.
  4. Merges the JSON records returned by Claude with any prose records caught by
     the fallback regex (in case SMW ever publishes both formats).
 
Output
------
  data/raw/viewership/smw_{year}.json          — one file per season
  data/processed/viewership_pairs.json         — team-pair lookup table
"""
 
from __future__ import annotations
 
import base64
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
 
import requests
from bs4 import BeautifulSoup
 
from src.acquisition.cfbd_client import fetch_postseason_lookup, fetch_historical_games

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
SMW_URLS = {
    2023: "https://www.sportsmediawatch.com/college-football-tv-ratings/2023-season/",
    2024: "https://www.sportsmediawatch.com/college-football-tv-ratings/2024-season/",
    2025: "https://www.sportsmediawatch.com/college-football-tv-ratings/",
}
 
# ---------------------------------------------------------------------------
# Config — override via environment or config module as needed
# ---------------------------------------------------------------------------
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
 
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
VISION_MODEL      = "claude-opus-4-5"
MAX_TOKENS        = 4096
 
_HEADERS_HTTP = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
 
# ---------------------------------------------------------------------------
# Big Ten roster (update as conference membership changes)
# ---------------------------------------------------------------------------
BIG_TEN_TEAMS: list[str] = [
    "Illinois", "Indiana", "Iowa", "Maryland", "Michigan", "Michigan State",
    "Minnesota", "Nebraska", "Northwestern", "Notre Dame", "Ohio State", "Oregon",
    "Penn State", "Purdue", "Rutgers", "UCLA", "USC", "Washington", "Wisconsin",
]
 
# ---------------------------------------------------------------------------
# Fallback regex — catches the small number of records SMW publishes in prose
# ---------------------------------------------------------------------------
_NETWORK_FRAG = r"(?:FOX|CBS|NBC(?:/Peacock)?|ESPN(?:\+)?|ABC|BTN|Big\s+Ten\s+Network|FS1|Peacock)"
_TEAM_FRAG    = r"[\w\s\'\.\-]"
_SEP          = r"\s*(?:[-\u2013]|(?:\bat\b|\bvs\.?\b))\s*"
 
_PAT_A = re.compile(
    rf"({_TEAM_FRAG}+?){_SEP}({_TEAM_FRAG}+?)"
    r"\s*\(" rf"({_NETWORK_FRAG})" r"\)[:\s]*"
    r"([\d\.]+)\s*million",
    re.IGNORECASE,
)
_PAT_B = re.compile(
    rf"({_TEAM_FRAG}+?){_SEP}({_TEAM_FRAG}+?)"
    r"[:\s]+?([\d\.]+)\s*million"
    r"[^.\n]*?(?:\(|on\s|via\s|,\s*)" rf"({_NETWORK_FRAG})",
    re.IGNORECASE,
)
 
 
# ===========================================================================
# Public API
# ===========================================================================
 
def scrape_season(
    year: int,
    force_refresh: bool = False,
    api_key: str | None = None,
) -> list[dict]:
    """
    Scrape one SMW season page and return a list of game viewership records.
 
    Parameters
    ----------
    year          : Season year (2023-2025).
    force_refresh : Ignore on-disk cache and re-scrape.
    api_key       : Anthropic API key.  Falls back to the ANTHROPIC_API_KEY
                    environment variable if not supplied.
    """
    cache_path = RAW_DIR / "viewership" / f"smw_{year}.json"
    if cache_path.exists() and not force_refresh:
        logger.info("Loading SMW %d viewership from cache.", year)
        with open(cache_path) as f:
            return json.load(f)
 
    url = SMW_URLS.get(year)
    if not url:
        raise ValueError(f"No SMW URL configured for year {year}")
 
    logger.info("Fetching SMW %d page: %s", year, url)
    resp = requests.get(url, headers=_HEADERS_HTTP, timeout=15)
    resp.raise_for_status()
    time.sleep(2)
 
    soup = BeautifulSoup(resp.text, "lxml")
    content_root = (
        soup.find("div", class_="entry-content")
        or soup.find("article")
        or soup.find("div", class_="post-content")
        or soup.body
    )
 
    records: list[dict] = []
 
    # Step 1: Vision extraction from embedded images
    image_urls = _find_content_images(content_root, base_url=url)
    logger.info("Found %d content images to process with vision.", len(image_urls))
 
    for img_url in image_urls:
        img_records = _extract_from_image(img_url, year, api_key=api_key)
        records.extend(img_records)
        time.sleep(1)
 
    # Step 2: Fallback regex over prose text
    prose_blocks = [
        tag.get_text(separator=" ", strip=True)
        for tag in content_root.find_all(["li", "p"])
        if tag.get_text(strip=True)
    ]
    records.extend(_extract_from_prose(prose_blocks, year))
 
    # Step 3: Deduplicate
    records = _deduplicate(records)
 
    b1g = sum(1 for r in records if r["is_b1g_game"])
    logger.info("Year %d: %d total records, %d Big Ten games.", year, len(records), b1g)
 
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)
 
    return records
 
 
# Name normalization: SMW scraper output → CFBD canonical names
_CFBD_NAME_MAP: dict[str, str] = {
    "Cal":           "California",
    "Mississippi":   "Ole Miss",
    "Miami (Ohio)":  "Miami (OH)",
    "Hawaii":        "Hawai'i",
    "Central Mich.": "Central Michigan",
    "Western Mich.": "Western Michigan",
    "FSU":           "Florida State",
    "Youngstown St.": "Youngstown State",
    "Jacksonville St.": "Jacksonville State",
    "La Tech":       "Louisiana Tech",
    "FAU":           "Florida Atlantic",
    "WKU":           "Western Kentucky",
    "UCF":           "Central Florida",
    "UMass":         "Massachusetts",
    "BYU":           "BYU",
    "Sam Houston":   "Sam Houston State",
}


def _to_cfbd_name(name: str) -> str:
    """Normalize a scraper team name to CFBD canonical form."""
    return _CFBD_NAME_MAP.get(name, name)


def _time_slot(hour_et: int | None) -> str:
    """Bucket ET kickoff hour into a broadcast window."""
    if hour_et is None:
        return "unknown"
    if hour_et < 15:
        return "noon"
    if hour_et < 19:
        return "afternoon"
    return "primetime"


def build_viewership_pairs(
    seasons: list[int] = [2023, 2024, 2025],
    force_refresh: bool = False,
    api_key: str | None = None,
) -> list[dict]:
    """Collect every individual B1G game record across seasons into a flat list,
    enriched with week, time slot, scores, and pregame Elo from CFBD."""
    postseason = fetch_postseason_lookup(years=seasons, force_refresh=force_refresh)
    historical = fetch_historical_games(years=seasons, force_refresh=force_refresh)
    records: list[dict] = []

    for year in seasons:
        for r in scrape_season(year, force_refresh=force_refresh, api_key=api_key):
            if not r["is_b1g_game"]:
                continue
            # Fall back to raw name when the opponent isn't in BIG_TEN_TEAMS
            name_a = r["team_a"] or r["team_a_raw"]
            name_b = r["team_b"] or r["team_b_raw"]
            # Skip garbage records (self-matches, partial sentences)
            if name_a == name_b or len(name_a) > 40 or len(name_b) > 40:
                continue
            ta, tb = sorted([name_a, name_b])
            # Normalize to CFBD naming for lookup
            cfbd_a = _to_cfbd_name(name_a)
            cfbd_b = _to_cfbd_name(name_b)
            key = (year, frozenset({cfbd_a, cfbd_b}))
            game_info = postseason.get(key, {})
            hist = historical.get(key, {})
            records.append({
                "season":            r["season"],
                "week":              hist.get("week"),
                "team_a":            ta,
                "team_b":            tb,
                "viewers_millions":  r["viewers_millions"],
                "network":           r["network"],
                "time_slot":         _time_slot(hist.get("start_hour_et")),
                "home_team":         hist.get("home_team"),
                "away_team":         hist.get("away_team"),
                "home_points":       hist.get("home_points"),
                "away_points":       hist.get("away_points"),
                "home_pregame_elo":  hist.get("home_pregame_elo"),
                "away_pregame_elo":  hist.get("away_pregame_elo"),
                "is_conference_game": hist.get("conference_game", False),
                "is_bowl_game":      game_info.get("is_bowl_game", False),
                "is_playoff_game":   game_info.get("is_playoff_game", False),
                "source":            r.get("source", "unknown"),
            })

    out = PROCESSED_DIR / "viewership_pairs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Wrote %d individual game records to %s", len(records), out)
    return records


# ===========================================================================
# Image handling
# ===========================================================================
 
def _find_content_images(content_root, base_url: str) -> list[str]:
    """
    Return absolute URLs for every content <img> in the article.
    Skips small decorative images (pixel width < 200).
    Resolves lazy-loaded images by preferring data-src / data-lazy-src over src
    (SMW uses a perfmatters lazy-loader that replaces real URLs with SVG placeholders
    in the src attribute until JS fires — the real URL is always in data-src).
    """
    from urllib.parse import urljoin
    urls: list[str] = []
    for img in content_root.find_all("img"):
        # Prefer lazy-load attributes; fall back to src only if no data-src exists
        src = (
            img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("src")
        )
        if not src or src.startswith("data:"):
            continue
        width = img.get("width")
        if width and str(width).isdigit() and int(width) < 200:
            continue
        abs_url = urljoin(base_url, src)
        if re.search(r"\.(png|jpe?g|webp|gif)(\?|$)", abs_url, re.IGNORECASE):
            urls.append(abs_url)
    return urls
 
 
def _image_to_base64(url: str) -> tuple[str, str] | None:
    """Download an image and return (base64_data, media_type), or None on failure."""
    try:
        resp = requests.get(url, headers=_HEADERS_HTTP, timeout=15)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if "png"  in ct: media_type = "image/png"
        elif "webp" in ct: media_type = "image/webp"
        elif "gif"  in ct: media_type = "image/gif"
        else:              media_type = "image/jpeg"
        return base64.standard_b64encode(resp.content).decode(), media_type
    except Exception as exc:
        logger.warning("Could not download image %s: %s", url, exc)
        return None
 
 
def _extract_from_image(
    img_url: str,
    year: int,
    api_key: str | None = None,
) -> list[dict]:
    """
    Send one image to Claude's vision API and parse out viewership records.
    Returns a (possibly empty) list of record dicts.
    """
    import os
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "No Anthropic API key found. Pass api_key= or set ANTHROPIC_API_KEY."
        )
 
    result = _image_to_base64(img_url)
    if result is None:
        return []
    b64, media_type = result
 
    prompt = (
        "This image is from Sports Media Watch and shows college football TV viewership ratings. "
        "Extract every game entry visible in the image.\n\n"
        "Return ONLY a JSON array with no prose or markdown fences. Each element must have:\n"
        '  "team_a"           : string  (home or first-listed team, full school name)\n'
        '  "team_b"           : string  (away or second-listed team, full school name)\n'
        '  "viewers_millions" : number  (e.g. 7.87)\n'
        '  "network"          : string  (e.g. "FOX", "ABC", "ESPN", "BTN", "Peacock")\n\n'
        "Use null for any field not visible. "
        "Return [] if the image contains no game viewership data."
    )
 
    payload = {
        "model": VISION_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    }
 
    headers = {
        "x-api-key":         key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
 
    try:
        resp = requests.post(ANTHROPIC_API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Vision API call failed for %s: %s", img_url, exc)
        return []
 
    raw = resp.json()["content"][0]["text"].strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
 
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Bad JSON from vision API for %s: %s\nRaw: %.300s", img_url, exc, raw)
        return []
 
    records: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        a_raw = str(entry.get("team_a") or "").strip()
        b_raw = str(entry.get("team_b") or "").strip()
        try:
            viewers = float(entry["viewers_millions"])
        except (KeyError, TypeError, ValueError):
            continue
        network = _normalise_network(str(entry.get("network") or "UNKNOWN"))
        team_a  = _fuzzy_match_team(a_raw)
        team_b  = _fuzzy_match_team(b_raw)
        records.append({
            "season":           year,
            "team_a_raw":       a_raw,
            "team_b_raw":       b_raw,
            "team_a":           team_a,
            "team_b":           team_b,
            "viewers_millions": viewers,
            "network":          network,
            "is_b1g_game":      team_a is not None or team_b is not None,
            "needs_review":     team_a is None or team_b is None,
            "source":           "vision",
        })
 
    logger.info("  %s -> %d records", img_url.split("/")[-1], len(records))
    return records
 
 
# ===========================================================================
# Prose fallback
# ===========================================================================
 
def _extract_from_prose(blocks: list[str], year: int) -> list[dict]:
    records: list[dict] = []
    seen: set[tuple] = set()
    for text in blocks:
        for match, pat_name in _iter_matches(text):
            if pat_name == "A":
                a_raw, b_raw = match.group(1).strip(), match.group(2).strip()
                network      = match.group(3).strip()
                viewers      = float(match.group(4))
            else:
                a_raw, b_raw = match.group(1).strip(), match.group(2).strip()
                viewers      = float(match.group(3))
                network      = match.group(4).strip()
            network = _normalise_network(network)
            key = (a_raw.lower(), b_raw.lower(), viewers)
            if key in seen:
                continue
            seen.add(key)
            team_a = _fuzzy_match_team(a_raw)
            team_b = _fuzzy_match_team(b_raw)
            records.append({
                "season":           year,
                "team_a_raw":       a_raw,
                "team_b_raw":       b_raw,
                "team_a":           team_a,
                "team_b":           team_b,
                "viewers_millions": viewers,
                "network":          network,
                "is_b1g_game":      team_a is not None or team_b is not None,
                "needs_review":     team_a is None or team_b is None,
                "source":           "prose",
            })
    return records
 
 
def _iter_matches(text: str):
    hits  = [(m.start(), m, "A") for m in _PAT_A.finditer(text)]
    hits += [(m.start(), m, "B") for m in _PAT_B.finditer(text)]
    hits.sort(key=lambda x: (x[0], x[2]))
    last_end = 0
    for start, m, pat in hits:
        if start >= last_end:
            yield m, pat
            last_end = m.end()
 
 
# ===========================================================================
# Shared helpers
# ===========================================================================
 
def _deduplicate(records: list[dict]) -> list[dict]:
    """Keep one record per (team_a, team_b, viewers); prefer vision over prose."""
    seen: dict[tuple, dict] = {}
    for r in records:
        key = (
            (r.get("team_a") or r["team_a_raw"]).lower(),
            (r.get("team_b") or r["team_b_raw"]).lower(),
            r["viewers_millions"],
        )
        if key not in seen or r.get("source") == "vision":
            seen[key] = r
    return list(seen.values())
 
 
def _normalise_network(raw: str) -> str:
    up = raw.upper().strip()
    if "PEACOCK" in up:  return "Peacock"
    if "BIG TEN" in up or up == "BTN": return "BTN"
    if "ESPN+"   in up:  return "ESPN+"
    return up
 
 
def _fuzzy_match_team(raw: str) -> str | None:
    raw_lower = raw.lower().strip()
    for team in BIG_TEN_TEAMS:
        if raw_lower == team.lower():
            return team
    matches = [t for t in BIG_TEN_TEAMS if t.lower() in raw_lower or raw_lower in t.lower()]
    if matches:
        return max(matches, key=len)
    aliases = {
        "buckeyes":      "Ohio State",     "wolverines":   "Michigan",
        "nittany lions": "Penn State",     "hoosiers":     "Indiana",
        "ducks":         "Oregon",         "trojans":      "USC",
        "bruins":        "UCLA",           "badgers":      "Wisconsin",
        "hawkeyes":      "Iowa",           "gophers":      "Minnesota",
        "illini":        "Illinois",       "wildcats":     "Northwestern",
        "cornhuskers":   "Nebraska",       "terrapins":    "Maryland",
        "scarlet knights": "Rutgers",      "boilermakers": "Purdue",
        "spartans":      "Michigan State", "huskies":      "Washington",
    }
    for alias, team in aliases.items():
        if alias in raw_lower:
            return team
    return None
 
 
# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY before running.")
    else:
        for yr in [2023, 2024, 2025]:
            scrape_season(yr, force_refresh=False, api_key=api_key)
        pairs = build_viewership_pairs()
        print(f"\nDone - {len(pairs)} team pairs written to data/processed/viewership_pairs.json")
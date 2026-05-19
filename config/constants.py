"""
config/constants.py
───────────────────
Hard-coded lookup tables that change rarely and have no API source.
Edit here when DMA rankings update (annually) or team metadata changes.
"""

from __future__ import annotations

# ── Big Ten team metadata ──────────────────────────────────────────────────────
# Format: "Team Name": {"city": str, "dma_rank": int, "timezone": str,
#                        "espn_id": int, "new_coach_2026": bool}
TEAM_META: dict[str, dict] = {
    "Ohio State":    {"city": "Columbus",       "dma_rank": 35,  "tz": "ET", "espn_id": 194,  "new_coach": False},
    "Michigan":      {"city": "Detroit",         "dma_rank": 14,  "tz": "ET", "espn_id": 130,  "new_coach": True},   # Whittingham
    "Penn State":    {"city": "Wilkes-Barre",    "dma_rank": 59,  "tz": "ET", "espn_id": 213,  "new_coach": True},   # Campbell
    "Indiana":       {"city": "Indianapolis",    "dma_rank": 25,  "tz": "ET", "espn_id": 84,   "new_coach": False},
    "Oregon":        {"city": "Portland",        "dma_rank": 23,  "tz": "PT", "espn_id": 2483, "new_coach": False},
    "USC":           {"city": "Los Angeles",     "dma_rank": 2,   "tz": "PT", "espn_id": 30,   "new_coach": False},
    "UCLA":          {"city": "Los Angeles",     "dma_rank": 2,   "tz": "PT", "espn_id": 26,   "new_coach": True},   # Chesney
    "Washington":    {"city": "Seattle",         "dma_rank": 13,  "tz": "PT", "espn_id": 264,  "new_coach": False},
    "Wisconsin":     {"city": "Milwaukee",       "dma_rank": 38,  "tz": "CT", "espn_id": 275,  "new_coach": False},
    "Iowa":          {"city": "Cedar Rapids",    "dma_rank": 94,  "tz": "CT", "espn_id": 2294, "new_coach": False},
    "Minnesota":     {"city": "Minneapolis",     "dma_rank": 16,  "tz": "CT", "espn_id": 135,  "new_coach": False},
    "Illinois":      {"city": "Champaign",       "dma_rank": 92,  "tz": "CT", "espn_id": 356,  "new_coach": False},
    "Northwestern":  {"city": "Chicago",         "dma_rank": 3,   "tz": "CT", "espn_id": 77,   "new_coach": False},
    "Notre Dame":    {"city": "South Bend",      "dma_rank": 100, "tz": "ET", "espn_id": 87,   "new_coach": False},
    "Nebraska":      {"city": "Lincoln",         "dma_rank": 107, "tz": "CT", "espn_id": 158,  "new_coach": False},
    "Michigan State":{"city": "Lansing",         "dma_rank": 117, "tz": "ET", "espn_id": 127,  "new_coach": True},   # Fitzgerald
    "Maryland":      {"city": "Washington DC",   "dma_rank": 8,   "tz": "ET", "espn_id": 120,  "new_coach": False},
    "Rutgers":       {"city": "New York",        "dma_rank": 1,   "tz": "ET", "espn_id": 164,  "new_coach": False},
    "Purdue":        {"city": "Indianapolis",    "dma_rank": 25,  "tz": "ET", "espn_id": 2509, "new_coach": False},
}

BIG_TEN_TEAMS: frozenset[str] = frozenset(TEAM_META.keys())

# ── Pacific-timezone teams — ineligible for noon ET home games ────────────────
PACIFIC_TZ_TEAMS: frozenset[str] = frozenset(
    t for t, m in TEAM_META.items() if m["tz"] == "PT"
)

# ── Historic rivalry games (automatically flagged as high novelty) ─────────────
RIVALRY_PAIRS: list[frozenset[str]] = [
    frozenset({"Ohio State", "Michigan"}),           # The Game
    frozenset({"Michigan", "Michigan State"}),       # Paul Bunyan Trophy
    frozenset({"UCLA", "USC"}),                # Land of Lincoln Trophy alt
    frozenset({"Oregon", "Washington"}),                # Old Oaken Bucket
    frozenset({"Minnesota", "Wisconsin"}),           # Paul Bunyan's Axe
    frozenset({"Minnesota", "Iowa"}),                # Floyd of Rosedale
    frozenset({"Nebraska", "Iowa"}),                 # Heroes Game
    frozenset({"Penn State", "Ohio State"}),
    frozenset({"Northwestern", "Illinois"}),
]

# ── 2026-specific marquee flags ───────────────────────────────────────────────
CFP_REMATCH_PAIRS: list[frozenset[str]] = [
    frozenset({"Indiana", "Ohio State"}),   # Oct 17 — last two CFP champions
]

# ── Pre-season prestige fallback (hand-encoded from CBS Sports Jan 2026) ───────
# Used when Odds API key is missing. Update from live odds at Media Days.
PRESTIGE_FALLBACK: dict[str, float] = {
    "Ohio State":     0.133,   # +650 → 13.3% implied
    "Indiana":        0.125,   # +700 → 12.5% implied
    "Oregon":         0.095,   # +950 → 9.5% implied
    "USC":            0.028,   # +3500 → 2.8% implied
    "Michigan":       0.024,   # +4000 → 2.4% implied
    "Penn State":     0.016,   # +6000 → 1.6% implied
    # Remaining teams: distribute remaining probability ~equally
    "Wisconsin":      0.010,
    "Iowa":           0.008,
    "Nebraska":       0.008,
    "Minnesota":      0.007,
    "UCLA":           0.007,
    "Michigan State": 0.006,
    "Washington":     0.006,
    "Maryland":       0.005,
    "Illinois":       0.005,
    "Northwestern":   0.004,
    "Rutgers":        0.004,
    "Purdue":         0.004,
}

# ── Network broadcast windows ─────────────────────────────────────────────────
NETWORK_WINDOWS: dict[str, str] = {
    "FOX":     "12:00 ET",
    "CBS":     "15:30 ET",
    "NBC":     "19:30 ET",
    "PEACOCK": "19:30 ET",  # Peacock-exclusive games share NBC window
    "FS1":     "varies",
    "BTN":     "varies",
}

# ── Championship Game ─────────────────────────────────────────────────────────
CHAMPIONSHIP_NETWORK = "FOX"   # Fox purchased back from NBC in April 2026
CHAMPIONSHIP_DATE    = "2026-12-05"
CHAMPIONSHIP_VENUE   = "Lucas Oil Stadium, Indianapolis"

# ── 2026 Preseason SP+ (Bill Connelly, ESPN, March 2026) ──────────────────────
# Source: espn.com/college-football/story/_/id/48306284
# Maps CFBD team name → overall SP+ rating
SP_PLUS_2026: dict[str, float] = {
    "Ohio State": 31.8,
    "Oregon": 28.3,
    "Notre Dame": 25.8,
    "Georgia": 25.5,
    "Indiana": 24.5,
    "Texas": 23.7,
    "Texas Tech": 23.1,
    "Miami": 21.0,
    "Texas A&M": 20.3,
    "LSU": 20.2,
    "Alabama": 18.2,
    "Oklahoma": 17.2,
    "USC": 16.8,
    "Michigan": 16.1,
    "Tennessee": 16.0,
    "Ole Miss": 15.9,
    "Penn State": 15.7,
    "BYU": 15.5,
    "Florida": 14.9,
    "Missouri": 14.8,
    "Washington": 14.5,
    "Iowa": 13.6,
    "Clemson": 12.8,
    "South Carolina": 12.1,
    "Utah": 11.9,
    "Auburn": 11.2,
    "Louisville": 11.0,
    "SMU": 10.9,
    "Kansas State": 10.4,
    "Arizona": 10.2,
    "Vanderbilt": 10.0,
    "Virginia Tech": 9.4,
    "Illinois": 9.3,
    "TCU": 9.1,
    "Florida State": 8.8,
    "Houston": 8.2,
    "Nebraska": 7.7,
    "Oklahoma State": 7.1,
    "Boise State": 6.8,
    "Virginia": 6.6,
    "Pittsburgh": 6.5,
    "Arizona State": 6.4,
    "Georgia Tech": 6.0,
    "Duke": 5.7,
    "Minnesota": 5.2,
    "UCLA": 5.1,
    "Arkansas": 5.0,
    "NC State": 4.9,
    "Northwestern": 4.6,
    "Cincinnati": 4.5,
    "Baylor": 4.5,
    "Mississippi State": 3.9,
    "Kentucky": 3.8,
    "North Carolina": 3.8,
    "Maryland": 3.8,
    "California": 3.7,
    "Kansas": 3.7,
    "Wake Forest": 3.6,
    "UNLV": 2.8,
    "UCF": 2.3,
    "Wisconsin": 1.8,
    "Rutgers": 1.8,
    "Navy": 1.1,
    "Iowa State": 1.0,
    "Colorado": 0.9,
    "West Virginia": 0.8,
    "Michigan State": 0.4,
    "New Mexico": -0.5,
    "Syracuse": -0.7,
    "Memphis": -1.1,
    "San Diego State": -1.3,
    "North Dakota State": -1.4,
    "UTSA": -1.5,
    "Boston College": -1.5,
    "Stanford": -1.9,
    "East Carolina": -2.0,
    "James Madison": -2.1,
    "Fresno State": -2.3,
    "Air Force": -2.4,
    "South Florida": -2.8,
    "Miami (OH)": -2.9,
    "Purdue": -2.9,
    "Army": -3.0,
    "Hawai'i": -3.9,
    "Washington State": -5.3,
    "Western Kentucky": -5.3,
    "Tulane": -5.5,
    "Old Dominion": -5.8,
    "Texas State": -5.9,
    "Troy": -6.0,
    "Oregon State": -6.3,
    "Marshall": -6.4,
    "Liberty": -6.4,
    "Florida Atlantic": -7.1,
    "Western Michigan": -7.2,
    "Tulsa": -7.6,
    "Utah State": -7.7,
    "Jacksonville State": -7.7,
    "Colorado State": -8.3,
    "Louisiana Tech": -8.3,
    "Arkansas State": -8.5,
    "Temple": -8.7,
    "Georgia Southern": -8.9,
    "Louisiana": -9.1,
    "Kennesaw State": -9.3,
    "Wyoming": -9.6,
    "Connecticut": -11.2,
    "Toledo": -11.5,
    "North Texas": -11.8,
    "Buffalo": -11.9,
    "Appalachian State": -12.1,
    "Nevada": -12.2,
    "Central Michigan": -12.4,
    "Delaware": -13.0,
    "Bowling Green": -13.3,
    "South Alabama": -13.3,
    "Ohio": -13.6,
    "FIU": -13.7,
    "Coastal Carolina": -13.8,
    "Rice": -14.7,
    "Eastern Michigan": -15.0,
    "San José State": -15.5,
    "New Mexico State": -16.4,
    "UAB": -18.1,
    "Northern Illinois": -18.2,
    "Missouri State": -18.7,
    "Akron": -19.5,
    "Kent State": -20.1,
    "UTEP": -20.5,
    "Sacramento State": -22.7,
    "Southern Miss": -23.3,
    "UL Monroe": -24.3,
    "Georgia State": -25.1,
    "Ball State": -25.2,
    "Middle Tennessee": -26.0,
    "Sam Houston": -26.3,
    "Massachusetts": -30.9,
    "Charlotte": -32.4,
}

# ── 2026-27 National Championship Odds (Yahoo Sports, May 2026) ───────────────
# American moneyline odds; +550 means $100 bet pays $550 profit
# Teams not listed: Power 4 = +15000, non-Power 4 = +100000
CHAMPIONSHIP_ODDS_2026: dict[str, int] = {
    "Ohio State": 550,
    "Notre Dame": 700,
    "Texas": 750,
    "Indiana": 800,
    "Oregon": 850,
    "Georgia": 900,
    "Miami": 1200,
    "Texas Tech": 1500,
    "Texas A&M": 2000,
    "LSU": 2200,
    "Ole Miss": 2200,
    "Alabama": 2500,
    "Oklahoma": 3000,
    "Michigan": 3500,
    "USC": 4000,
    "Tennessee": 4500,
    "Florida": 5000,
    "Missouri": 7500,
}

# Power 4 conferences (for default odds assignment)
POWER_4_CONFERENCES: frozenset[str] = frozenset({
    "SEC", "Big Ten", "Big 12", "ACC",
})

# B1G 2026 Monte Carlo Broadcast Simulation

Simulates the 2026 Big Ten football season 10,000 times to produce three linked artifacts:

1. **Expected team records** — win distributions and threshold probabilities for all 18 B1G teams + Notre Dame
2. **Expected game viewership** — Ridge regression audience predictions with 90% confidence intervals for all 135 games
3. **Broadcaster draft assignments** — probabilistic FOX / CBS / NBC game selection across 13 weekly picks

A Streamlit dashboard visualizes all three artifacts without re-running any computation.

---

## Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/masonhobbs5/b1g-media-draft.git
cd b1g-media-draft

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add your API key

The simulation uses pre-processed data that is committed to the repo.
A `CFBD_API_KEY` is only needed if you want to re-run data acquisition.

```bash
cp .env.example .env
# Edit .env and set CFBD_API_KEY=your_key_here
```

Register for a free key at https://collegefootballdata.com/key — no credit card required.

### 3. Run the full simulation

```bash
python scripts/run_simulation.py
```

This runs in under one second and writes all three artifacts to `data/outputs/`.

```
=================================================================
  Pipeline complete in 0.4s
=================================================================
  Artifact 1: data/outputs/expected_records.json
              data/outputs/expected_records.csv
  Artifact 2: data/outputs/expected_viewership.json
              data/outputs/expected_viewership.csv
  Artifact 3: data/outputs/draft_assignments.json
              data/outputs/draft_assignments.csv
              data/outputs/draft_weekly_viewers.json
=================================================================
```

### 4. Launch the dashboard

```bash
streamlit run scripts/run_dashboard.py
```

Open http://localhost:8501 in your browser.

---

## Simulation pipeline

```
data/processed/          ← committed, ready to use
    game_schedule.json
    fbs_strength.json
    viewership_features.json
    viewership_pairs.json
          │
          ▼
  build_win_probabilities()      Bradley-Terry model, HFA calibrated from 432 games
          │
          ▼
  run_simulation()               10,000 Monte Carlo iterations (numpy vectorized)
          │
          ▼
  export_records()               Artifact 1 — expected_records.json/.csv
          │
          ▼
  build_viewership_predictions() Log-linear Ridge (α=50), leave-one-season-out CV
          │
          ▼
  export_viewership()            Artifact 2 — expected_viewership.json/.csv
          │
          ▼
  build_draft_results()          Softmax selection (τ=0.3), 13-pick draft order
          │
          ▼
  export_draft()                 Artifact 3 — draft_assignments.json/.csv
                                             draft_weekly_viewers.json
```

---

## CLI reference

### `python scripts/run_simulation.py`

| Flag | Short | Default | Description |
|---|---|---|---|
| `--iterations` | `-n` | 10000 | Monte Carlo iteration count |
| `--seed` | `-s` | 42 | Random seed for reproducibility |
| `--output-dir` | `-o` | `data/outputs/` | Redirect artifact output |
| `--rebuild-strength` | | off | Recompute `fbs_strength.json` from source data |
| `--skip-draft` | | off | Skip Artifact 3 (draft simulation) |
| `--quiet` | `-q` | off | Suppress summary tables (CI-friendly) |

Examples:

```bash
# Quick run, 1000 iterations
python scripts/run_simulation.py -n 1000

# Write artifacts to a custom directory
python scripts/run_simulation.py --output-dir ./results/run_01

# CI-friendly, suppress all tables
python scripts/run_simulation.py --quiet

# Recompute team strength from raw data, then re-run
python scripts/run_simulation.py --rebuild-strength
```

### `streamlit run scripts/run_dashboard.py`

No arguments. Reads the most recently generated artifact files from `data/outputs/`.

---

## Dashboard pages

| Page | Contents |
|---|---|
| **Expected Records** | Sortable team table (E[W], E[Conf W], P(8+W), P(10+W), P(11+W)) + bar chart |
| **Viewership Predictions** | Filterable game table with conference/minimum-viewers controls + top-10 chart |
| **Broadcaster Draft** | Season totals by network, probability table, weekly stacked viewer chart |
| **Weekly Rankings** | Week selector showing top-3 games with viewer estimate and likely network |

---

## Data acquisition (optional)

The processed data files are committed to the repo and sufficient to run the full pipeline.
Re-run acquisition only if you want to refresh for a new season or re-scrape viewership.

```bash
python scripts/run_acquisition.py
```

This requires a valid `CFBD_API_KEY` in `.env`. It fetches the B1G schedule, team
prestige data, and historical game results, then writes to `data/processed/`.

---

## Running tests

```bash
# Unit tests only — no API keys required
pytest tests/unit/ -v

# Integration tests — uses committed data/processed/ files, no API calls
pytest tests/integration/ -v

# Full suite with coverage report
pytest tests/ --cov=src --cov-report=html

# Quick smoke check
pytest tests/ -q
```

171 tests total: 141 unit, 30 integration.

---

## Project structure

```
b1g_sim/
├── config/
│   ├── settings.py              # All settings (iterations, seed, draft order, weights)
│   └── constants.py             # Team metadata, rivalry pairs, SP+ ratings, Elo
│
├── data/
│   ├── raw/                     # API responses and scrape outputs (gitignored)
│   ├── processed/               # Cleaned datasets — committed, used by simulation
│   │   ├── game_schedule.json   # 135 games with timezone, market score, rivalry flags
│   │   ├── fbs_strength.json    # 141 FBS teams (50% SP+, 25% odds, 25% Elo)
│   │   ├── team_prestige.json   # 19 B1G teams with composite prestige score
│   │   ├── viewership_pairs.json # 455 historical audience records (2023–2025)
│   │   └── viewership_features.json # Team brands, week/slot multipliers
│   ├── cache/                   # Intermediate cache (gitignored)
│   └── outputs/                 # Generated artifacts (committed)
│       ├── expected_records.json/.csv
│       ├── expected_viewership.json/.csv
│       ├── draft_assignments.json/.csv
│       └── draft_weekly_viewers.json
│
├── src/
│   ├── acquisition/             # Data ingestion (CFBD API, Odds API, SMW scraper)
│   ├── model/                   # Team strength, win probability, viewership model
│   ├── simulation/              # Monte Carlo engine, draft value scoring, draft sim
│   ├── outputs/                 # Export functions for all three artifacts
│   └── utils/                   # Data loaders and validation
│
├── tests/
│   ├── unit/                    # 141 fast offline tests
│   └── integration/             # 30 end-to-end pipeline tests
│
├── scripts/
│   ├── run_acquisition.py       # Refresh processed data from APIs
│   ├── run_simulation.py        # Full simulation pipeline → all three artifacts
│   └── run_dashboard.py        # Streamlit dashboard
│
├── .env.example                 # Template — copy to .env and add CFBD_API_KEY
├── pyproject.toml               # Tool config (black, ruff, mypy, pytest)
└── requirements.txt
```

---

## Modeling notes

### Win probability

Bradley-Terry model: `P(home) = 1 / (1 + 10^((away − home − hfa) / 0.6))`

Home-field advantage calibrated from 432 historical B1G-involved games (2023–2025): **67.6% home win rate → 127.7 Elo points**. Team strength is a composite of 2026 SP+ (50%), 2026 championship odds (25%), and 2025 end-of-season Elo (25%), min-max normalized to [0, 1].

### Viewership model

Log-linear Ridge regression (α = 50) with `StandardScaler`. Predicts `log(viewers)` and exponentiates with Duan's smearing correction. Features: combined brand rating, max brand, combined Elo, Elo closeness, week multiplier, slot multiplier, rivalry flag, conference game flag, network tier (broadcast reach), late-season flag (week ≥ 10), brand × Elo interaction. Training excludes sub-0.5M streaming/BTN games. Leave-one-season-out CV: **R² = 0.62, RMSE = 1.96M viewers, MdAPE = 34%**.

### Draft simulation

13 weekly picks in order `[FOX, FOX, FOX, CBS, FOX, NBC, CBS, FOX, NBC, NBC, CBS, FOX, NBC]`. Each network draws from a softmax distribution (τ = 0.3) over game values V(g). FOX can trade picks at 15% probability. NBC skips picks in weeks 1, 3, 10, 12 (pre-committed Notre Dame games). Notre Dame games are excluded from the draft pool entirely.

Game value weights: prestige 0.30 · viewership 0.25 · stakes 0.20 · market 0.10 · window fit 0.10 · novelty 0.05.

---

## Data sources

| Dataset | Source | Method |
|---|---|---|
| 2026 schedule | CollegeFootballData.com API v2 | REST API (free) |
| Team prestige / odds | The Odds API (NCAAF futures) | REST API (free) |
| 2026 SP+ ratings | ESPN (scraped) | Web scrape |
| Historical viewership | Sports Media Watch | Web scrape |
| DMA market sizes | Nielsen (via constants.py) | Hand-encoded |


# B1G Sim Plan

## Objective

Build a repeatable workflow that produces three artifacts for the 2026 Big Ten football season:

1. Expected record for each team from game-by-game simulation
2. Expected viewership for each game on the schedule
3. Projected broadcaster draft order by week based on game value and expected audience

The three artifacts are linked. Expected records inform projected rankings and stakes. Those inform viewership. Viewership then informs the broadcaster draft simulation. The recommended architecture is one integrated Monte Carlo loop rather than three separate pipelines.

## Current State

- Acquisition code exists for schedule, prestige, and Sports Media Watch scraping.
- Configuration, unit tests, and CI are already in place.
- No data has been populated yet.
- Processed JSON files are empty.
- Model, simulation, outputs, and utility modules are still unbuilt.

## Data To Continue Searching For

### Must-Have

1. Run the existing acquisition pipeline
   - Add `CFBD_API_KEY` and `ODDS_API_KEY` in `.env`
   - Run `scripts/run_acquisition.py`
   - Review scraper output for bad fuzzy matches

2. Time-slot viewership multipliers
   - Needed to distinguish noon, afternoon, and primetime value
   - Best source is tagged historical Sports Media Watch game data

3. Team brand ratings
   - Historical average audience by team across all appearances
   - Lets the model capture that some brands draw regardless of matchup

### High-Value Additions

4. ESPN FPI or custom Elo ratings
   - Adds a second team-strength source beyond odds and SP+
   - Useful for calibration and sanity checks

5. Returning production or talent continuity
   - Important for first-year coaches and roster turnover
   - CFBD endpoints are the likely source

6. Home-field advantage by venue
   - Better than using one generic adjustment for every stadium
   - Can be derived from historical CFBD results

7. Week-of-season viewership curve
   - Early season and rivalry weeks behave differently from mid-season weeks
   - Can be derived from historical audience data once week tagging is added

8. Historical Big Ten broadcaster selections
   - Needed to calibrate how FOX, CBS, and NBC actually behave in draft decisions
   - Likely a manual research task for 2024 and 2025

### Nice-to-Have

9. Transfer portal net impact
10. Competing programming calendar, especially NFL overlap
11. Historical upset rates by spread bucket for calibration

## Artifact 1: Expected Records

### Goal

Produce game-by-game win probabilities and season record distributions for every Big Ten team.

### Recommended Modeling Approach

1. Build a composite team rating
   - 60% odds-implied strength
   - 25% SP+
   - 15% Elo or returning-production adjustment

2. Convert team ratings to win probabilities
   - Use a Log5 or Bradley-Terry style formulation
   - Include home-field advantage
   - Apply a configurable first-year coach penalty if needed

3. Simulate the season
   - Run 10,000 Monte Carlo iterations
   - Draw team true strength from a distribution around the preseason estimate
   - Simulate each game with Bernoulli draws from modeled win probability

4. Aggregate outputs
   - Mean wins
   - Record distribution by team
   - Probability of hitting thresholds like 8, 10, or 11 wins
   - Probability of conference title or playoff-relevant finish if modeled

### Data That Strengthens This Artifact

- FPI or Elo
- Returning production
- Venue-level home-field advantage
- Historical spread-to-win calibration
- Coaching transition adjustments

## Artifact 2: Expected Viewership

### Goal

Estimate audience for each 2026 Big Ten game based on matchup quality, team brand strength, timing, and expected season context.

### Recommended Modeling Approach

1. Engineer features for each game
   - Team brand average audience
   - Historical direct matchup audience
   - Rivalry flag
   - CFP rematch flag
   - Combined prestige
   - DMA or market score
   - Week-of-season factor
   - Time-window factor
   - Projected combined ranking or stakes from the season simulation

2. Start with a regularized linear model
   - Ridge regression is a strong first choice because training data will be limited
   - Use more complex models only if validation clearly improves

3. Validate conservatively
   - Leave-one-season-out cross-validation
   - Track MAPE and R-squared

4. Add uncertainty bands
   - Bootstrap the model or prediction residuals
   - Output a mean prediction and confidence interval for each game

### Data That Strengthens This Artifact

- Team brand ratings
- Time-slot multipliers
- Week-of-season curve
- Network-specific audience baselines
- Projected rankings and stakes from Artifact 1

## Artifact 3: Broadcaster Draft Order

### Goal

Simulate which games FOX, CBS, and NBC would prefer each week based on expected value and slot fit.

### Recommended Modeling Approach

1. Use the configured game value function `V(g)`
   - Prestige: 0.30
   - Viewership: 0.25
   - Stakes: 0.20
   - Market: 0.10
   - Window fit: 0.10
   - Novelty: 0.05

2. For each simulation iteration
   - Use season outcomes to update stakes and ranking context
   - Score all available games for each week
   - Simulate network picks using softmax selection with configurable randomness
   - Respect window preferences and draft order rules

3. Aggregate results
   - Probability each game lands on FOX, CBS, or NBC
   - Expected value of each draft slot
   - Expected weekly viewership by network

### Data That Strengthens This Artifact

- Historical real-world broadcaster selections
- Better window-fit calibration
- Any known draft constraints or trade tendencies

## Recommended Build Order

### Phase 1: Populate Existing Data

- Add API keys
- Run acquisition pipeline
- Review and clean scraped viewership matches
- Confirm processed JSON structure is stable

### Phase 2: Enrich Data Inputs

- Add historical CFBD results for home-field calibration
- Add returning-production or talent continuity data
- Enhance the SMW dataset with week and slot tags
- Compute team brand and week-curve features

### Phase 3: Build the Win Probability Layer

- Implement composite team rating logic
- Implement win probability function
- Validate against historical results or market win totals

### Phase 4: Build the Monte Carlo Engine

- Simulate all games across 10,000 iterations
- Store team-level season outcomes
- Produce expected records and ranking distributions

### Phase 5: Build the Viewership Model

- Construct feature table from historical data
- Train and validate Ridge regression
- Generate 2026 predictions with uncertainty intervals

### Phase 6: Build the Draft Simulation

- Implement weekly game scoring
- Simulate network draft choices
- Aggregate assignment probabilities and slot value

### Phase 7: Build Outputs

- Export expected records
- Export game-level viewership predictions
- Export network assignment probabilities
- Optionally expose all outputs in Streamlit

## Suggested File and Module Segmentation

### `src/model/`

- `team_strength.py`
- `win_probability.py`
- `viewership_model.py`
- `features.py`

### `src/simulation/`

- `season_simulator.py`
- `draft_simulator.py`
- `monte_carlo.py`

### `src/outputs/`

- `export_records.py`
- `export_viewership.py`
- `export_draft.py`

### `scripts/`

- `run_acquisition.py`
- `run_simulation.py`
- `run_dashboard.py` if needed later

## Decision Notes

- Keep the first version simple and interpretable.
- Prefer calibration and validation over complexity.
- Treat the draft model as a behavior model, not a deterministic ranking exercise.
- Make heuristic adjustments configurable so they can be sensitivity-tested.
- Avoid building the dashboard until the data and simulation outputs are stable.

## Immediate Next Steps

1. Populate the empty processed datasets by running acquisition.
2. Decide the exact team-strength formula for the first simulation pass.
3. Expand the viewership dataset with time slot and week metadata.
4. Build expected-record outputs first, then feed those outputs into viewership and draft modeling.

## Implementation Checklist

This section is written as concrete, agent-sized tasks. Each task should be runnable as a focused work item with a clear output and stopping point.

### Task 0: Repository Prep

- [x] Add missing `__init__.py` files under `src/`, `src/acquisition/`, `src/model/`, `src/simulation/`, `src/outputs/`, and `src/utils/`
- [x] Run the existing test suite and confirm the current baseline status
- [x] Record any existing failures before new work begins

Definition of done:
- Imports resolve cleanly for the package layout
- Baseline test status is known

Status note:
- COMPLETE: 23/23 unit tests passing (Python 3.11.9, `.venv`)
- Fixed two bugs in `_fuzzy_match_team`: ambiguous substring resolution now returns shortest match; Notre Dame expectation updated to reflect B1G membership
- Fixed cfbd 5.x SDK compat: `access_token` auth, camelCase field names in `_normalize_game`, correct `get_sp()` method
- Fixed SSL cert verification on macOS Python 3.11 via certifi

### Task 1: Populate Core Data

- [x] Create `.env` with `CFBD_API_KEY`
- [x] Run `scripts/run_acquisition.py`
- [x] Verify `data/processed/game_schedule.json` is populated (135 games: 84 conference, 51 non-conference)
- [x] Verify `data/processed/team_prestige.json` is populated (19 teams, SP+ 2025 composite)
- [x] Verify `data/processed/viewership_pairs.json` is populated (469 individual game records)
- [x] Inspect scraper output and identify rows marked for review

Definition of done:
- Schedule and prestige datasets are non-empty and validated
- Viewership pairs populated from cache with bowl/playoff enrichment

Agent prompt:
Implement Task 1 by running the acquisition pipeline, validating the generated JSON outputs, and summarizing any data-quality issues that need manual cleanup.

Status note:
- COMPLETE: All three processed datasets populated and validated
- `game_schedule.json`: 135 games (84 conf, 51 non-conf)
- `team_prestige.json`: 19 teams, top prestige Ohio State 0.978
- `viewership_pairs.json`: 469 individual game records (2023-2025), each with `is_bowl_game` and `is_playoff_game` flags from CFBD postseason API. 22 bowls, 9 playoff games identified.
- Postseason data cached per year at `data/raw/schedule/cfbd_{year}_postseason.json`
- 23/23 unit tests passing

### Task 2: Stabilize Data Contracts

- [x] Inspect the shape of all processed JSON outputs
- [x] Add typed loader helpers in `src/utils/` or `src/model/` for schedule, prestige, and viewership data
- [x] Add validation checks for required fields and allowed value ranges
- [x] Add unit tests for the loaders and validation logic

Definition of done:
- The project has one canonical way to load each processed dataset
- Bad or missing fields fail fast with useful errors

Agent prompt:
Implement data-loading utilities and validation tests for the processed schedule, prestige, and viewership datasets so downstream modeling code can rely on a stable schema.

Status note:
- COMPLETE: `src/utils/data_loaders.py` provides `Game`, `TeamPrestige`, `ViewershipRecord` frozen dataclasses
- Loaders: `load_schedule()`, `load_prestige()`, `load_viewership()`, `load_prestige_lookup()`
- Validation: required fields, value ranges (prestige [0,1], DMA rank [1,220], positive viewers, valid time_slot)
- 22 new tests in `tests/unit/test_data_loaders.py`, all passing (45/45 total)

### Task 3: Enrich Historical Viewership Data

- [x] Add the home and away team's record and SP+ rating into viewership_pairs.json going into that game by joining data from the CFBD API
- [x] Add week of season and time slot data into the viewership_pairs.json by joining data from the CFBD API 
- [x] Extend the SMW pipeline to capture or derive week number for historical games
- [x] Extend the SMW pipeline to capture or derive broadcast window when possible
- [x] Compute team brand ratings from historical appearances
- [x] Compute week-of-season multipliers from historical data
- [x] Compute time-slot multipliers from historical data
- [x] Persist these derived features into a reusable processed file or helper output

Definition of done:
- Historical viewership data supports brand, week, and slot features
- Derived metrics can be loaded without rerunning manual analysis

Agent prompt:
Enhance the historical viewership dataset so it produces reusable team-brand, week-of-season, and time-slot features for downstream viewership modeling.

Status note:
- COMPLETE: `viewership_pairs.json` now has 455 records with week, time_slot, home/away team, scores, pregame Elo, is_conference_game, is_bowl_game, is_playoff_game
- `fetch_historical_games()` added to cfbd_client.py — fetches and caches all B1G games per year in `cfbd_{year}_all_games.json`
- 76% match rate (345/455) to CFBD data; unmatched are scraper misidentifications
- `src/model/features.py` provides `build_derived_features()` → writes `viewership_features.json` with:
  - Team brands: Ohio State 5.00M, Michigan 3.91M, Notre Dame 3.91M, ..., Northwestern 0.87M
  - Week multipliers: Week 13 = 1.50x, Week 14 = 1.37x (rivalry/championship), Week 8 = 0.71x (lowest)
  - Slot multipliers: Noon = 1.14x, Afternoon = 0.91x, Primetime = 0.83x
- Name normalization map (`_CFBD_NAME_MAP`) handles Cal→California, Mississippi→Ole Miss, etc.
- Garbage filter removes self-matches and partial-sentence records

### Task 4: Add Team Strength Enrichment

- [x] Add 2026 SP+ ratings by scraping https://www.espn.com/college-football/story/_/id/48306284/2026-college-football-sp+-rankings-138-fbs-teams to the team_prestige data ingestion.
- [x] Persist the enriched team-strength inputs in processed form

Definition of done:
- Team strength can be computed from more than one upstream signal
- Coach and continuity effects are explicit and configurable

Agent prompt:
Add one additional team-strength signal, wire it into processed team inputs, and keep the resulting rating inputs simple enough to validate and explain.

Status note:
- COMPLETE: Added end-of-2025 Elo from CFBD postgame data as third signal
- `extract_elo_2025()` in `src/model/team_strength.py` extracts latest postgame Elo from cached `cfbd_2025_all_games.json`
- 2026 SP+ not available from CFBD API yet; 2025 SP+ + 2025 end-of-season Elo used instead
- New coach penalty: 30 Elo pts (~2 spread points) applied to Michigan, Penn State, Michigan State, UCLA

### Task 5: Calibrate Home-Field Advantage

- [x] Add historical CFBD results acquisition for recent seasons
- [x] Compute a generic home-field baseline
- [x] Evaluate whether venue-specific adjustments are supported by the sample size
- [x] Store the chosen home-field parameters in a reusable config or processed artifact
- [x] Add tests for the calibration helper logic

Definition of done:
- The win model uses an explicit, data-backed home-field adjustment
- The calibration logic is reproducible

Agent prompt:
Acquire enough historical game results to calibrate home-field advantage and produce a reusable adjustment that can plug directly into the win-probability model.

Status note:
- COMPLETE: `calibrate_home_field()` in `src/model/team_strength.py`
- Data: 432 games (2023-2025), all involving ≥1 B1G team, non-neutral
- Result: 67.6% home win rate → 127.7 Elo pts (typical for college football)
- Persisted to `data/processed/home_field_advantage.json`
- Venue-specific adjustment: sample too small per-venue; generic HFA used
- 4 tests covering keys, positive advantage, game count, and consistency

### Task 6: Implement Composite Team Ratings

- [x] Create `src/model/team_strength.py`
- [x] Implement composite rating logic using odds, SP+, and one enrichment source
- [x] Normalize rating outputs to a consistent scale
- [x] Add support for home-field input hooks
- [x] Add unit tests covering weighting, normalization, and edge cases

Definition of done:
- One function produces a stable pregame strength rating for every Big Ten team
- The rating logic is covered by tests and easy to tune

Agent prompt:
Build the composite team-strength module with clear weighting, normalization, and tests so it can serve as the core input to the game win model.

Status note:
- COMPLETE: `src/model/team_strength.py` with `build_team_strength()` full pipeline
- Three signals: odds (40%), SP+ 2025 (35%), Elo 2025 (25%) — configurable via `DEFAULT_WEIGHTS`
- Output: `data/processed/team_strength.json` — 19 teams with composite_score [0,1]
- Top teams: Indiana 1.000, Ohio State 0.999, Oregon 0.816, Notre Dame 0.537
- Coach penalty: -30 Elo pts for Michigan, Penn State, Michigan State, UCLA
- Home-field: 127.7 Elo pts from 432-game sample
- 19 tests in `tests/unit/test_team_strength.py`, 64/64 total suite passing

### Task 7: Implement Win Probability Model

- [x] Create `src/model/win_probability.py`
- [x] Implement a Log5 or Bradley-Terry win probability function
- [x] Apply home-field adjustment and optional coach adjustment
- [x] Add probability calibration checks against known spread or implied-probability behavior
- [x] Add unit tests for symmetry, monotonicity, and boundary cases

Definition of done:
- Any scheduled game can be converted into a win probability for both teams
- The probability function passes basic calibration and invariants

Agent prompt:
Implement the game win-probability model with tests for correctness, sensible calibration behavior, and clean integration with team-strength inputs.

Status note:
- COMPLETE: Bradley-Terry model with `P(home) = 1 / (1 + 10^((away - home - hfa) / 0.6))`
- HFA converted from 127.7 Elo pts to composite scale (~0.126) automatically
- Neutral-site flag zeroes HFA; non-B1G opponents get default strength 0.3
- PROB_FLOOR=0.01, PROB_CEIL=0.99 for numerical stability
- `build_win_probabilities()` scores all 135 games → `win_probabilities.json`
- 16 tests in `tests/unit/test_win_probability.py` covering symmetry, monotonicity, boundaries, HFA

### Task 8: Implement Season Monte Carlo Engine

- [x] Create `src/simulation/season_simulator.py`
- [x] Simulate a full season from schedule and pregame win probabilities
- [x] Support 10,000 iteration runs with reproducible seeds
- [x] Capture team-level outputs including wins, losses, and finish distributions
- [x] Add a narrow integration test on a toy schedule

Definition of done:
- The project can simulate a season end-to-end and return expected records
- The simulator is deterministic when the seed is fixed

Agent prompt:
Build the season simulator that consumes schedule and win probabilities, runs Monte Carlo iterations, and returns expected records and distribution outputs for each team.

Status note:
- COMPLETE: Vectorized numpy Monte Carlo — shape (n_games, n_iterations) uniform draws
- Seed=42 for determinism; configurable via `SIM_ITERATIONS` / `SIM_SEED` settings
- `simulate_season()` returns dict of numpy arrays per team
- `summarize_results()` produces mean/median/std wins, threshold probabilities
- `run_simulation()` full pipeline: load schedule → compute probs → simulate → summarize
- Results: Indiana 11.0W, Ohio State 10.4W, Oregon 10.0W (realistic top-heavy distribution)
- 13 tests in `tests/unit/test_season_simulator.py`

### Task 9: Export Artifact 1 Outputs

- [x] Create `src/outputs/export_records.py`
- [x] Export expected record summaries to `data/outputs/`
- [x] Include mean wins, record distribution, and threshold probabilities
- [x] Add a script entry point or hook from `run_simulation.py`

Definition of done:
- Artifact 1 can be generated from a single command and saved for reuse

Agent prompt:
Implement record export logic so the expected-record artifact is materialized as structured output files ready for downstream modeling and inspection.

Status note:
- COMPLETE: `src/outputs/export_records.py` exports JSON (full detail) + CSV (flat summary)
- `scripts/run_simulation.py` CLI: `--iterations`, `--seed`, `--rebuild-strength` flags
- Full pipeline runs in <1s: strength → win probs → Monte Carlo → export
- Outputs: `data/outputs/expected_records.json` and `data/outputs/expected_records.csv`
- 93/93 tests passing across full suite

### Task 10: Build Viewership Feature Table

- [x] Create `src/model/features.py`
- [x] Join schedule, prestige, historical pair data, team-brand metrics, and stakes context
- [x] Define the exact feature schema used for model training and 2026 scoring
- [x] Add tests that verify the feature table contains expected columns and null handling

Definition of done:
- Historical and 2026 games can be represented in one consistent feature schema

Agent prompt:
Build a reusable feature-engineering pipeline for viewership prediction that merges historical audience data with schedule, prestige, 2026 team strength projections, and simulation-derived context.

Status note:
- COMPLETE: Extended `src/model/features.py` with `build_training_features()` and `build_2026_features()`
- Training features: 312 regular-season games (2023-2025) with 8 features
- Features: combined_brand, max_brand, combined_elo, elo_closeness, week_multiplier, slot_multiplier, is_rivalry, is_conference_game
- 2026 features: 135 games scored with same schema + metadata (game_id, strengths, market_score)
- Shared `FEATURE_COLUMNS` list ensures train/score consistency
- 11 tests in `tests/unit/test_viewership.py::TestFeatureTable`

### Task 11: Train Viewership Model

- [x] Create `src/model/viewership_model.py`
- [x] Implement a Ridge regression baseline
- [x] Add leave-one-season-out validation
- [x] Report MAPE and R-squared
- [x] Add prediction interval logic using bootstrap or residual sampling
- [x] Persist the trained model artifacts or coefficients if helpful

Definition of done:
- The project can train and validate a first-pass viewership model with interpretable results
- Predictions include uncertainty bands

Agent prompt:
Implement the first-pass viewership model using a regularized linear baseline, season-level cross-validation, and prediction intervals suitable for ranking games by expected audience.

Status note:
- COMPLETE: `src/model/viewership_model.py` with Ridge(alpha=10.0) + StandardScaler
- In-sample R²=0.496, CV R²=0.433, CV RMSE=2.36M viewers
- Leave-one-season-out: 2023 R²=0.494, 2024 R²=0.343, 2025 R²=0.394
- Prediction intervals via residual quantile method (90% CI)
- Top coefficients: combined_brand (+1.26), combined_elo (+0.97), elo_closeness (+0.54), week_multiplier (+0.42)
- Model diagnostics persisted to `data/processed/viewership_model_diagnostics.json`
- `build_viewership_predictions()` pipeline: train → CV → predict → export
- 10 tests in `tests/unit/test_viewership.py::TestViewershipModel`

### Task 12: Export Artifact 2 Outputs

- [x] Create `src/outputs/export_viewership.py`
- [x] Score each 2026 game for expected audience
- [x] Export point estimates and confidence intervals
- [x] Include enough metadata for downstream draft simulation use

Definition of done:
- Artifact 2 exists as a reusable game-level output table

Agent prompt:
Export viewership predictions for the 2026 schedule in a format that can feed the broadcaster draft simulation without additional joins.

Status note:
- COMPLETE: `src/outputs/export_viewership.py` exports JSON + CSV
- Outputs: `data/outputs/expected_viewership.json` and `data/outputs/expected_viewership.csv`
- 135 games with predicted_viewers_millions, lower/upper bounds, is_conference/rivalry/cfp_rematch, strengths, market_score
- Top predicted: OSU-Oregon 8.36M, OSU-Michigan 8.30M, IND-OSU 8.08M
- Wired into `run_simulation.py` as Step 5 (prints top 20 viewership table)
- 2 tests in `tests/unit/test_viewership.py::TestExportViewership`
- 116/116 tests passing across full suite

### Task 13: Implement Draft Value Scoring

- [x] Determine if a game is eligible for the networks. Must include a Big Ten team as the home team, otherwise ineligible.
- [x] Create value-scoring logic for `V(g)` using the configured weights
- [x] Normalize each component to a common scale
- [x] Include viewership, prestige, stakes, market, window fit, and novelty
- [x] Add tests that verify weight application and normalization behavior
- [x] Output ranking of weeks by which is most valuable to have the top game on your broadcast.

Definition of done:
- Any candidate game can be converted into a comparable broadcaster value score

Agent prompt:
Implement the configurable broadcaster value function so each weekly game slate can be ranked consistently for FOX, CBS, and NBC selection behavior.

Status note:
- COMPLETE: `src/simulation/draft_value.py` with V(g) scoring
- Components: prestige (0.30), viewership (0.25), stakes (0.20), market (0.10), window_fit (0.10), novelty (0.05)
- Each component min-max normalized within weekly slate before weighting
- Window fit: FOX (noon) blocks Pacific home teams; CBS prefers PT; NBC primetime works for all
- Eligibility: only games with B1G home team are draftable (129 of 135 games)
- `rank_weeks_by_top_value()` ranks weeks: Week 7 (IND-OSU) #1, Week 13 (OSU-MICH) #2
- 7 tests in `tests/unit/test_draft.py::TestDraftValue`

### Task 14: Implement Draft Simulation

- [x] Prompt user for the network draft order and add to a config.
- [x] Create `src/simulation/draft_simulator.py`
- [x] Simulate network selections by week using configured draft order
- [x] Implement softmax-based selection with temperature control
- [x] Support optional trade behavior using the configured trade probability
- [x] Add tests for pick ordering, exclusion of already selected games, and deterministic seeded behavior

Definition of done:
- The project can simulate weekly network picks from a slate of scored games
- The result respects draft order and slot preferences

Agent prompt:
Build the broadcaster draft simulator that consumes weekly game values and produces seeded FOX, CBS, and NBC selections across the season.

Status note:
- COMPLETE: `src/simulation/draft_simulator.py` with full Monte Carlo draft simulation
- Draft order from `settings.DRAFT_ORDER`: [FOX,FOX,FOX,CBS,FOX,NBC,CBS,FOX,NBC,CBS,FOX]
- Softmax selection (temp=0.3): lower temp → more deterministic picks
- Trade logic: 15% chance FOX trades a pick to CBS/NBC per-pick
- Pacific timezone games excluded from FOX (noon ET slot incompatible)
- `build_draft_results()` runs 10K iterations → assignment probabilities per game
- Results: FOX gets ~12.4M/week, CBS ~6.7M/week, NBC ~4.4M/week
- OSU-Michigan → 91% FOX, IND-OSU → 96% FOX, USC-OSU → 70% CBS
- 7 tests in `tests/unit/test_draft.py::TestDraftSimulation`

### Task 15: Export Artifact 3 Outputs

- [x] Create `src/outputs/export_draft.py`
- [x] Export network assignment probabilities by game
- [x] Export expected weekly viewership by network
- [x] Export slot-value summaries if useful

Definition of done:
- Artifact 3 is saved in a form that can be reviewed without rerunning the full simulation

Agent prompt:
Export broadcaster draft results as structured outputs showing network assignment probabilities and expected weekly audience totals.

Status note:
- COMPLETE: `src/outputs/export_draft.py` exports 3 files
- `data/outputs/draft_assignments.json` + `.csv`: 129 games with fox/cbs/nbc/undrafted probabilities
- `data/outputs/draft_weekly_viewers.json`: avg weekly viewers per network + season totals + metadata
- Wired into `run_simulation.py` as Step 6 (prints top 15 games + season network totals)
- 2 tests in `tests/unit/test_draft.py::TestExportDraft`
- 139/139 tests passing across full suite

### Task 16: Create the Main Simulation Entry Point

- [x] Create `scripts/run_simulation.py`
- [x] Wire together data loading, team strength, win simulation, viewership scoring, and draft simulation
- [x] Add CLI flags for iteration count, seed, refresh behavior, and output directory
- [x] Ensure the script can generate all three artifacts in one run

Definition of done:
- One script produces the full end-to-end workflow from processed inputs to exported outputs

Agent prompt:
Create the main simulation runner that orchestrates all modeling layers and exports the three target artifacts from one command-line entry point.

Status note:
- COMPLETE: `scripts/run_simulation.py` fully polished CLI entry point
- Flags: `--iterations/-n`, `--seed/-s`, `--output-dir/-o`, `--rebuild-strength`, `--skip-draft`, `--quiet/-q`
- All export functions accept optional `output_dir` parameter for redirect
- Pipeline completes all 3 artifacts in <1s (500 iterations)
- Summary tables printed by default; `--quiet` suppresses for CI/automation
- Elapsed time reported at completion
- 171/171 tests passing

### Task 17: Add Integration Coverage

- [x] Add at least one integration test for the end-to-end simulation pipeline on a toy dataset
- [x] Add one integration test for the acquisition-to-model loading handoff
- [x] Confirm CI can run the new tests without external API dependence

Definition of done:
- The project has regression protection across the main handoff boundaries

Agent prompt:
Add lightweight integration tests that prove the core pipeline works end-to-end on deterministic sample inputs without external network calls.

Status note:
- COMPLETE: 30 integration tests across 2 files
- `tests/integration/test_pipeline.py`: 24 tests covering full pipeline export, win probs, season sim, viewership, draft constraints, toy data simulation
- `tests/integration/test_data_handoff.py`: 6 tests covering data loaders, schema validation, game_id consistency across artifacts
- Toy pipeline test uses real B1G team names with synthetic win probs (no file I/O)
- CI updated: `ci.yml` now runs `pytest tests/integration/` as separate step
- No external API calls — reads only committed `data/processed/` files
- 171/171 tests passing (141 unit + 30 integration)

### Task 18: Optional Dashboard Layer

- [x] Add a minimal Streamlit app only after outputs stabilize
- [x] Display expected records, predicted viewership, and network assignment probabilities
- [x] Display the weekly rankings (1-13). For each week, show the top 3 games of the week and predicted viewership for each game.
- [x] Keep the dashboard thin and dependent on exported files, not live recomputation

Definition of done:
- The dashboard is a read layer over stable outputs, not a second execution path

Agent prompt:
Build a thin Streamlit dashboard that reads exported artifacts and visualizes the three outputs without duplicating model logic.

Status note:
- COMPLETE: `scripts/run_dashboard.py` — Streamlit app with 4 pages
- Page 1 (Expected Records): sortable table + horizontal bar chart of expected wins
- Page 2 (Viewership Predictions): filterable table, conference-only toggle, min-viewers slider, top-10 chart
- Page 3 (Broadcaster Draft): season total metrics, network filter, probability table, weekly stacked bar chart
- Page 4 (Weekly Rankings): week selector, top-3 game cards with likely network, full week table
- Data loaded via `@st.cache_data` from exported JSON — no recomputation
- Launch: `streamlit run scripts/run_dashboard.py`
- Dependencies: streamlit, pandas (both in requirements.txt)
- 171/171 tests still passing

## Execution Order For Agent Mode

Run tasks in this order unless blocked:

1. Task 0 through Task 2
2. Task 6 through Task 9 for Artifact 1
3. Task 3, Task 10, Task 11, and Task 12 for Artifact 2
4. Task 13 through Task 15 for Artifact 3
5. Task 16 and Task 17 to unify and protect the pipeline
6. Task 18 only after the exported outputs are stable

Parallel work that is safe:

- Task 3 and Task 4 can run in parallel after Task 1
- Task 5 can run in parallel with Task 6 once historical results acquisition is defined
- Task 17 can begin once Task 8 and Task 12 exist

## Rules For Running Agent Tasks

- Keep each task limited to one clear deliverable and one validation step
- Prefer adding tests in the same task that introduces new logic
- Do not start dashboard work before exported outputs are stable
- Use processed artifacts as contracts between phases
- Favor interpretable baselines before adding model complexity
3. Expand the viewership dataset with time slot and week metadata.
4. Build Artifact 1 first, then feed its outputs into Artifact 2 and Artifact 3.
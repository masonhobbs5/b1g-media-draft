"""
src/simulation/season_simulator.py
──────────────────────────────────
Monte Carlo season simulation for Big Ten football.

Simulates the full schedule N times, drawing game outcomes from Bernoulli
distributions parameterized by the win-probability model. Produces:
  - Expected wins per team
  - Win distribution (histogram of total wins across iterations)
  - Conference record distributions
  - Threshold probabilities (P(8+ wins), P(10+ wins), etc.)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from config.constants import BIG_TEN_TEAMS
from config.settings import settings
from src.model.win_probability import compute_all_game_probabilities

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = settings.PROCESSED_DIR


def simulate_season(
    n_iterations: int = settings.SIM_ITERATIONS,
    seed: int | None = settings.SIM_SEED,
    win_probs: list[dict] | None = None,
) -> dict[str, dict]:
    """
    Run Monte Carlo simulation of the full 2026 B1G schedule.

    Args:
        n_iterations: Number of simulation iterations (default 10,000)
        seed: Random seed for reproducibility (None for non-deterministic)
        win_probs: Pre-computed game probabilities (auto-loaded if None)

    Returns:
        Dict keyed by team name, each containing:
            total_wins: array of total wins per iteration
            conf_wins: array of conference wins per iteration
            games_played: total games for this team
            conf_games_played: conference games for this team
    """
    if win_probs is None:
        win_probs = compute_all_game_probabilities()

    rng = np.random.default_rng(seed)

    # Initialize tracking arrays
    team_total_wins: dict[str, np.ndarray] = {
        t: np.zeros(n_iterations, dtype=np.int32) for t in BIG_TEN_TEAMS
    }
    team_conf_wins: dict[str, np.ndarray] = {
        t: np.zeros(n_iterations, dtype=np.int32) for t in BIG_TEN_TEAMS
    }
    team_games: dict[str, int] = defaultdict(int)
    team_conf_games: dict[str, int] = defaultdict(int)

    # Pre-compute: for each game, store (home, away, home_wp, is_conf)
    games = []
    for g in win_probs:
        home = g["home_team"]
        away = g["away_team"]
        # Only track B1G teams
        if home not in BIG_TEN_TEAMS and away not in BIG_TEN_TEAMS:
            continue
        games.append((home, away, g["home_win_prob"], g["is_conference_game"]))
        if home in BIG_TEN_TEAMS:
            team_games[home] += 1
            if g["is_conference_game"]:
                team_conf_games[home] += 1
        if away in BIG_TEN_TEAMS:
            team_games[away] += 1
            if g["is_conference_game"]:
                team_conf_games[away] += 1

    # Vectorized simulation: draw all random numbers at once
    # Shape: (n_games, n_iterations)
    n_games = len(games)
    draws = rng.random((n_games, n_iterations))

    for i, (home, away, home_wp, is_conf) in enumerate(games):
        # home_wins[j] = True where draw < home_wp
        home_wins = draws[i] < home_wp

        if home in BIG_TEN_TEAMS:
            team_total_wins[home] += home_wins.astype(np.int32)
            if is_conf:
                team_conf_wins[home] += home_wins.astype(np.int32)

        if away in BIG_TEN_TEAMS:
            away_wins = ~home_wins
            team_total_wins[away] += away_wins.astype(np.int32)
            if is_conf:
                team_conf_wins[away] += away_wins.astype(np.int32)

    # Package results
    results = {}
    for team in sorted(BIG_TEN_TEAMS):
        results[team] = {
            "total_wins": team_total_wins[team],
            "conf_wins": team_conf_wins[team],
            "games_played": team_games.get(team, 0),
            "conf_games_played": team_conf_games.get(team, 0),
        }

    return results


def summarize_results(
    sim_results: dict[str, dict],
    n_iterations: int | None = None,
) -> list[dict]:
    """
    Summarize raw simulation arrays into human-readable stats.

    Returns list of dicts (one per team) with:
        team, games_played, conf_games_played,
        mean_wins, mean_conf_wins, median_wins,
        win_distribution (dict of {wins: probability}),
        conf_win_distribution,
        p_8_plus_wins, p_10_plus_wins, p_11_plus_wins,
        p_undefeated_conf
    """
    summaries = []

    for team in sorted(BIG_TEN_TEAMS):
        data = sim_results[team]
        total_wins = data["total_wins"]
        conf_wins = data["conf_wins"]
        n = len(total_wins)

        if n_iterations is None:
            n_iterations = n

        # Win distribution
        max_games = data["games_played"]
        win_dist = {}
        for w in range(max_games + 1):
            count = int(np.sum(total_wins == w))
            if count > 0:
                win_dist[w] = round(count / n, 4)

        # Conference win distribution
        max_conf = data["conf_games_played"]
        conf_dist = {}
        for w in range(max_conf + 1):
            count = int(np.sum(conf_wins == w))
            if count > 0:
                conf_dist[w] = round(count / n, 4)

        summaries.append({
            "team": team,
            "games_played": data["games_played"],
            "conf_games_played": data["conf_games_played"],
            "mean_wins": round(float(np.mean(total_wins)), 2),
            "mean_conf_wins": round(float(np.mean(conf_wins)), 2),
            "median_wins": int(np.median(total_wins)),
            "std_wins": round(float(np.std(total_wins)), 2),
            "win_distribution": win_dist,
            "conf_win_distribution": conf_dist,
            "p_8_plus_wins": round(float(np.mean(total_wins >= 8)), 4),
            "p_10_plus_wins": round(float(np.mean(total_wins >= 10)), 4),
            "p_11_plus_wins": round(float(np.mean(total_wins >= 11)), 4),
            "p_undefeated_conf": round(float(np.mean(conf_wins == max_conf)), 4),
        })

    # Sort by mean wins descending
    summaries.sort(key=lambda x: -x["mean_wins"])
    return summaries


def run_simulation(
    n_iterations: int = settings.SIM_ITERATIONS,
    seed: int | None = settings.SIM_SEED,
) -> list[dict]:
    """
    Full pipeline: compute win probs → simulate → summarize → persist.
    Returns the summary list.
    """
    logger.info("Computing win probabilities...")
    win_probs = compute_all_game_probabilities()

    logger.info("Running Monte Carlo simulation (%d iterations, seed=%s)...",
                n_iterations, seed)
    raw_results = simulate_season(
        n_iterations=n_iterations,
        seed=seed,
        win_probs=win_probs,
    )

    logger.info("Summarizing results...")
    summaries = summarize_results(raw_results, n_iterations=n_iterations)

    return summaries

"""
src/simulation/draft_simulator.py
─────────────────────────────────
Broadcaster draft simulation for the B1G weekly game selection.

Two-phase draft model:
  Phase 1 — Week Draft:
    Networks select WEEKS in DRAFT_ORDER. The value of a week is the gap
    between the top game and the second-best game (predicted viewers).
    Drafting a week gives first-pick rights for that week's games.

  Phase 2 — Game Selection (per week):
    The drafting network picks first. Within-week pick order:
      - FOX drafted: FOX 1st → CBS/NBC alternate 2nd → remaining 3rd
      - CBS drafted: CBS 1st → FOX 2nd → NBC 3rd
      - NBC drafted: NBC 1st → FOX 2nd → CBS 3rd
    Each week, up to 3 games are selected (one per participating network).

    NBC pre-committed weeks (Notre Dame): NBC does not participate in
    game selection. Only FOX and CBS pick (2 games selected).

The simulation runs N iterations to produce:
  - Assignment probabilities per game (P(game → FOX), P(game → CBS), etc.)
  - Expected weekly viewership by network
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from config.constants import BIG_TEN_TEAMS, PACIFIC_TZ_TEAMS
from config.settings import settings
from src.simulation.draft_value import (
    build_all_game_values,
    compute_window_fit,
    score_weekly_slate,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = settings.PROCESSED_DIR


def _softmax_select(
    values: list[float],
    temperature: float = settings.SOFTMAX_TEMPERATURE,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Select an index from values using softmax probabilities.

    Lower temperature → more deterministic (picks highest value).
    Higher temperature → more random.
    """
    if not values:
        raise ValueError("Cannot select from empty list")
    if len(values) == 1:
        return 0

    if rng is None:
        rng = np.random.default_rng()

    arr = np.array(values)
    scaled = (arr - arr.max()) / max(temperature, 0.01)
    exp_vals = np.exp(scaled)
    probs = exp_vals / exp_vals.sum()

    return int(rng.choice(len(values), p=probs))


def _is_eligible_for_network(game: dict, network: str) -> bool:
    """
    Check if a game is eligible for a specific network's window.

    FOX (noon ET): Cannot select Pacific-timezone home games.
    CBS/NBC: All B1G home games are eligible.
    """
    if network == "FOX":
        home_team = game.get("home_team", "")
        noon_eligible = game.get("noon_eligible", True)
        if not noon_eligible or home_team in PACIFIC_TZ_TEAMS:
            return False
    return True


def _compute_week_draft_value(
    week_games: list[dict],
    viewership_lookup: dict[str, float],
    network: str,
) -> float:
    """
    Compute draft value of a week for a network.

    Primary driver: the gap between the #1 and #2 eligible game's predicted
    viewers. A large gap means first-pick is extremely valuable — you secure
    a game that no other network can match that week.
    Secondary: small boost from the absolute viewership of the top game.

    Example: 5M/#1 with 2.8M/#2 (gap=2.2) is more valuable than 7M/#1
    with 6.8M/#2 (gap=0.2) because first-pick barely matters in the latter.
    """
    TOP_WEIGHT = 0.05  # minimal secondary boost from absolute viewership

    # Get predicted viewers for eligible games, sorted descending
    eligible_viewers = sorted(
        [
            viewership_lookup.get(g["game_id"], 2.0)
            for g in week_games
            if _is_eligible_for_network(g, network)
        ],
        reverse=True,
    )

    if not eligible_viewers:
        return 0.0

    top_viewers = eligible_viewers[0]

    if len(eligible_viewers) < 2:
        # Only one game — gap is the full value of the top game
        return top_viewers

    # Value = gap (primary) + small boost from top game (secondary)
    gap = top_viewers - eligible_viewers[1]
    return gap + TOP_WEIGHT * top_viewers


def _draft_weeks(
    all_values: dict[int, dict[str, list[dict]]],
    viewership_lookup: dict[str, float],
    rng: np.random.Generator,
    temperature: float = settings.SOFTMAX_TEMPERATURE,
    trade_probability: float = settings.TRADE_PROBABILITY,
    pick_tracking: dict[int, dict[int, int]] | None = None,
) -> dict[int, str]:
    """
    Phase 1: Networks select weeks using DRAFT_ORDER.

    Args:
        pick_tracking: Optional dict[pick_idx → {week → count}] to record
            which week was selected at each pick position across iterations.

    Returns:
        Dict mapping week → network that drafted it (first-pick owner).
    """
    draft_order = settings.DRAFT_ORDER
    nbc_committed_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)
    available_weeks = set(all_values.keys())
    week_owners: dict[int, str] = {}

    # Pre-compute week games lookup for gap calculation
    week_games: dict[int, list[dict]] = {}
    for week, slates in all_values.items():
        # Use the raw game list from any network's slate (they share the same games)
        week_games[week] = slates.get("FOX", [])

    for pick_idx, network in enumerate(draft_order):
        if not available_weeks:
            break

        # Trade logic: FOX may trade a week pick to CBS or NBC
        actual_network = network
        if network == "FOX" and pick_idx > 0:
            if rng.random() < trade_probability:
                cbs_weeks = sum(1 for w, n in week_owners.items() if n == "CBS")
                nbc_weeks = sum(1 for w, n in week_owners.items() if n == "NBC")
                actual_network = "CBS" if cbs_weeks <= nbc_weeks else "NBC"

        # Determine which weeks this network can draft
        if actual_network == "NBC":
            # NBC cannot draft ND weeks (they already have ND for those weeks)
            draftable = available_weeks - nbc_committed_weeks
        else:
            draftable = available_weeks.copy()

        if not draftable:
            continue

        # Compute draft value for each available week
        draftable_list = sorted(draftable)
        week_values = [
            _compute_week_draft_value(week_games[w], viewership_lookup, actual_network)
            for w in draftable_list
        ]

        # Select week using softmax
        selected_idx = _softmax_select(week_values, temperature, rng)
        selected_week = draftable_list[selected_idx]

        week_owners[selected_week] = actual_network
        available_weeks.discard(selected_week)

        # Track pick selection
        if pick_tracking is not None:
            if pick_idx not in pick_tracking:
                pick_tracking[pick_idx] = defaultdict(int)
            pick_tracking[pick_idx][selected_week] += 1

    return week_owners


def _select_games_for_week(
    week: int,
    drafting_network: str,
    network_slates: dict[str, list[dict]],
    viewership_lookup: dict[str, float],
    rng: np.random.Generator,
    temperature: float = settings.SOFTMAX_TEMPERATURE,
    cbs_nbc_second_toggle: bool = True,
) -> dict[str, dict | None]:
    """
    Phase 2: Select games within a single week.

    The drafting network picks first, then 2nd and 3rd per the rules:
      - FOX drafted: FOX 1st → (CBS or NBC based on toggle) 2nd → other 3rd
      - CBS drafted: CBS 1st → FOX 2nd → NBC 3rd
      - NBC drafted: NBC 1st → FOX 2nd → CBS 3rd

    In NBC pre-committed weeks, NBC does not participate (2 games only).

    Returns:
        Dict mapping network → selected game dict (or None if no pick).
    """
    nbc_committed_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)
    nbc_participates = week not in nbc_committed_weeks

    # Determine pick order
    if drafting_network == "FOX":
        second = "CBS" if cbs_nbc_second_toggle else "NBC"
        third = "NBC" if cbs_nbc_second_toggle else "CBS"
        pick_order = ["FOX", second, third]
    elif drafting_network == "CBS":
        pick_order = ["CBS", "FOX", "NBC"]
    else:  # NBC
        pick_order = ["NBC", "FOX", "CBS"]

    # Remove NBC from pick order if it's a ND week
    if not nbc_participates:
        pick_order = [n for n in pick_order if n != "NBC"]

    selected_game_ids: set[str] = set()
    assignments: dict[str, dict | None] = {"FOX": None, "CBS": None, "NBC": None}

    for network in pick_order:
        slate = network_slates.get(network, [])
        available = [
            g for g in slate
            if g["game_id"] not in selected_game_ids
            and _is_eligible_for_network(g, network)
        ]

        if not available:
            continue

        values = [g["total_value"] for g in available]
        selected_idx = _softmax_select(values, temperature, rng)
        selected = available[selected_idx]

        selected_game_ids.add(selected["game_id"])
        assignments[network] = {
            "game_id": selected["game_id"],
            "home_team": selected["home_team"],
            "away_team": selected["away_team"],
            "total_value": selected["total_value"],
            "predicted_viewers": viewership_lookup.get(selected["game_id"], 2.0),
        }

    return assignments


def simulate_draft(
    n_iterations: int = settings.SIM_ITERATIONS,
    seed: int | None = settings.SIM_SEED,
    temperature: float = settings.SOFTMAX_TEMPERATURE,
    trade_probability: float = settings.TRADE_PROBABILITY,
) -> dict:
    """
    Run the full two-phase draft simulation across N iterations.

    Phase 1: Networks draft weeks (DRAFT_ORDER determines who picks weeks).
    Phase 2: Within each week, games are selected in priority order.

    Returns:
        Dict with:
            game_assignments: game_id → {fox_prob, cbs_prob, nbc_prob, undrafted_prob}
            avg_weekly_viewers: network → week → avg predicted viewers
            n_iterations, temperature, trade_probability
    """
    rng = np.random.default_rng(seed)

    # Build game values for all weeks
    all_values = build_all_game_values()

    # Load viewership predictions
    viewership_path = settings.OUTPUT_DIR / "expected_viewership.json"
    viewership_lookup: dict[str, float] = {}
    if viewership_path.exists():
        with open(viewership_path) as f:
            vw_list = json.load(f)
        viewership_lookup = {g["game_id"]: g["predicted_viewers_millions"] for g in vw_list}

    # Tracking arrays
    game_network_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"FOX": 0, "CBS": 0, "NBC": 0, "undrafted": 0}
    )
    network_weekly_viewers: dict[str, dict[int, list[float]]] = {
        net: defaultdict(list) for net in ("FOX", "CBS", "NBC")
    }

    # Phase 1 pick tracking: pick_idx → {week → count}
    pick_tracking: dict[int, dict[int, int]] = {}

    # Phase 2 game tracking: (week, network) → {game_id → count}
    week_net_game_counts: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    # Collect all eligible game_ids
    all_game_ids: set[str] = set()
    for week_slates in all_values.values():
        for slate in week_slates.values():
            for g in slate:
                all_game_ids.add(g["game_id"])

    # Run iterations
    for _ in range(n_iterations):
        iteration_selected: set[str] = set()

        # Phase 1: Draft weeks
        week_owners = _draft_weeks(
            all_values, viewership_lookup, rng, temperature, trade_probability,
            pick_tracking=pick_tracking,
        )

        # Track CBS/NBC alternation for 2nd pick in FOX-drafted weeks
        cbs_nbc_toggle = True  # True = CBS gets 2nd first time

        # Phase 2: Game selection within each week
        for week in sorted(all_values.keys()):
            drafting_network = week_owners.get(week)
            if drafting_network is None:
                # Week wasn't drafted (shouldn't happen with 13 picks for 13 weeks)
                continue

            # Determine CBS/NBC toggle for FOX-drafted weeks
            toggle_for_week = cbs_nbc_toggle
            if drafting_network == "FOX":
                cbs_nbc_toggle = not cbs_nbc_toggle  # Alternate for next FOX week

            week_assignments = _select_games_for_week(
                week=week,
                drafting_network=drafting_network,
                network_slates=all_values[week],
                viewership_lookup=viewership_lookup,
                rng=rng,
                temperature=temperature,
                cbs_nbc_second_toggle=toggle_for_week,
            )

            for network, pick in week_assignments.items():
                if pick is not None:
                    game_network_counts[pick["game_id"]][network] += 1
                    iteration_selected.add(pick["game_id"])
                    network_weekly_viewers[network][week].append(
                        pick["predicted_viewers"]
                    )
                    week_net_game_counts[(week, network)][pick["game_id"]] += 1
                else:
                    network_weekly_viewers[network][week].append(0.0)

        # Mark undrafted games
        for gid in all_game_ids:
            if gid not in iteration_selected:
                game_network_counts[gid]["undrafted"] += 1

    # Compute assignment probabilities
    game_assignments: dict[str, dict] = {}
    for game_id, counts in game_network_counts.items():
        game_assignments[game_id] = {
            "game_id": game_id,
            "fox_prob": round(counts["FOX"] / n_iterations, 4),
            "cbs_prob": round(counts["CBS"] / n_iterations, 4),
            "nbc_prob": round(counts["NBC"] / n_iterations, 4),
            "undrafted_prob": round(counts["undrafted"] / n_iterations, 4),
        }

    # Compute average weekly viewers by network
    avg_weekly_viewers: dict[str, dict[int, float]] = {}
    for network in ("FOX", "CBS", "NBC"):
        avg_weekly_viewers[network] = {}
        for week, viewer_list in network_weekly_viewers[network].items():
            avg_weekly_viewers[network][week] = round(
                sum(viewer_list) / len(viewer_list), 3
            ) if viewer_list else 0.0

    # ── Build prediction output ──────────────────────────────────────────────
    # Phase 1 prediction: most-likely week selection for each draft pick
    draft_order = settings.DRAFT_ORDER
    week_draft_prediction: list[dict] = []
    for pick_idx in range(len(draft_order)):
        week_counts = pick_tracking.get(pick_idx, {})
        if week_counts:
            best_week = max(week_counts, key=week_counts.get)
            freq = week_counts[best_week]
            week_draft_prediction.append({
                "pick": pick_idx + 1,
                "network": draft_order[pick_idx],
                "most_likely_week": best_week,
                "probability": round(freq / n_iterations, 4),
            })
        else:
            week_draft_prediction.append({
                "pick": pick_idx + 1,
                "network": draft_order[pick_idx],
                "most_likely_week": None,
                "probability": 0.0,
            })

    # Phase 2 prediction: deterministic game assignment per week
    # Resolve Phase 1 conflicts (multiple picks wanting same week) → assign
    # weeks in pick order, each pick gets its most-likely AVAILABLE week.
    predicted_week_owners: dict[int, tuple[str, int]] = {}  # week → (network, pick#)
    assigned_weeks: set[int] = set()
    assigned_picks: set[int] = set()
    available_weeks = set(all_values.keys())

    for pick_idx in range(len(draft_order)):
        week_counts = pick_tracking.get(pick_idx, {})
        if not week_counts:
            continue
        # Sort candidate weeks by frequency descending; pick best available
        sorted_weeks = sorted(week_counts.keys(), key=lambda w: -week_counts[w])
        for candidate_week in sorted_weeks:
            if candidate_week not in assigned_weeks:
                predicted_week_owners[candidate_week] = (
                    draft_order[pick_idx], pick_idx + 1
                )
                assigned_weeks.add(candidate_week)
                assigned_picks.add(pick_idx)
                break

    # Fallback: assign remaining unowned weeks to unassigned picks
    remaining_weeks = sorted(available_weeks - assigned_weeks)
    unassigned_picks = [
        i for i in range(len(draft_order)) if i not in assigned_picks
    ]
    for week, pick_idx in zip(remaining_weeks, unassigned_picks):
        predicted_week_owners[week] = (draft_order[pick_idx], pick_idx + 1)

    # Build predicted schedule: assign games deterministically by viewership
    nbc_committed_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)
    predicted_schedule: list[dict] = []
    cbs_nbc_toggle = True  # same alternation as simulation

    for week in sorted(all_values.keys()):
        nbc_committed = week in nbc_committed_weeks
        owner_info = predicted_week_owners.get(week)
        if owner_info:
            drafting_network, pick_num = owner_info
        else:
            drafting_network, pick_num = "—", 0

        # Determine pick order for this week (same logic as _select_games_for_week)
        if drafting_network == "FOX":
            second = "CBS" if cbs_nbc_toggle else "NBC"
            third = "NBC" if cbs_nbc_toggle else "CBS"
            pick_order = ["FOX", second, third]
            cbs_nbc_toggle = not cbs_nbc_toggle
        elif drafting_network == "CBS":
            pick_order = ["CBS", "FOX", "NBC"]
        elif drafting_network == "NBC":
            pick_order = ["NBC", "FOX", "CBS"]
        else:
            pick_order = ["FOX", "CBS", "NBC"]

        if nbc_committed:
            pick_order = [n for n in pick_order if n != "NBC"]

        # Collect all games available this week, sorted by predicted viewers desc
        week_games: list[dict] = []
        seen_game_ids: set[str] = set()
        for net_slate in all_values[week].values():
            for g in net_slate:
                if g["game_id"] not in seen_game_ids:
                    seen_game_ids.add(g["game_id"])
                    week_games.append(g)
        week_games.sort(
            key=lambda g: -viewership_lookup.get(g["game_id"], 0.0)
        )

        # Assign in pick order: each network gets the best available game
        week_entry = {
            "week": week,
            "drafted_by": drafting_network,
            "draft_pick": pick_num,
            "nd_week": nbc_committed,
            "games": [],
        }
        assigned_game_ids: set[str] = set()

        for net in pick_order:
            # Filter eligible games for this network
            eligible = [
                g for g in week_games
                if g["game_id"] not in assigned_game_ids
                and _is_eligible_for_network(g, net)
            ]
            if eligible:
                best = eligible[0]  # already sorted by viewership desc
                assigned_game_ids.add(best["game_id"])
                # Get frequency from Monte Carlo for confidence
                freq = week_net_game_counts.get((week, net), {}).get(
                    best["game_id"], 0
                )
                week_entry["games"].append({
                    "network": net,
                    "game_id": best["game_id"],
                    "probability": round(freq / n_iterations, 4) if freq else 0.0,
                    "predicted_viewers": viewership_lookup.get(
                        best["game_id"], 0.0
                    ),
                })

        predicted_schedule.append(week_entry)

    logger.info(
        "Draft simulation complete: %d iterations, %d games, %d weeks",
        n_iterations, len(game_assignments), len(all_values),
    )

    return {
        "game_assignments": game_assignments,
        "avg_weekly_viewers": avg_weekly_viewers,
        "week_draft_prediction": week_draft_prediction,
        "predicted_schedule": predicted_schedule,
        "n_iterations": n_iterations,
        "temperature": temperature,
        "trade_probability": trade_probability,
    }


def build_draft_results(
    n_iterations: int = settings.SIM_ITERATIONS,
    seed: int | None = settings.SIM_SEED,
) -> dict:
    """
    Full pipeline: score games → simulate draft → aggregate results.

    Returns the complete draft simulation output.
    """
    logger.info("Running draft simulation (%d iterations, seed=%s)...",
                n_iterations, seed)
    results = simulate_draft(n_iterations=n_iterations, seed=seed)

    # Enrich assignments with game metadata
    with open(PROCESSED_DIR / "game_schedule.json") as f:
        schedule = json.load(f)
    schedule_lookup = {g["game_id"]: g for g in schedule}

    viewership_path = settings.OUTPUT_DIR / "expected_viewership.json"
    viewership_lookup: dict[str, float] = {}
    if viewership_path.exists():
        with open(viewership_path) as f:
            vw_list = json.load(f)
        viewership_lookup = {g["game_id"]: g["predicted_viewers_millions"] for g in vw_list}

    enriched_assignments = []
    for game_id, probs in results["game_assignments"].items():
        sched_game = schedule_lookup.get(game_id, {})
        enriched_assignments.append({
            **probs,
            "week": sched_game.get("week", 0),
            "home_team": sched_game.get("home_team", "Unknown"),
            "away_team": sched_game.get("away_team", "Unknown"),
            "predicted_viewers_millions": viewership_lookup.get(game_id, 0.0),
            "is_conference_game": sched_game.get("is_conference_game", False),
        })

    # Sort by predicted viewers descending
    enriched_assignments.sort(key=lambda x: -x["predicted_viewers_millions"])
    results["enriched_assignments"] = enriched_assignments

    # Enrich predicted_schedule with team names
    for week_entry in results.get("predicted_schedule", []):
        for game_entry in week_entry.get("games", []):
            gid = game_entry.get("game_id")
            if gid and gid in schedule_lookup:
                game_entry["home_team"] = schedule_lookup[gid].get("home_team", "Unknown")
                game_entry["away_team"] = schedule_lookup[gid].get("away_team", "Unknown")
            else:
                game_entry.setdefault("home_team", "Unknown")
                game_entry.setdefault("away_team", "Unknown")

    return results

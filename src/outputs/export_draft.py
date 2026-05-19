"""
src/outputs/export_draft.py
───────────────────────────
Export Artifact 3: Broadcaster draft simulation results.

Produces three output files:
  data/outputs/draft_assignments.json    — game-level network assignment probabilities
  data/outputs/draft_assignments.csv     — flat summary
  data/outputs/draft_weekly_viewers.json  — expected weekly viewership by network
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

OUTPUT_DIR: Path = settings.OUTPUT_DIR


def export_draft(draft_results: dict, output_dir: Path | None = None) -> dict[str, Path]:
    """
    Export draft simulation results to JSON and CSV.

    Args:
        draft_results: Output from draft_simulator.build_draft_results()
        output_dir: Optional override for output directory.

    Returns:
        Dict with paths to the exported files.
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    assignments = draft_results.get("enriched_assignments", [])
    avg_weekly = draft_results.get("avg_weekly_viewers", {})

    # ── JSON export: game assignments ─────────────────────────────────────────
    json_path = out / "draft_assignments.json"
    with open(json_path, "w") as f:
        json.dump(assignments, f, indent=2)
    logger.info("Exported %d game assignments to %s", len(assignments), json_path)

    # ── CSV export: game assignments ──────────────────────────────────────────
    csv_path = out / "draft_assignments.csv"
    fieldnames = [
        "game_id", "week", "home_team", "away_team",
        "predicted_viewers_millions", "is_conference_game",
        "fox_prob", "cbs_prob", "nbc_prob", "undrafted_prob",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(assignments)
    logger.info("Exported CSV assignments to %s", csv_path)

    # ── JSON export: weekly viewers by network ────────────────────────────────
    weekly_path = out / "draft_weekly_viewers.json"
    weekly_summary = {
        "metadata": {
            "n_iterations": draft_results.get("n_iterations", 0),
            "temperature": draft_results.get("temperature", 0.3),
            "trade_probability": draft_results.get("trade_probability", 0.15),
            "draft_order": settings.DRAFT_ORDER,
        },
        "weekly_viewers_by_network": avg_weekly,
        "season_totals": {},
        "week_draft_prediction": draft_results.get("week_draft_prediction", []),
        "predicted_schedule": draft_results.get("predicted_schedule", []),
    }

    # Compute season total expected viewers per network (Monte Carlo average)
    for network in ("FOX", "CBS", "NBC"):
        if network in avg_weekly:
            weekly_summary["season_totals"][network] = round(
                sum(avg_weekly[network].values()), 3
            )

    # Compute predicted season totals from deterministic predicted_schedule
    # (matches what the dashboard game cards display)
    predicted_totals: dict[str, float] = {"FOX": 0.0, "CBS": 0.0, "NBC": 0.0}
    for week_entry in weekly_summary["predicted_schedule"]:
        for game in week_entry.get("games", []):
            net = game.get("network", "")
            if net in predicted_totals:
                predicted_totals[net] += game.get("predicted_viewers", 0.0)
    # Add NBC Notre Dame game viewers for ND weeks
    nd_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)
    viewership_path = settings.OUTPUT_DIR / "expected_viewership.json"
    if viewership_path.exists():
        with open(viewership_path) as f:
            vw_list = json.load(f)
        for g in vw_list:
            if g["week"] in nd_weeks and "Notre Dame" in (g["home_team"], g["away_team"]):
                predicted_totals["NBC"] += g["predicted_viewers_millions"]
    weekly_summary["predicted_season_totals"] = {
        k: round(v, 3) for k, v in predicted_totals.items()
    }

    with open(weekly_path, "w") as f:
        json.dump(weekly_summary, f, indent=2)
    logger.info("Exported weekly viewer summary to %s", weekly_path)

    return {"json": json_path, "csv": csv_path, "weekly": weekly_path}

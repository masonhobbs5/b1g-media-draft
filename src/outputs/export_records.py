"""
src/outputs/export_records.py
─────────────────────────────
Export Artifact 1: Expected team records from Monte Carlo simulation.

Produces two output files:
  data/outputs/expected_records.json    — full detail per team
  data/outputs/expected_records.csv     — flat summary for easy consumption
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

OUTPUT_DIR: Path = settings.OUTPUT_DIR


def export_records(summaries: list[dict], output_dir: Path | None = None) -> dict[str, Path]:
    """
    Export simulation summaries to JSON and CSV.

    Args:
        summaries: Output from season_simulator.summarize_results()
        output_dir: Optional override for output directory.

    Returns:
        Dict with paths to the exported files.
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ── JSON export (full detail) ─────────────────────────────────────────────
    json_path = out / "expected_records.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Exported %d team records to %s", len(summaries), json_path)

    # ── CSV export (flat summary) ─────────────────────────────────────────────
    csv_path = out / "expected_records.csv"
    fieldnames = [
        "team", "games_played", "conf_games_played",
        "mean_wins", "mean_conf_wins", "median_wins", "std_wins",
        "p_8_plus_wins", "p_10_plus_wins", "p_11_plus_wins",
        "p_undefeated_conf",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    logger.info("Exported CSV summary to %s", csv_path)

    return {"json": json_path, "csv": csv_path}

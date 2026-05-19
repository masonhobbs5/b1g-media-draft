"""
src/outputs/export_viewership.py
────────────────────────────────
Export Artifact 2: Expected viewership predictions for 2026 B1G games.

Produces two output files:
  data/outputs/expected_viewership.json  — full detail per game
  data/outputs/expected_viewership.csv   — flat summary for easy consumption
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

OUTPUT_DIR: Path = settings.OUTPUT_DIR


def export_viewership(predictions: list[dict], output_dir: Path | None = None) -> dict[str, Path]:
    """
    Export viewership predictions to JSON and CSV.

    Args:
        predictions: Output from viewership_model.predict_2026()
        output_dir: Optional override for output directory.

    Returns:
        Dict with paths to the exported files.
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ── JSON export (full detail) ─────────────────────────────────────────────
    json_path = out / "expected_viewership.json"
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info("Exported %d game predictions to %s", len(predictions), json_path)

    # ── CSV export (flat summary) ─────────────────────────────────────────────
    csv_path = out / "expected_viewership.csv"
    fieldnames = [
        "game_id", "week", "game_date", "home_team", "away_team",
        "predicted_viewers_millions", "lower_bound_millions",
        "upper_bound_millions", "is_conference_game", "is_rivalry",
        "is_cfp_rematch", "combined_brand", "home_strength", "away_strength",
        "market_score",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    logger.info("Exported CSV viewership to %s", csv_path)

    return {"json": json_path, "csv": csv_path}

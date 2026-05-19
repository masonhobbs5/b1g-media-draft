#!/usr/bin/env python3
"""
scripts/run_simulation.py
─────────────────────────
End-to-end simulation pipeline producing all three B1G artifacts:
  Artifact 1: Expected team records
  Artifact 2: Expected game viewership
  Artifact 3: Broadcaster draft assignments

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --iterations 5000 --seed 123
    python scripts/run_simulation.py --output-dir ./results --quiet
    python scripts/run_simulation.py --rebuild-strength --skip-draft
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.model.team_strength import build_team_strength
from src.model.win_probability import build_win_probabilities
from src.simulation.season_simulator import run_simulation
from src.outputs.export_records import export_records
from src.model.viewership_model import build_viewership_predictions
from src.outputs.export_viewership import export_viewership
from src.simulation.draft_simulator import build_draft_results
from src.outputs.export_draft import export_draft

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run B1G season simulation end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Produces expected records, viewership predictions, and draft assignments.",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=settings.SIM_ITERATIONS,
        help=f"Number of Monte Carlo iterations (default: {settings.SIM_ITERATIONS})",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=settings.SIM_SEED,
        help=f"Random seed (default: {settings.SIM_SEED})",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Output directory for artifacts (default: data/outputs/)",
    )
    parser.add_argument(
        "--rebuild-strength", action="store_true",
        help="Recompute team_strength.json and fbs_strength.json from source data",
    )
    parser.add_argument(
        "--skip-draft", action="store_true",
        help="Skip the draft simulation (Artifact 3)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress summary tables (only log progress)",
    )
    return parser.parse_args()


def print_records_table(summaries: list[dict], iterations: int, seed: int) -> None:
    print("\n" + "=" * 65)
    print("  2026 BIG TEN EXPECTED RECORDS")
    print(f"  Monte Carlo Simulation ({iterations:,} iterations, seed={seed})")
    print("=" * 65)
    print(f"{'Team':20s} {'E[W]':>5s} {'E[CW]':>6s} {'P(8+)':>7s} {'P(10+)':>7s} {'P(11+)':>7s}")
    print("-" * 65)
    for s in summaries:
        print(f"{s['team']:20s} {s['mean_wins']:5.1f} {s['mean_conf_wins']:6.1f}"
              f" {s['p_8_plus_wins']:7.1%} {s['p_10_plus_wins']:7.1%}"
              f" {s['p_11_plus_wins']:7.1%}")
    print("-" * 65)


def print_viewership_table(predictions: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("  2026 BIG TEN EXPECTED VIEWERSHIP (Top 20)")
    print("=" * 65)
    print(f"{'Wk':>3s} {'Home':15s} {'Away':15s} {'Viewers':>8s} {'90% CI':>15s}")
    print("-" * 65)
    for g in predictions[:20]:
        ci = f"[{g['lower_bound_millions']:.1f}-{g['upper_bound_millions']:.1f}]"
        print(f"{g['week']:3d} {g['home_team']:15s} {g['away_team']:15s}"
              f" {g['predicted_viewers_millions']:8.2f}M {ci:>15s}")
    print("-" * 65)


def print_draft_table(draft_results: dict, iterations: int) -> None:
    print("\n" + "=" * 65)
    print("  2026 BIG TEN BROADCASTER DRAFT (Top 15 Games)")
    print(f"  {iterations:,} iterations, temp={settings.SOFTMAX_TEMPERATURE},"
          f" trade_prob={settings.TRADE_PROBABILITY}")
    print("=" * 65)
    print(f"{'Wk':>3s} {'Home':13s} {'Away':13s} {'Viewers':>7s}"
          f" {'FOX':>5s} {'CBS':>5s} {'NBC':>5s} {'None':>5s}")
    print("-" * 65)
    for g in draft_results["enriched_assignments"][:15]:
        print(f"{g['week']:3d} {g['home_team']:13s} {g['away_team']:13s}"
              f" {g['predicted_viewers_millions']:6.1f}M"
              f" {g['fox_prob']:5.0%} {g['cbs_prob']:5.0%}"
              f" {g['nbc_prob']:5.0%} {g['undrafted_prob']:5.0%}")
    print("-" * 65)

    weekly = draft_results.get("avg_weekly_viewers", {})
    print("\n  Season expected viewers by network:")
    for net in ("FOX", "CBS", "NBC"):
        total = sum(weekly.get(net, {}).values())
        print(f"    {net}: {total:.1f}M total ({total / 13:.1f}M/week avg)")


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = args.output_dir or settings.OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ── Step 1: Team strength ─────────────────────────────────────────────────
    if args.rebuild_strength or not (settings.PROCESSED_DIR / "team_strength.json").exists():
        logger.info("=== Step 1: Building team strength ratings ===")
        build_team_strength()
    else:
        logger.info("=== Step 1: Using existing team_strength.json ===")

    # ── Step 2: Win probabilities ─────────────────────────────────────────────
    logger.info("=== Step 2: Computing win probabilities ===")
    build_win_probabilities()

    # ── Step 3: Monte Carlo season simulation ─────────────────────────────────
    logger.info("=== Step 3: Running Monte Carlo (%d iterations, seed=%s) ===",
                args.iterations, args.seed)
    summaries = run_simulation(n_iterations=args.iterations, seed=args.seed)

    # ── Step 4: Export Artifact 1 (expected records) ──────────────────────────
    logger.info("=== Step 4: Exporting Artifact 1 ===")
    rec_paths = export_records(summaries, output_dir=output_dir)

    # ── Step 5: Viewership predictions (Artifact 2) ───────────────────────────
    logger.info("=== Step 5: Building viewership predictions ===")
    vw_result = build_viewership_predictions()
    vw_paths = export_viewership(vw_result["predictions"], output_dir=output_dir)

    # ── Step 6: Draft simulation (Artifact 3) ─────────────────────────────────
    draft_paths = {}
    draft_results = None
    if not args.skip_draft:
        logger.info("=== Step 6: Running broadcaster draft simulation ===")
        draft_results = build_draft_results(n_iterations=args.iterations, seed=args.seed)
        draft_paths = export_draft(draft_results, output_dir=output_dir)

    elapsed = time.perf_counter() - t0

    # ── Summary output ────────────────────────────────────────────────────────
    if not args.quiet:
        print_records_table(summaries, args.iterations, args.seed)
        print_viewership_table(vw_result["predictions"])
        if draft_results:
            print_draft_table(draft_results, args.iterations)

    # ── File manifest ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 65}")
    print(f"  Artifact 1: {rec_paths['json']}")
    print(f"              {rec_paths['csv']}")
    print(f"  Artifact 2: {vw_paths['json']}")
    print(f"              {vw_paths['csv']}")
    if draft_paths:
        print(f"  Artifact 3: {draft_paths['json']}")
        print(f"              {draft_paths['csv']}")
        print(f"              {draft_paths['weekly']}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

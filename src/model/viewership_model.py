"""
src/model/viewership_model.py
─────────────────────────────
Viewership prediction model for B1G football games.

Uses a log-linear Ridge regression trained on historical viewership data with
features derived from team brands, Elo ratings, week/slot multipliers, network
reach, and game context. Predicts log(viewers) and exponentiates with Duan's
smearing correction for unbiased point estimates.

Includes:
  - Leave-one-season-out cross-validation
  - Median APE and R² metrics
  - Prediction intervals via residual bootstrap
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config.settings import settings
from src.model.features import (
    FEATURE_COLUMNS,
    build_training_features,
    build_2026_features,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR: Path = settings.PROCESSED_DIR
OUTPUT_DIR: Path = settings.OUTPUT_DIR

# Model hyperparameters
RIDGE_ALPHA = 50.0  # Regularization strength (tuned via LOSO-CV)
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.90  # 90% prediction interval
# Minimum viewership threshold for training: excludes streaming/BTN games that
# the draft model is never asked to predict (sub-0.5M games corrupt metrics).
MIN_VIEWERSHIP_THRESHOLD = 0.5  # millions


def _extract_X_y(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from feature rows."""
    X = np.array([[row[col] for col in FEATURE_COLUMNS] for row in rows])
    y = np.array([row["viewers_millions"] for row in rows])
    return X, y


def _extract_X(rows: list[dict]) -> np.ndarray:
    """Extract feature matrix X from feature rows (no target needed)."""
    return np.array([[row[col] for col in FEATURE_COLUMNS] for row in rows])


def train_model(
    training_rows: list[dict] | None = None,
    alpha: float = RIDGE_ALPHA,
) -> dict:
    """
    Train a log-linear Ridge regression viewership model on all historical data.

    Fits in log-space: log(viewers) ~ X, then uses Duan's smearing estimator
    to correct bias when exponentiating back to viewers-space.

    Args:
        training_rows: Feature dicts from build_training_features()
        alpha: Ridge regularization parameter

    Returns:
        Dict with model, scaler, smearing_factor, residuals, and diagnostics.
    """
    if training_rows is None:
        training_rows = build_training_features()

    # Drop sub-threshold games (streaming/BTN exclusives) — not draft-eligible
    training_rows = [
        r for r in training_rows
        if r["viewers_millions"] >= MIN_VIEWERSHIP_THRESHOLD
    ]

    X, y = _extract_X_y(training_rows)
    log_y = np.log(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, log_y)

    # Duan's smearing factor: E[exp(ε)] where ε = log(y) - log(ŷ)
    log_pred_train = model.predict(X_scaled)
    log_residuals = log_y - log_pred_train
    smearing_factor = float(np.mean(np.exp(log_residuals)))

    # Back-transform predictions to viewers-space for diagnostics
    y_pred = np.exp(log_pred_train) * smearing_factor
    residuals = y - y_pred  # residuals in viewers-space (for prediction intervals)

    # Diagnostics — use median APE (robust to outliers)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot
    mape = float(np.median(np.abs(residuals / y)) * 100.0)

    logger.info(
        "Trained log-linear Ridge: R²=%.3f, MdAPE=%.1f%%, smear=%.3f, n=%d",
        r_squared, mape, smearing_factor, len(y),
    )

    # Feature importance (coefficients on standardized features in log-space)
    coef_dict = {
        col: round(float(coef), 4)
        for col, coef in zip(FEATURE_COLUMNS, model.coef_)
    }
    logger.info("Coefficients (log-space): %s", coef_dict)

    return {
        "model": model,
        "scaler": scaler,
        "smearing_factor": smearing_factor,
        "residuals": residuals,
        "r_squared": r_squared,
        "mdape": mape,
        "coefficients": coef_dict,
        "intercept": float(model.intercept_),
        "n_train": len(y),
    }


def cross_validate(
    training_rows: list[dict] | None = None,
    alpha: float = RIDGE_ALPHA,
) -> dict:
    """
    Leave-one-season-out cross-validation.

    Trains on 2 seasons, tests on the held-out season. Reports median APE and
    R² for each fold and overall. Games below MIN_VIEWERSHIP_THRESHOLD are
    excluded from both training and evaluation (streaming/BTN exclusives).

    Returns dict with per-fold and aggregate metrics.
    """
    if training_rows is None:
        training_rows = build_training_features()

    # Drop sub-threshold games consistent with train_model()
    training_rows = [
        r for r in training_rows
        if r["viewers_millions"] >= MIN_VIEWERSHIP_THRESHOLD
    ]

    seasons = sorted(set(row["season"] for row in training_rows))
    fold_results = []
    all_residuals = []
    all_actuals = []

    for held_out in seasons:
        train = [r for r in training_rows if r["season"] != held_out]
        test = [r for r in training_rows if r["season"] == held_out]

        if not train or not test:
            continue

        X_train, y_train = _extract_X_y(train)
        X_test, y_test = _extract_X_y(test)

        log_y_train = np.log(y_train)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_s, log_y_train)

        # Smearing factor from this fold's training residuals
        log_resid_train = log_y_train - model.predict(X_train_s)
        smear = np.mean(np.exp(log_resid_train))

        # Predict in viewers-space
        log_pred = model.predict(X_test_s)
        y_pred = np.exp(log_pred) * smear
        residuals = y_test - y_pred

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mdape = float(np.median(np.abs(residuals / y_test)) * 100.0)

        fold_results.append({
            "held_out_season": held_out,
            "n_train": len(train),
            "n_test": len(test),
            "r_squared": round(r2, 4),
            "mdape": round(mdape, 2),
            "rmse": round(float(np.sqrt(np.mean(residuals ** 2))), 3),
        })

        all_residuals.extend(residuals.tolist())
        all_actuals.extend(y_test.tolist())

    # Aggregate metrics
    all_residuals = np.array(all_residuals)
    all_actuals = np.array(all_actuals)
    overall_ss_res = np.sum(all_residuals ** 2)
    overall_ss_tot = np.sum((all_actuals - np.mean(all_actuals)) ** 2)
    overall_r2 = 1.0 - overall_ss_res / overall_ss_tot
    overall_mdape = float(np.median(np.abs(all_residuals / all_actuals)) * 100.0)

    cv_results = {
        "folds": fold_results,
        "overall_r_squared": round(overall_r2, 4),
        "overall_mdape": round(overall_mdape, 2),
        "overall_rmse": round(float(np.sqrt(np.mean(all_residuals ** 2))), 3),
    }

    logger.info(
        "CV results: overall R²=%.3f, MdAPE=%.1f%%, RMSE=%.3f",
        overall_r2, overall_mdape, cv_results["overall_rmse"],
    )
    for fold in fold_results:
        logger.info(
            "  Fold %d: R²=%.3f, MdAPE=%.1f%%, n_test=%d",
            fold["held_out_season"], fold["r_squared"],
            fold["mdape"], fold["n_test"],
        )

    return cv_results


def predict_2026(
    model_result: dict | None = None,
    game_features: list[dict] | None = None,
    confidence: float = CONFIDENCE_LEVEL,
) -> list[dict]:
    """
    Generate viewership predictions with prediction intervals for 2026 games.

    Uses residual-based prediction intervals: the interval is derived from
    the empirical distribution of training residuals.

    Args:
        model_result: Output from train_model() (auto-computed if None)
        game_features: Output from build_2026_features() (auto-loaded if None)
        confidence: Confidence level for prediction interval (default 0.90)

    Returns:
        List of dicts per game with predicted_viewers, lower_bound, upper_bound.
    """
    if model_result is None:
        model_result = train_model()

    if game_features is None:
        game_features = build_2026_features()

    model = model_result["model"]
    scaler = model_result["scaler"]
    smearing_factor = model_result["smearing_factor"]
    residuals = model_result["residuals"]

    X = _extract_X(game_features)
    X_scaled = scaler.transform(X)
    log_predictions = model.predict(X_scaled)
    predictions = np.exp(log_predictions) * smearing_factor

    # Prediction interval from residual quantiles
    alpha = 1.0 - confidence
    lower_q = np.percentile(residuals, 100 * alpha / 2)
    upper_q = np.percentile(residuals, 100 * (1 - alpha / 2))

    # ── Post-prediction adjustments ────────────────────────────────────────────
    # Rivalry floor: Ohio State–Michigan consistently draws 15-19M regardless
    # of what the regression predicts from generic brand/Elo features.
    OSU_MICHIGAN_FLOOR = 10.0  # millions — lowest recent primary broadcast was ~12M
    # Michigan home boost: Big House + noon FOX window draws ~1.5M above model
    MICHIGAN_HOME_BOOST = 0.5  # millions added to games at Michigan

    results = []
    for i, game in enumerate(game_features):
        pred = float(predictions[i])

        # Michigan home boost (applies to all games at Michigan)
        if game["home_team"] == "Michigan":
            pred += MICHIGAN_HOME_BOOST

        # Ohio State–Michigan rivalry floor
        teams = {game["home_team"], game["away_team"]}
        if teams == {"Ohio State", "Michigan"}:
            pred = max(pred, OSU_MICHIGAN_FLOOR)

        # Floor predictions at a minimum (games can't have < 0 viewers)
        pred = max(pred, 0.3)
        lower = max(pred + lower_q, 0.1)
        upper = pred + upper_q

        results.append({
            "game_id": game["game_id"],
            "week": game["week"],
            "game_date": game.get("game_date"),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "predicted_viewers_millions": round(pred, 3),
            "lower_bound_millions": round(lower, 3),
            "upper_bound_millions": round(upper, 3),
            "is_conference_game": bool(game["is_conference_game"]),
            "is_rivalry": bool(game["is_rivalry"]),
            "is_cfp_rematch": bool(game.get("is_cfp_rematch", 0)),
            "combined_brand": round(game["combined_brand"], 3),
            "home_strength": round(game.get("home_strength", 0.0), 4),
            "away_strength": round(game.get("away_strength", 0.0), 4),
            "market_score": round(game.get("market_score", 0.0), 4),
        })

    # Sort by predicted viewers descending
    results.sort(key=lambda x: -x["predicted_viewers_millions"])

    logger.info(
        "Generated 2026 predictions: %d games, top=%.2fM, median=%.2fM",
        len(results),
        results[0]["predicted_viewers_millions"] if results else 0,
        results[len(results) // 2]["predicted_viewers_millions"] if results else 0,
    )

    return results


def build_viewership_predictions() -> dict:
    """
    Full pipeline: train model, cross-validate, and predict 2026.

    Returns dict with predictions, model diagnostics, and CV results.
    """
    logger.info("Building viewership feature table...")
    training_rows = build_training_features()

    logger.info("Cross-validating model...")
    cv_results = cross_validate(training_rows)

    logger.info("Training final model on all data...")
    model_result = train_model(training_rows)

    logger.info("Scoring 2026 games...")
    predictions = predict_2026(model_result)

    # Persist model diagnostics
    diagnostics = {
        "cv_results": cv_results,
        "model_r_squared": model_result["r_squared"],
        "model_mdape": model_result["mdape"],
        "smearing_factor": model_result["smearing_factor"],
        "coefficients": model_result["coefficients"],
        "intercept": model_result["intercept"],
        "n_training_samples": model_result["n_train"],
        "feature_columns": FEATURE_COLUMNS,
        "ridge_alpha": RIDGE_ALPHA,
    }

    diag_path = PROCESSED_DIR / "viewership_model_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Wrote model diagnostics to %s", diag_path)

    return {
        "predictions": predictions,
        "diagnostics": diagnostics,
        "model_result": model_result,
    }

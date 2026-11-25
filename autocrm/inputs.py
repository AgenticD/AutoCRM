"""Data contracts and default project paths."""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "config.yaml"
DICTIONARY = ROOT / "config" / "biomarker_dictionary.json"
INPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "output"

REQUIRED_COLUMNS = {
    "users.json": ["_id", "status", "createdAt"],
    "files.json": ["_id", "userId", "createdAt", "testDate"],
    "usagetokens.json": ["_id", "userId", "fileId", "createdAt"],
    "paymenttransactions.json": ["userId", "status", "amountRial", "createdAt", "paidAt"],
    "labresults.json": ["_id", "fileId", "type", "text", "createdAt"],
}
OPTIONAL_USER_COLUMNS = ["age", "bmi"]


def load_config(path):
    text = path.read_text(encoding="utf-8-sig")
    config = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a configuration object in {path}")
    return config


def feature_columns(config):
    columns = [
        "upload_count", "uploads_last_window", "successful_payment_count", "failed_payment_count",
        "payments_last_window", "ai_request_count", "ai_event_count", "ai_events_last_window",
        "activity_basis", "recency_label", "lab_count", "extracted_biomarker_count",
        "abnormal_biomarker_count", "borderline_count", "normalized_abnormal_ratio",
        "peak_severity_score", "severity_score", "severity_tier", "unmatched_biomarker_count",
        "health_score", "health_tier", "health_status", "health_missing_factors",
    ]
    trend = config.get("health_trend", {})
    if trend.get("enabled"):
        deltas = [point["column"] for point in trend.get("points", []) if point.get("enabled")]
        slope = trend.get("column_names", {}).get("slope", "health_slope_3pt")
        if not all(column.startswith("health_delta_") for column in deltas) or not slope.startswith("health_slope_"):
            raise ValueError("Trend columns must use the health_delta_ and health_slope_ prefixes")
        columns.extend([
            *deltas, slope,
            "health_trend_label", "health_trend_confidence", "health_history_points",
        ])
    return columns + [
        "recency_score", "frequency_score", "freq_lifetime_component", "freq_window_component",
        "monetary_score", "recency_band", "frequency_band", "monetary_band", "has_ever_paid", "rfm_segment",
    ]

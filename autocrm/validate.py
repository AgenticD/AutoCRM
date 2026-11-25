"""Validate generated CRM features against the input files and scoring rules."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import CONFIG, DICTIONARY, INPUT_DIR, OUTPUT_DIR, feature_columns, load_config
from .segmentation import VALID_SEGMENTS, rfm_band, segment


SCORE_NAMES = ("recency", "frequency", "monetary")
NUMERIC_COLUMNS = {
    "extracted_biomarker_count", "abnormal_biomarker_count", "normalized_abnormal_ratio",
    "peak_severity_score", "severity_score", "health_score",
    *(f"{name}_score" for name in SCORE_NAMES),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def eligible_ids(users):
    ids = set()
    for user in users:
        if str(user.get("status", "")).lower() == "deleted":
            continue
        user_id = user.get("_id")
        if isinstance(user_id, dict) and "$oid" in user_id:
            user_id = user_id["$oid"]
        ids.add(str(user_id))
    return ids


def validate(base, out, dictionary_path, config):
    results = []

    def check(name, passed, detail=""):
        results.append({"check": name, "level": "PASS" if passed else "FAIL", "detail": detail})
        return passed

    def read(name, reader, expected_type):
        try:
            value = reader()
            if isinstance(value, expected_type):
                check(f"{name} parses", True)
                return value
            check(f"{name} parses", False, f"Expected {expected_type.__name__}")
        except (OSError, ValueError):
            check(f"{name} parses", False)
        return None

    users = read("Users", lambda: read_json(base / "users.json"), list)
    severity_summary = read("Severity summary", lambda: read_json(out / "severity_breakdown.json"), dict)
    metadata = read("Run metadata", lambda: read_json(out / "run_metadata.json"), dict)
    dictionary = read("Dictionary", lambda: read_json(dictionary_path), dict)
    crm = read("CRM table", lambda: pd.read_csv(out / "crm_features.csv"), pd.DataFrame)

    if dictionary is not None:
        records = dictionary.get("records", [])
        aliases = [alias.casefold() for record in records for alias in record.get("aliases", [])]
        check("Dictionary version exists", bool(dictionary.get("dictionary_version") or dictionary.get("meta", {}).get("version") or dictionary.get("version")))
        check("Dictionary aliases unique", len(aliases) == len(set(aliases)))
        check("Dictionary records complete", all({"canonical_id", "category", "severity_weight", "aliases"} <= record.keys() for record in records))
        check("Dictionary weights valid", all(0 <= float(record.get("severity_weight", -1)) <= 1.5 for record in records))

    if crm is None:
        return results
    if not check("Output columns match the feature schema", list(crm.columns) == feature_columns(config)):
        return results

    if users is not None and check("User records are objects", all(isinstance(user, dict) for user in users)):
        eligible = eligible_ids(users)
        check("One row per eligible user", len(crm) == len(eligible), f"{len(crm)} rows, {len(eligible)} eligible users")
        if metadata is not None:
            check("Run metadata counts match", metadata.get("output_user_count") == len(crm)
                  and metadata.get("eligible_user_count") == len(eligible))
    if severity_summary is not None:
        expected = Counter(crm.loc[crm.severity_score.notna(), "severity_tier"])
        check("Severity summary matches feature counts", severity_summary == dict(expected))
    if crm.empty:
        return results

    numeric = crm[list(NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not check("Score and count columns are numeric", crm[numeric.columns].notna().equals(numeric.notna())):
        return results
    crm[numeric.columns] = numeric
    health = config.get("health", config.get("health_score"))
    severity = health.get("severity") or config.get("technical", {}).get("severity")
    max_finding = severity["max_finding_score"]
    lab = crm[crm.severity_score.notna()]
    expected_ratio = lab.abnormal_biomarker_count / lab.extracted_biomarker_count
    expected_severity = (lab.peak_severity_score / max_finding).clip(0, 1) * 100
    check("Abnormal ratio reproducible", np.allclose(expected_ratio, lab.normalized_abnormal_ratio, atol=0.0001))
    check("Severity normalization reproducible", np.allclose(expected_severity, lab.severity_score, atol=0.02))
    check("Abnormal count bounded", (lab.abnormal_biomarker_count <= lab.extracted_biomarker_count).all())
    check("Abnormal ratios bounded", lab.normalized_abnormal_ratio.between(0, 1).all())
    check("Severity scores bounded", lab.severity_score.between(0, 100).all())
    check("Health scores bounded", crm.health_score.dropna().between(0, 100).all())
    check("Users without lab scores have unknown health", (crm.loc[crm.severity_score.isna(), "health_status"] == "Unknown").all())

    rfm = config.get("rfm", {})
    for name in SCORE_NAMES:
        thresholds = config.get(name, rfm.get(name))["band_thresholds"]
        scores = crm[f"{name}_score"]
        check(f"{name.title()} scores bounded", scores.between(0, 5).all())
        check(f"{name.title()} bands reproducible", (scores.map(lambda score: rfm_band(score, thresholds)) == crm[f"{name}_band"]).all())
    check("RFM segments valid", set(crm.rfm_segment) <= VALID_SEGMENTS)
    check("RFM rules reproducible", (crm.apply(segment, axis=1) == crm.rfm_segment).all())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=INPUT_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()

    try:
        results = validate(args.base, args.out, args.dictionary, load_config(args.config))
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        parser.exit(1, "Validation failed. Check the configuration and input formats.\n")
    counts = Counter(result["level"] for result in results)
    for result in results:
        print(f"[{result['level']}] {result['check']} {result['detail']}".rstrip())
    summary = {"PASS": counts["PASS"], "FAIL": counts["FAIL"]}
    print(f"SUMMARY: {counts['PASS']} PASS, {counts['FAIL']} FAIL")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validation_report.json").write_text(
        json.dumps({"summary": summary, "checks": results}, indent=2), encoding="utf-8"
    )
    return int(counts["FAIL"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())

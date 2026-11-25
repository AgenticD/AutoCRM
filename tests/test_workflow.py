import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path

from autocrm.inputs import ROOT
from autocrm.segmentation import VALID_SEGMENTS, segment


class WorkflowTests(unittest.TestCase):
    def test_demo_and_missing_output_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "input"
            out = Path(temp) / "output"
            shutil.copytree(ROOT / "examples", base)
            users_path = base / "users.json"
            users = json.loads(users_path.read_text(encoding="utf-8"))
            contact = "private-contact-canary"
            phone = "555" * 5
            users[0].update(name=contact, email=contact, phone=phone, birthDate="1987-01-03")
            users_path.write_text(json.dumps(users), encoding="utf-8")
            lab_path = base / "labresults.json"
            lab_path.write_text(
                lab_path.read_text(encoding="utf-8").replace("Glucose", "private-lab-canary"),
                encoding="utf-8",
            )
            paths = ["--base", str(base), "--out", str(out)]
            result = self.run_python("run.py", *paths, "--as-of-date", "2026-07-22")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            features = out / "crm_features.csv"
            with features.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["rfm_segment"] for row in rows}, {"champion", "new_user"})
            self.assertTrue({"userId", "name", "age", "bmi", "email", "phone", "total_payment_rial"}.isdisjoint(header))
            self.assertEqual({row["health_history_points"] for row in rows}, {"0", "2"})
            summary = json.loads((out / "severity_breakdown.json").read_text(encoding="utf-8"))
            self.assertEqual(summary, {"low": 1})
            metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["as_of_date"], "2026-07-22")
            self.assertNotIn("input_sha256", metadata)
            self.assertFalse((out / "health_history.json").exists())
            artifacts = [path.read_text(encoding="utf-8-sig") for path in out.iterdir()]
            for artifact in [result.stdout, result.stderr, *artifacts]:
                for private_value in (
                    contact, phone, "1987-01-03", users[0]["createdAt"], "private-lab-canary",
                    "user-a", "report-a", "250000",
                ):
                    self.assertNotIn(private_value, artifact)

            with features.open("w", encoding="utf-8-sig", newline="") as handle:
                csv.writer(handle).writerow(header)
            result = self.run_python("-m", "autocrm.validate", *paths)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("[FAIL] One row per eligible user", result.stdout)

    def test_segmentation_combinations(self):
        bands = ("low", "medium", "high")
        for recency, frequency, monetary, paid in product(
            ("fresh", "warm", "cold", "new", "never_active"), bands, bands, ("yes", "no")
        ):
            row = {
                "recency_label": recency,
                "frequency_band": frequency,
                "monetary_band": monetary,
                "has_ever_paid": paid,
            }
            self.assertIn(segment(row), VALID_SEGMENTS, row)

    @staticmethod
    def run_python(*args):
        return subprocess.run(
            [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
        )

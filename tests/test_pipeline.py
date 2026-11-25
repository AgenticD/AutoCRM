# ══════════════════════════════════════════════════════════════════════
# یونیت‌تست‌های پایپ‌لاین — هر تست با دیتای ساختگیِ کوچک، پایپ‌لاین را
# زیرفرایند جدا اجرا می‌کند (ایزوله از محیط) و رفتار را از ابتدا تا انتها چک می‌کند:
#   - داده خالی → خروجی خالی اما سالم (جدول با هدر، فایل‌های جیسون ریسپانسیو)
#   - کاربر تکراری → حذف تکراری (یک ردیف به‌ازای هر کاربر)
#   - ستون الزامی کم → پیش‌از-اجرا با پیام واضح، نه کرش مبهم
#   - کانفیگ وزن خراب (درستی‌نما/تهی/بی‌نهایت/جمع نادرست/…) → رد صریح (کلاس تست‌های رگرسیون)
# اجرا: دستور "python -m unittest discover" روی پوشه تست با الگوی نام فایل تست
# ══════════════════════════════════════════════════════════════════════
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from autocrm.inputs import CONFIG, DICTIONARY, REQUIRED_COLUMNS, ROOT
_MISSING = object()

class PipelineTests(unittest.TestCase):
    def run_pipeline(self, payloads):
        temp = tempfile.TemporaryDirectory()
        base = Path(temp.name) / "raw"
        out = Path(temp.name) / "out"
        base.mkdir()
        for name in REQUIRED_COLUMNS:
            (base / name).write_text(json.dumps(payloads.get(name, [])), encoding="utf-8")
        result = subprocess.run([
            sys.executable, "-m", "autocrm.pipeline", "--base", str(base), "--out", str(out),
            "--config", str(CONFIG), "--dictionary", str(DICTIONARY), "--as-of-date", "2026-01-01",
        ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return temp, out, result

    def test_empty_arrays_write_valid_empty_artifacts(self):
        temp, out, result = self.run_pipeline({})
        self.addCleanup(temp.cleanup)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (out / "crm_features.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0])
        self.assertEqual(json.loads((out / "severity_breakdown.json").read_text(encoding="utf-8")), {})
        self.assertEqual(json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))["eligible_user_count"], 0)
        self.assertIn("examples", json.loads((out / "data_quality_report.json").read_text(encoding="utf-8")))

    def test_missing_required_column_fails_preflight(self):
        temp, _, result = self.run_pipeline({"users.json": [{"_id": "a" * 24}]})
        self.addCleanup(temp.cleanup)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Preflight error: users.json missing required columns", result.stderr)

    def test_numeric_coercion_and_unique_eligible_count(self):
        user = {"_id": "user-a", "status": "active", "createdAt": "2025-01-01", "age": True, "bmi": 24}
        payment = {"userId": "user-a", "status": "successful", "amountRial": "bad", "createdAt": "2025-01-01", "paidAt": "2025-01-01"}
        temp, out, result = self.run_pipeline({"users.json": [user, dict(user)], "paymenttransactions.json": [payment]})
        self.addCleanup(temp.cleanup)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))["eligible_user_count"], 1)
        report = json.loads((out / "data_quality_report.json").read_text(encoding="utf-8"))
        fields = {example["field"] for example in report["examples"]}
        self.assertTrue({"age", "amountRial"}.issubset(fields))


class InvalidHealthWeightRegressionTests(unittest.TestCase):
    def run_invalid_config(self, weights, optimized=False):
        import yaml
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8")); cfg["health_score"]["weights"]=weights
            cp=root/"config.yaml"; cp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            cmd=[sys.executable]+(["-O"] if optimized else [])+["-m", "autocrm.pipeline", "--config",str(cp),"--base",str(root),"--out",str(root/"out"),"--dictionary",str(DICTIONARY)]
            return subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace")
    def check(self,w,msg):
        r=self.run_invalid_config(w); self.assertNotEqual(r.returncode,0); self.assertIn(msg,r.stderr+r.stdout)
    def test_missing_required_key(self): self.check({"severity":55,"count":20},"missing required weights")
    def test_boolean(self): self.check({"severity":True,"count":20,"demographics":25},"finite number")
    def test_non_numeric(self): self.check({"severity":"x","count":20,"demographics":25},"finite number")
    def test_nan_infinity(self):
        for v in (float("nan"),float("inf"),float("-inf")):
            with self.subTest(v=v): self.check({"severity":v,"count":20,"demographics":25},"finite number")
    def test_negative_out_of_range(self):
        for v in (-.1,100.1):
            with self.subTest(v=v): self.check({"severity":v,"count":20,"demographics":25},"between 0 and 1")
    def test_incorrect_sum(self): self.check({"severity":55,"count":20,"demographics":20},"sum to 100")
    def test_optimized_mode(self):
        r=self.run_invalid_config({"severity":55,"count":20,"demographics":20},True); self.assertNotEqual(r.returncode,0); self.assertIn("sum to 100",r.stderr+r.stdout)


if __name__ == "__main__":
    unittest.main()

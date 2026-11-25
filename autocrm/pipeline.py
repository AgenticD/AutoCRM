# -*- coding: utf-8 -*-
"""پایپ‌لاین ساخت ویژگی‌های مدیریت ارتباط با مشتری (هسته اجرایی پروژه).

ورودی: فایل‌های JSON — خروجی: جدول ویژگی‌ها
به‌ازای هر مشتری به‌همراه فایل‌های گزارش برای هر اجرا.

مراحل اجرا (به ترتیب):
  ۱) بارگذاری کانفیگ و استخراج همه پارامترها
  ۲) خواندن ۵ فایل خام و پیش‌اعتبارسنجی سخت‌گیرانه
  ۳) دموگرافیک: سن، شاخص توده بدنی، مانده اعتبار، قدمت ثبت‌نام
  ۴) رفتار: آپلودها، پرداخت‌ها، مصرف دستیار هوشمند (کل عمر + پنجره اخیر)
  ۵) گردش‌کار آزمایشگاه: امتیاز شدت از یافته‌های غیرعادی
  ۶) ترکیب همه ابعاد روی یک ردیف به‌ازای هر کاربر واجد شرایط
  ۷) سگمنت‌بندی رفتاری: نمره و باند و برچسب سه‌گانه و سپس سگمنت نهایی
  ۸) امتیاز سلامت و طبقه (روتین/پایش/پیگیری/...)
  ۹) روند سلامت: تاریخچه هر آزمایش + دلتا + شیب + برچسب
  ۱۰) خروجی: جدول ویژگی‌ها و فایل‌های گزارش + متادیتای اجرا

اصل طراحی: هر عددِ مؤثر بر نمره‌ها از کانفیگ می‌آید (هیچ عدد جادویی
هاردکد نیست). اعداد داخل کامنت‌ها فقط «مثال» از کانفیگ فعلی هستند.
"""
import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import segmentation as SEG
from .inputs import (
    CONFIG, DICTIONARY, INPUT_DIR, OPTIONAL_USER_COLUMNS, OUTPUT_DIR, REQUIRED_COLUMNS,
    feature_columns, load_config,
)

ap = argparse.ArgumentParser(description="AutoCRM — CRM Feature Pipeline")
ap.add_argument("--base", type=Path, default=INPUT_DIR, help="Folder containing the five input JSON files")
ap.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output folder")
ap.add_argument("--config", type=Path, default=CONFIG, help="Path to the YAML or JSON configuration")
ap.add_argument("--as-of-date", default=None, help="YYYY-MM-DD")
ap.add_argument("--dictionary", type=Path, default=DICTIONARY, help="Biomarker dictionary path")
args = ap.parse_args()

CFG = load_config(args.config)
OUTPUT_COLUMNS = feature_columns(CFG)
# اگر تاریخ مبنای تحلیل در کانفیگ بود و آرگومان نداده بودیم
if not args.as_of_date and CFG.get("technical", {}).get("as_of_date"):
    args.as_of_date = CFG["technical"]["as_of_date"]

# استخراج پارامترهای کانفیگ — پشتیبانی از ساختار تازه و قدیم
# پنجره کاربر: چند روز «تازه‌وارد» و چند روز ملاک شمارش
# تازگی: نردبان روزها (مرز تازه/ولرم/سرد) + نمره هر برچسب + آستانه باند
# مالی/بسامد: پنجره و آستانه‌های باند (به‌همراه ترکیب وزنی و سقف‌ها)
# سلامت: وزن‌های امتیاز سلامت + جدول شدت + براکت‌های سن/شاخص توده بدنی
# روند سلامت: کل بلوک روند (نقاط، پنجره‌ها، آستانه ثبات)
# فنی: تنظیمات فنی عمومی (از جمله تاریخ مبنای ثابت برای بازتولیدپذیری)
_cfg_rfm = CFG.get("rfm", {})
UW = CFG.get("user_window", _cfg_rfm.get("user_window"))
REC = CFG.get("recency", _cfg_rfm.get("recency"))
MON = CFG.get("monetary", _cfg_rfm.get("monetary"))
FREQ = CFG.get("frequency", _cfg_rfm.get("frequency"))
HEALTH_CFG = CFG.get("health", CFG.get("health_score"))
HEALTH_TREND_CFG_RAW = CFG.get("health_trend", {})
TECH = CFG.get("technical", {})

LADDER = REC["ladder_days"]            # نردبان روزها برای برچسب تازگی
R_SCORES = REC["scores"]
R_BAND_T = REC["band_thresholds"]
# تازگی: بدون صدک — فقط پله مطلق (تازه/ولرم/سرد/بدون فعالیت/تازه‌وارد)
# امتیاز مستقیم از جدول نمره‌های کانفیگ خوانده می‌شود
M_BAND_T = MON["band_thresholds"]
# صدک و پنجره مالی — از ساختار تختِ جدید کانفیگ (percent / window_days)
MON_P90_PCT = int(MON.get("percentile", MON.get("hybrid", {}).get("p90_percentile", 90)))
MON_WIN_DAYS = int(MON.get("window_days", MON.get("hybrid", {}).get("window_days", 90)))
def validate_weights(weights, name, required_keys, expected_sum=1.0, tolerance=1e-6):
    if not isinstance(weights, dict):
        raise ValueError(f"{name} must be a dictionary")
    missing = [key for key in required_keys if key not in weights]
    if missing:
        raise ValueError(f"{name} missing required weights: {', '.join(missing)}")
    values = []
    for key in required_keys:
        value = weights[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name}.{key} must be a finite number")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name}.{key} must be a finite number")
        if not 0.0 <= value <= expected_sum:
            raise ValueError(f"{name}.{key} must be between 0 and {expected_sum:g}")
        values.append(value)
    if abs(sum(values) - expected_sum) > tolerance:
        raise ValueError(f"{name} weights must sum to {expected_sum:g}")
    return values

MON_W_W, MON_L_W = validate_weights(
    MON, "monetary", ("window_weight", "lifetime_weight")
)
W_DAYS = FREQ["window_days"]
F_P90_PCT = int(FREQ.get("percentile", 90))   # صدک بسامد — فقط بین کاربران دارای آپلود
W_W, L_W = validate_weights(
    FREQ, "frequency", ("window_weight", "lifetime_weight")
)
HEALTH_WEIGHTS = HEALTH_CFG["weights"]
validate_weights(HEALTH_WEIGHTS, "health_score", ("severity", "count", "demographics"), expected_sum=100.0)
HEALTH_DEMO_CFG = HEALTH_CFG.get("demographics", {})
_trend_w = HEALTH_TREND_CFG_RAW.get("weights")
if not _trend_w or "severity" not in _trend_w or "count" not in _trend_w:
    raise ValueError("config must contain health_trend.weights.severity and count")
HEALTH_TREND_WEIGHTS = _trend_w

# لاگر — همه پیام‌های کنسول انگلیسی‌اند (فارسی در برخی کنسول‌های ویندوز خراب دیده می‌شود)
log = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="  %(message)s")
log.info(f"Config loaded: ladder={LADDER} rec_scores={R_SCORES} freq_window={W_DAYS}d w={W_W}/{L_W} mon_window={MON_WIN_DAYS}d w={MON_W_W}/{MON_L_W}")

BASE = args.base.resolve()
OUT = args.out.resolve()

# «تاریخ امروزِ تحلیل» — مبنای همه محاسبات روزشمار.
# آرگومان تاریخ مبنای برای بازتولیدپذیری است: اجرای دوباره با همان تاریخ همان خروجی را می‌دهد
# (کاربرد: ارزیابی روی داده تاریخی). پیش‌فرض = امروز واقعی سیستم.
if args.as_of_date:
    TODAY = pd.Timestamp(args.as_of_date).normalize().tz_localize(None)
else:
    TODAY = pd.Timestamp.now().normalize().tz_localize(None)

WINDOW_START = TODAY - pd.Timedelta(days=W_DAYS)
CONSIDER_DAYS = UW["consideration_days"]
NEW_USER_DAYS = UW["new_user_days"]

# ── هلپرها ───────────────────────────────────────────
def status_score(status):
    """نمره پایه (۱ تا ۳) برای وضعیت یک یافته؛ وضعیت ناشناخته → هیچ.
    جدول نمره‌ها از کانفیگ خوانده می‌شود (STATUS_SCORES که پایین‌تر ساخته می‌شود)."""
    return STATUS_SCORES.get(str(status or "").strip().casefold())
# جدول شدت:
#   سقف یافته = بیشینه نمره ممکن یک یافته (نرمال‌ساز نمره شدت)
#   جدول ضریب فاصله — هرچه مقدار بیمار از بازه مرجع دورتر، ضریب بزرگ‌تر؛
#     قالب هر ردیف [از ٪فاصله، تا ٪فاصله، ضریب] — مثال: فاصله ۱۰٪ تا ۲۵٪ → ضریب ۱.۲
# اولویت خواندن: بلوک شدتِ امتیاز سلامت، سپس بلوک فنی؛ پیش‌فرض هاردکد ممنوع.
_SEV_CFG = HEALTH_CFG.get("severity") or TECH.get("severity")
if not _SEV_CFG or "dist_mult" not in _SEV_CFG or "max_finding_score" not in _SEV_CFG:
    raise ValueError("config must contain health_score.severity.dist_mult and max_finding_score (or technical.severity)")
DIST_MULT_TBL = _SEV_CFG["dist_mult"]
MAX_FINDING = _SEV_CFG["max_finding_score"]
# نمره پایه هر یافته از روی وضعیت آن — از کانفیگ خوانده می‌شود
# (کلید «وضعیت‌ها» زیر بلوک شدت)؛ اگر کانفیگ نداشت، پیش‌فرض همین جدول استفاده می‌شود.
# این نمره پایه در امتیاز شدت با وزن بیومارکر و ضریب فاصله ضرب می‌شود.
_DEFAULT_STATUS_SCORES = {"بالا": 3, "high": 3, "پایین": 2, "low": 2, "positive": 2,
                          "در آستانه بالا": 1, "در آستانه پایین": 1}
STATUS_SCORES = {str(k).strip().casefold(): int(v)
                 for k, v in (_SEV_CFG.get("status_scores") or _DEFAULT_STATUS_SCORES).items()}

# حداقل ستون‌های لازم از هر فایل خام — پیش‌از-اجرا چک می‌شود تا
# تغییر ساختار داده زودتر و با پیام واضح خطا بدهد، نه وسط محاسبات.
# ── عیب‌یابی داده ──
# هر داده بدِ نادیده‌گرفته‌شده اینجا ثبت می‌شود (شمارش + تا ۲۰ نمونه) تا در
# گزارش کیفیت داده بیاید؛ یعنی هیچ داده‌ای بی‌صدا حذف نمی‌شود.
DIAGNOSTIC_EXAMPLE_LIMIT = 20
diagnostics = {"counts": {}, "examples": []}

def record_diagnostic(category, source, reason, field=None):
    """Record a data-quality issue without copying source values."""
    key = f"{category}:{source}:{field or ''}:{reason}"
    diagnostics["counts"][key] = diagnostics["counts"].get(key, 0) + 1
    if len(diagnostics["examples"]) < DIAGNOSTIC_EXAMPLE_LIMIT:
        diagnostics["examples"].append({
            "category": category, "source": source, "field": field, "reason": reason,
        })

def load_json(name):
    """خواندن یک فایل JSON و آماده‌سازی مقادیر:
    شناسه → رشته، تاریخ → زمان‌نشان، اعشار → عدد اعشاری.
    ساختار باید آرایه‌ای از آبجکت باشد؛ وگرنه خطای پیش‌از-اجرا می‌دهد."""
    with (BASE / name).open(encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Preflight error: {name} top-level JSON must be an array")
    columns = REQUIRED_COLUMNS[name] + (OPTIONAL_USER_COLUMNS if name == "users.json" else [])
    rows = []
    for index, doc in enumerate(data):
        if not isinstance(doc, dict):
            raise ValueError(f"Preflight error: {name} record {index} must be an object")
        doc = {key: doc[key] for key in columns if key in doc}
        for k, v in doc.items():
            if isinstance(v, dict) and "$oid" in v: doc[k] = v["$oid"]
            elif isinstance(v, dict) and "$date" in v:
                doc[k] = pd.to_datetime(v["$date"], errors="coerce", utc=True)
            elif isinstance(v, dict) and "$numberDecimal" in v:
                try: doc[k] = float(v["$numberDecimal"])
                except (TypeError, ValueError, OverflowError):
                    record_diagnostic("parse_error", name, "invalid $numberDecimal", k)
                    doc[k] = None
        rows.append(doc)
    return rows

def load_frame(name):
    """خواندن فایل خام به جدول؛ اگر ستون الزامی کم باشد خطای پیش‌از-اجرا می‌دهد."""
    data = load_json(name)
    frame = pd.DataFrame(data, columns=REQUIRED_COLUMNS[name]) if not data else pd.DataFrame(data)
    missing = [column for column in REQUIRED_COLUMNS[name] if column not in frame.columns]
    if missing:
        raise ValueError(f"Preflight error: {name} missing required columns: {', '.join(missing)}")
    if name == "users.json":
        return frame.reindex(columns=REQUIRED_COLUMNS[name] + OPTIONAL_USER_COLUMNS)
    return frame

def safe_json(t):
    """پارس متن ساخت‌یافته با تحمل خطا: متن خراب → هیچ + ثبت در گزارش کیفیت (نه کرش)."""
    try: return json.loads(t) if isinstance(t, str) and t.strip() else None
    except (json.JSONDecodeError, TypeError, ValueError):
        record_diagnostic("parse_error", "labresults.json", "invalid JSON", "text")
        return None

def coerce_numeric(series, source, field):
    """ستون را به عدد امن تبدیل می‌کند: درستی‌نما/متن نامعتبر/بی‌نهایت → تهی + ثبت تشخیصی.
    چرا؟ تبدیل معمولی درستی‌نما را بی‌صدا یک/صفر می‌کند — اینجا ممنوع است."""
    boolean_mask = series.map(lambda value: isinstance(value, (bool, np.bool_)))
    converted = pd.to_numeric(series.mask(boolean_mask), errors="coerce")
    finite_mask = pd.Series(np.isfinite(converted.to_numpy(dtype=float, na_value=np.nan)), index=series.index)
    converted = converted.where(finite_mask)
    nonempty = series.notna() & series.astype(str).str.strip().ne("")
    malformed = nonempty & (boolean_mask | converted.isna())
    for _ in range(int(malformed.sum())):
        record_diagnostic("malformed_numeric", source, "not a finite number", field)
    return converted
def num(v):
    """پارس امن عدد از مقدار خام نتیجه آزمایش (خالی/درستی‌نما/متن بی‌عدد → تهی).
    نکته: جداکننده هزار حذف می‌شود تا «۱٬۲۰۰» بی‌صدا ۱.۰ نشود؛ ممیز فارسی هم پذیرفته می‌شود."""
    if v is None: return None
    if isinstance(v, bool): return None
    if isinstance(v, (int,float)): return None if pd.isna(v) else float(v)
    if not isinstance(v, str): return None
    s=v.strip()
    if not s or s in ("-","N/A"): return None
    # "1,200" → 1.0 و "۵۵٫۵" → 55.0 پارس می‌شد (مقدار غلط بی‌صدا)
    s = s.replace("*","").replace("٫",".").rstrip(".")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    m=re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None
def parse_range(r):
    """بازه مرجع مثل «۷۰ تا ۱۰۰» یا «کمتر از ۵» یا «بیشتر از ۱۲۰» → (پایین، بالا)؛ ناتوان از پارس → (تهی، تهی)."""
    if not r or r=="N/A": return None,None
    r=r.strip()
    if r.startswith("<"):
        try: return None,float(r[1:])
        except (TypeError, ValueError, OverflowError): return None,None
    if r.startswith(">"):
        try: return float(r[1:]),None
        except (TypeError, ValueError, OverflowError): return None,None
    p=re.split(r"\s*-\s*", r)
    if len(p)==2:
        try: return float(p[0]),float(p[1])
        except (TypeError, ValueError, OverflowError): return None,None
    return None,None
def dist_pct(v, lo, hi):
    """٪ فاصله مقدار بیمار از بازه مرجع: داخل بازه = صفر؛ بیرون = نسبت به پهنای بازه.
    بازه تک‌طرفه (فقط کف یا فقط سقف) نسبت به همان مرز سنجیده می‌شود."""
    if v is None: return None
    if lo is not None and hi is not None and hi>lo:
        if v>hi: return (v-hi)/(hi-lo)*100
        if v<lo: return (lo-v)/(hi-lo)*100
        return 0
    if hi is not None and lo is None and v>hi: return (v-hi)/hi*100 if hi!=0 else None
    if lo is not None and hi is None and v<lo: return (lo-v)/lo*100 if lo!=0 else None
    return None
def dist_mult(p):
    """ضریب فاصله از جدول کانفیگ؛ خارج از جدول → ضریب سقف (کلمپ)."""
    if p is None: return 1.0
    for lo,hi,m in DIST_MULT_TBL:
        if lo <= p < hi: return m
    # حالا به آخرین ضریب کلمپ می‌شود
    return float(DIST_MULT_TBL[-1][2]) if DIST_MULT_TBL else 1.0
def bracket(v, dictionary):
    """جست‌وجو در جدول بازه‌به‌نمره؛ مقدار خارج از همه بازه‌ها → صفر."""
    for (lo,hi),s in dictionary.items():
        if lo <= v < hi: return s
    return 0

def bracket_score(value, brackets):
    """نمره ۰ تا ۱۰۰ سن/شاخص توده بدنی از براکت‌های کانفیگ [{تا، نمره}].
    مرز پایین‌باز: اولین براکتی که مقدار از سقفش کمتر باشد برنده است؛ مقدار گم → تهی."""
    # براکت: [{سقف، نمره}] — مرز پایین‌باز: نمره اولین براکتی که مقدار از سقفش کمتر باشد
    if pd.isna(value):
        return None
    for br in sorted(brackets, key=lambda x: x.get("up_to", 9999)):
        try:
            if float(value) < float(br.get("up_to")):
                return float(br.get("score", 0))
        except (TypeError, ValueError):
            continue
    # اگر بزرگتر یا مساوی آخرین سقف بود، نمره آخرین براکت
    return float(brackets[-1].get("score", 0)) if brackets else 0

# تاریخ آزمایش اگر میلادی قابل پارس باشد استفاده می‌شود، وگرنه تاریخ ثبت مبناست.

def parse_testDate(s):
    """پارس تاریخ آزمایش (فقط میلادی، هر فرمت استاندارد) → زمان‌نشان بدون منطقه زمانی؛ ناموفق → تهی."""
    # فقط میلادی — هر فرمت استاندارد زمان
    # هم پذیرفته می‌شوند و خروجی همیشه بدون منطقه زمانی است؛ وگرنه معیارهای روزانه با
    # خطای نوع کرش می‌کردند یا تاریخ آزمایش بی‌صدا نادیده گرفته می‌شد و تاریخ ثبت جایگزین می‌شد.
    if s is None or (not isinstance(s, str) and not isinstance(s, pd.Timestamp)): return None
    if pd.isna(s): return None
    try:
        if isinstance(s, pd.Timestamp):
            return s.tz_localize(None) if s.tzinfo is not None else s
        ts = pd.to_datetime(s.strip(), errors="coerce", utc=True)
        return None if pd.isna(ts) else ts.tz_localize(None)
    except (TypeError, ValueError, OverflowError):
        return None

# ── خواندن: خواندن ۵ فایل خام و فیلتر پایه ────────────
# همه ردیف‌های خراب/یتیم/تکراری همین‌جا پالایه می‌شوند تا بلوک‌های بعدی
# روی «کاربران واجد شرایط» تمیز کار کنند.
log.info(f"Loading JSON inputs as_of={TODAY.date()}")
users = load_frame("users.json")
files = load_frame("files.json")
lr = load_frame("labresults.json")
ut = load_frame("usagetokens.json")
pay = load_frame("paymenttransactions.json")

# فیلتر ۱: کاربران حذف‌شده از هر محاسبه کنار گذاشته می‌شوند
users = users[users["status"].astype(str).str.lower() != "deleted"].copy()
# فیلتر ۲: حذف تکراری با شناسه.
# بدون حذف تکراری، یک کاربر دو ردیف در جدول خروجی می‌گرفت (تست‌ها این را چک می‌کنند)
_dup_before = len(users)
users = users.drop_duplicates(subset="_id", keep="first").copy()
if _dup_before > len(users):
    record_diagnostic("duplicate_user_docs", "users.json", "duplicate records dropped (kept first)")
# فیلتر ۳: رکوردهای سایر مجموعه‌ها فقط به کاربر/فایل معتبر وصل می‌شوند (ضد یتیم)
eligible_user_count = users["_id"].astype(str).nunique()
valid_users = set(users["_id"].astype(str))
files = files[files["userId"].astype(str).isin(valid_users)].copy()
valid_files = set(files["_id"].astype(str))

ut = ut[ut["userId"].astype(str).isin(valid_users)].copy()
pay = pay[pay["userId"].astype(str).isin(valid_users)].copy()
# فقط نوع‌های مربوط به تحلیل غیرعادی‌ها نگه داشته می‌شود (بقیه رکوردهای نتایج آزمایش بی‌ربط‌اند)
lr = lr[lr["type"].isin({"abnormalities","abnormalities_extracted","abnormalities_classified","details"})].copy()
lr = lr[lr["fileId"].astype(str).isin(valid_files)].copy()

# اتصال مستقیم نتایج آزمایش به شناسه کاربر (از طریق فایل) — بعد از این، هر رکورد آزمایش صاحب دارد
fu = files[["_id","userId"]].rename(columns={"_id":"fileId"})
lr = lr.merge(fu, on="fileId", how="left")
lr = lr[lr["userId"].notna()].copy()

# ── مقادیر اختیاری سن و شاخص توده بدنی و قدمت حساب ─────────────
log.info("Demographics")
for field, lower, upper in (("age", 1, 120), ("bmi", 5, 80)):
    values = coerce_numeric(users[field], "users.json", field)
    users[field] = values.where(values.between(lower, upper))
users["createdAt"] = pd.to_datetime(users["createdAt"], errors="coerce", utc=True).dt.tz_localize(None)
users["days_since_signup"] = (TODAY - users["createdAt"]).dt.days
U = users[["_id", "age", "bmi", "days_since_signup"]].rename(columns={"_id": "userId"})

# ── آپلودها: شمارش آپلود گزارش آزمایش (کل عمر + پنجره اخیر) ──
files["createdAt"] = pd.to_datetime(files["createdAt"], errors="coerce", utc=True).dt.tz_localize(None)
files["_testDate_parsed"] = files.get("testDate", pd.Series([None]*len(files))).apply(parse_testDate)
UP = files.groupby("userId").agg(upload_count=("_id","count"), last_upload=("createdAt","max")).reset_index() if not files.empty else pd.DataFrame(columns=["userId", "upload_count", "last_upload"])
UP["last_upload"] = pd.to_datetime(UP["last_upload"], errors="coerce", utc=True).dt.tz_localize(None) if "last_upload" in UP.columns else pd.Series([], dtype="datetime64[ns]")
UP["days_since_last_upload"] = (TODAY - UP["last_upload"]).dt.days
UP = UP.drop(columns=["last_upload"])
# رویدادهای آینده (تاریخ سیستم اشتباه/داده تست) نباید به آمار «آخرین رویداد/فرکانس»
# نشت کنند — روزهای منفی از همین می‌آمد؛ پس پنجره اخیر دوطرفه بسته است:
# [امروز - پنجره آپلود، امروز]
files["_is_recent"] = (files["createdAt"] >= WINDOW_START) & (files["createdAt"] <= TODAY)
UP_W = (files[files["_is_recent"]].groupby("userId").agg(uploads_last_window=("_id","count")).reset_index()
        if not files[files["_is_recent"]].empty else pd.DataFrame(columns=["userId", "uploads_last_window"]))
UP = UP.merge(UP_W, on="userId", how="left")
UP["uploads_last_window"] = UP["uploads_last_window"].fillna(0).astype(int)

# ── پرداخت‌ها: تجمیع پرداخت‌ها (فقط تراکنش‌های موفق جمع می‌شوند) ──
pay["amountRial"] = coerce_numeric(pay["amountRial"], "paymenttransactions.json", "amountRial")
pay["createdAt"] = pd.to_datetime(pay["createdAt"], errors="coerce", utc=True).dt.tz_localize(None)
pay["paidAt"] = pd.to_datetime(pay["paidAt"], errors="coerce", utc=True).dt.tz_localize(None)
ok = pay[pay["status"].astype(str).str.strip().str.lower()=="successful"]
PS = (ok.groupby("userId").agg(total_payment_rial=("amountRial","sum"), successful_payment_count=("status","count"), last_payment=("paidAt","max")).reset_index()
      if not ok.empty else pd.DataFrame(columns=["userId", "total_payment_rial", "successful_payment_count", "last_payment"]))
PS["last_payment"] = pd.to_datetime(PS["last_payment"], errors="coerce", utc=True).dt.tz_localize(None) if "last_payment" in PS.columns else pd.Series([], dtype="datetime64[ns]")
PS["days_since_last_payment"] = (TODAY - PS["last_payment"]).dt.days
PA = (pay.groupby("userId").agg(failed_payment_count=("status", lambda x: (x.astype(str).str.strip().str.lower()=="failed").sum())).reset_index()
      if not pay.empty else pd.DataFrame(columns=["userId", "failed_payment_count"]))
PY = PS.merge(PA, on="userId", how="outer")[["userId","total_payment_rial","successful_payment_count","failed_payment_count","days_since_last_payment"]]
PY["failed_payment_count"] = PY["failed_payment_count"].fillna(0).astype(int)
PY["successful_payment_count"] = PY["successful_payment_count"].fillna(0).astype(int)
PY["total_payment_rial"] = PY["total_payment_rial"].fillna(0)
# پنجره اخیر پرداخت — ورودی بعد مالی؛ طولش از کانفیگ می‌آید (مستقل از پنجره آپلودهاست
# چون پرداخت کم‌تکرارتر از آپلود است و پنجره کوتاه بیشتر صفر می‌شد — توضیح در کانفیگ)
MON_WINDOW_START = TODAY - pd.Timedelta(days=MON_WIN_DAYS)
pay_mon_source = ok[(ok["paidAt"] >= MON_WINDOW_START) & (ok["paidAt"] <= TODAY)]
pay_mon_win = (pay_mon_source.groupby("userId").agg(total_payment_window_rial=("amountRial","sum"), payments_last_window=("status","count")).reset_index()
               if not pay_mon_source.empty else pd.DataFrame(columns=["userId", "total_payment_window_rial", "payments_last_window"]))
PY = PY.merge(pay_mon_win, on="userId", how="left")
PY["total_payment_window_rial"] = PY["total_payment_window_rial"].fillna(0)
PY["payments_last_window"] = PY["payments_last_window"].fillna(0).astype(int)

# ── مصرف دستیار هوشمند ──
# دو شمارش: تعداد درخواست = هر ردیف توکن یک درخواست؛ تعداد رویداد = چند فایل
# مجزا با دستیار پردازش شده (فایل‌های مجزا). هر دو با نسخه پنجره‌ای‌شان.
ut["createdAt"] = pd.to_datetime(ut["createdAt"], errors="coerce", utc=True).dt.tz_localize(None)
AI_tok = (ut.groupby("userId").agg(ai_request_count=("_id","count"), last_ai=("createdAt","max")).reset_index()
          if not ut.empty else pd.DataFrame(columns=["userId", "ai_request_count", "last_ai"]))
AI_tok["last_ai"] = pd.to_datetime(AI_tok["last_ai"], errors="coerce", utc=True).dt.tz_localize(None) if "last_ai" in AI_tok.columns else pd.Series([], dtype="datetime64[ns]")
AI_tok["days_since_last_ai"] = (TODAY - AI_tok["last_ai"]).dt.days
# رویداد: تعداد فایل‌های مجزا
ai_event_source = ut[ut["fileId"].notna()]
ai_recent_source = ai_event_source[ai_event_source["createdAt"].between(WINDOW_START, TODAY)]
AI_evt = ai_event_source.groupby("userId", as_index=False).agg(ai_event_count=("fileId", "nunique"))
AI_evt_recent = ai_recent_source.groupby("userId", as_index=False).agg(ai_events_last_window=("fileId", "nunique"))
# ترکیب سه شمارش مصرف هوشمند روی یک جدول — کاربر بدون رکورد بعداً صفر می‌شود (نه تهی)
AI = AI_tok.merge(AI_evt, on="userId", how="left").merge(AI_evt_recent, on="userId", how="left")
AI["ai_event_count"] = AI["ai_event_count"].fillna(0).astype(int)
AI["ai_events_last_window"] = AI["ai_events_last_window"].fillna(0).astype(int)
AI = AI[["userId","ai_request_count","ai_event_count","ai_events_last_window","days_since_last_ai"]]

# ── گردش‌کار آزمایشگاه: امتیاز شدت آزمایش‌ها ──────────────
# قلب بخش بالینی پایپ‌لاین. برای آخرین گزارش آزمایش «کامل» هر کاربر:
#   ۱) بیومارکرهای استخراج‌شده → آزموده (مخرج نسبت)
#   ۲) یافته‌های غیرعادی/مرزی → نمره پایه × وزن بیومارکر × ضریب فاصله
#   ۳) نمره شدت = بیشینه نمره یافته‌ها نرمال‌شده به ۰..۱۰۰
# جزئیات کامل هر کاربر در ردپای حسابرسی می‌رود (فایل تفکیک شدت).
log.info("Lab workflow (testDate Gregorian + dictionary)")
# آفست‌های مخلوط ساختار ناهمگن می‌ساختند و مرتب/تفریق تاریخ‌ها خطای نوع می‌داد
lr["createdAt"] = pd.to_datetime(lr["createdAt"], errors="coerce", utc=True).dt.tz_localize(None)
dictionary_path = args.dictionary.resolve()
dictionary_payload = json.loads(dictionary_path.read_text(encoding="utf-8-sig"))
# ساختار فایل: فراداده + رکوردها؛ هر رکورد یک بیومارکر مرجع با نام‌های جایگزین
# نام جایگزین: نام خام آزمایش (نرمال‌شده) → رکورد مرجع — هسته نگاشت نام‌ها
records = dictionary_payload.get("records", [])
alias_map = {}
for rec in records:
    for alias in rec.get("aliases", []):
        key = re.sub(r"\s+", " ", str(alias).replace("\n"," ")).strip().casefold()
        alias_map[key] = rec

def lookup_biomarker(name):
    """نگاشت نام خام آزمایش → رکورد canonical دیکشنری.
    اگر در alias_map نبود: یک رکورد موقت unmatched می‌سازد با id یکتای hash-based —
    id یکتا مهم است چون شمارش تست‌ها/یافته‌های غیرعادی به canonical_id تکیه دارد و
    دو نام متفاوت نباید روی هم سوار شوند."""
    cleaned = re.sub(r"\s+", " ", str(name or "").replace("\n"," ")).strip()
    key = cleaned.casefold()
    rec = alias_map.get(key)
    if rec: return rec
    fallback = re.sub(r"[^a-z0-9]+","_", key).strip("_")
    if not fallback:
        # شمارش آزمایش‌ها/یافته‌های غیرعادی خراب می‌شد — حالا هر نام مجزا شناسه مجزا می‌گیرد
        fallback = "n" + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return {"canonical_id": "unmatched_" + fallback, "severity_weight": 1.0, "exclude_from_test_count": False}

def usable_result(v): return str(v or "").strip().casefold() not in {"","-","n/a","na","none","null"}
# ↑ آیا نتیجه بیمار واقعاً یک مقدار دارد؟ (حباب‌های خالی شمرده نمی‌شوند)
def latest_per_file(frame):
    """آخرین نسخه هر fileId — برای یک گزارش آزمایش ممکن است چند رکورد
    (اصلاح‌شده/به‌روزشده) وجود داشته باشد؛ فقط آخرین انگار معتبر است."""
    if frame.empty: return frame
    return frame.sort_values(["fileId","createdAt","_id"], na_position="first").groupby("fileId", as_index=False).tail(1)

# یک گزارش «کامل» = هم استخراج دارد هم طبقه‌بندی؛ فقط این‌ها وارد تحلیل می‌شوند
classified = latest_per_file(lr[lr["type"]=="abnormalities_classified"].copy())
extracted = latest_per_file(lr[lr["type"]=="abnormalities_extracted"].copy())
completed_file_ids = set(classified["fileId"].astype(str)) & set(extracted["fileId"].astype(str))
completed_files = files[files["_id"].astype(str).isin(completed_file_ids)].copy()
# فقط آخرین گزارش کامل هر کاربر وارد شدت می‌شود (نمره لحظه‌ای سلامت)
latest_files = (completed_files.sort_values(["userId","createdAt","_id"], na_position="first").groupby("userId", as_index=False).tail(1))
LT = (completed_files.groupby("userId").agg(lab_count=("_id","nunique")).reset_index()
      if not completed_files.empty else pd.DataFrame(columns=["userId", "lab_count"]))
cls_by_file = classified.set_index(classified["fileId"].astype(str))
ext_by_file = extracted.set_index(extracted["fileId"].astype(str))


def score_lab_report(file_id):
    """Return the tested count, abnormal count, and deduplicated findings."""
    cls_obj = safe_json(cls_by_file.loc[file_id, "text"])
    ext_obj = safe_json(ext_by_file.loc[file_id, "text"])
    if not isinstance(cls_obj, dict) or not isinstance(ext_obj, list):
        return None

    tested = set()
    for item in ext_obj:
        if not isinstance(item, dict) or not item.get("name") or not usable_result(item.get("patientResult")):
            continue
        match = lookup_biomarker(item["name"])
        if not match.get("exclude_from_test_count"):
            tested.add(match["canonical_id"])

    findings = {}
    abnormal_ids = set()
    for group in ("abnormal", "borderline"):
        for item in cls_obj.get(group, []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            match = lookup_biomarker(item["name"])
            marker_id = match["canonical_id"]
            if group == "abnormal":
                abnormal_ids.add(marker_id)

            base = status_score(item.get("status"))
            if base is None:
                base = 1 if group == "borderline" else 2
            weight = float(match.get("severity_weight", 1.0))
            value = num(item.get("patientResult"))
            lo, hi = parse_range(item.get("referenceRange", ""))
            distance = dist_pct(value, lo, hi)
            multiplier = dist_mult(distance)
            weighted = round(base * weight * multiplier, 3)
            if marker_id not in findings or weighted > findings[marker_id][0]:
                findings[marker_id] = (weighted, group)
    return len(tested), len(abnormal_ids), findings


def lab_report_date(report):
    parsed = parse_testDate(report.get("testDate"))
    if parsed is not None:
        return parsed
    return pd.to_datetime(report["createdAt"], errors="coerce")


# ── حلقه اصلی شدت — به‌ازای هر آخرین گزارش کامل ──
sev_rows = []
for _, report in latest_files.iterrows():
    uid, fid = str(report["userId"]), str(report["_id"])
    scores = score_lab_report(fid)
    if scores is None:
        continue
    tested_count, abnormal_count, findings = scores
    borderline_count = sum(group == "borderline" for _, group in findings.values())
    abnormal_ratio = abnormal_count / tested_count if tested_count else np.nan
    peak = max((score for score, _ in findings.values()), default=0.0)
    # ردیف متناقض می‌شد (غیرعادی بیش از استخراج و شدت مثبت با نسبت تهی)
    # نمره شدت نهایی: بیشینه یافته‌ها ÷ سقف ممکن (سقف یافته) → ۰..۱۰۰
    # بدون آزمایش (شمارش صفر) → تهی (نه صفر! تا «بدون داده» با «سالم» اشتباه نشود)
    severity_score=(min(peak/MAX_FINDING,1.0)*100) if tested_count else np.nan
    # طبقه‌بندی چهارگانه شدت برای گزارش سریع
    tier="very_high" if severity_score>=75 else "high" if severity_score>=50 else "moderate" if severity_score>=25 else "low"
    sev_rows.append({
        "userId": uid,
        "extracted_biomarker_count": tested_count,
        "abnormal_biomarker_count": abnormal_count,
        "borderline_count": borderline_count,
        "normalized_abnormal_ratio": round(abnormal_ratio, 4),
        "peak_severity_score": round(peak, 3),
        "severity_score": round(severity_score, 2),
        "severity_tier": tier,
        "unmatched_biomarker_count": sum(key.startswith("unmatched_") for key in findings),
    })

DF_SEV=pd.DataFrame(sev_rows, columns=[
    "userId",
    "extracted_biomarker_count", "abnormal_biomarker_count", "borderline_count",
    "normalized_abnormal_ratio", "peak_severity_score", "severity_score",
    "severity_tier", "unmatched_biomarker_count",
])

# ── Merge ───────────────────────────────────────────
log.info("Merge")
df=U.copy()
for n,e in [("up",UP),("py",PY),("lab",LT),("ai",AI),("sev",DF_SEV)]:
    if e is None or e.empty or "userId" not in e.columns:
        log.info(f"  (skip {n})"); continue
    df=df.merge(e, on="userId", how="left")

for col in ["days_since_last_upload","days_since_last_ai","days_since_last_payment"]:
    if col not in df.columns: df[col]=np.nan
df["days_since_last_activity"] = df[["days_since_last_upload","days_since_last_ai","days_since_last_payment"]].min(axis=1)
# activity_basis
def activity_basis(row):
    cands = {k: row[k] for k in ["days_since_last_upload","days_since_last_ai","days_since_last_payment"] if pd.notna(row[k])}
    if not cands: return ""
    return min(cands, key=cands.get).replace("days_since_last_","")
df["activity_basis"] = df.apply(activity_basis, axis=1)

for c in ["upload_count","uploads_last_window","lab_count","ai_request_count","ai_event_count","ai_events_last_window","successful_payment_count","failed_payment_count","payments_last_window"]:
    if c in df.columns: df[c]=df[c].fillna(0).astype(int)
    else: df[c]=pd.Series(0, index=df.index, dtype="int64")
for c in ["total_payment_rial","total_payment_window_rial"]:
    if c in df.columns: df[c]=df[c].fillna(0)
    else: df[c]=0.0

# ── RFM ──
# تازگی: بدون صدک — فقط پله گرم/سرد + بدون استفاده + تازه (نردبان و نمره‌ها از کانفیگ)
# مالی: نرمال ۰ تا ۵ وزن‌دار (صدک پنجره + صدک کل‌عمر)، بدون بند ریالی
log.info("RFM (recency=ladder, no P90 | monetary=P90 normalized 60/40)")

def recency_label(days, has_any):
    if pd.isna(days):
        tenure = has_any
        if pd.notna(tenure) and tenure < NEW_USER_DAYS: return "new"
        return "never_active"
    if days <= LADDER[0]: return "fresh"
    if days <= LADDER[1]: return "warm"
    return "cold"

df["recency_label"] = df.apply(lambda r: recency_label(r["days_since_last_activity"], r["days_since_signup"]), axis=1)

# --- نمره تازگی: مستقیم از نردبان (بدون صدک) ---
# تازه/ولرم/سرد/بدون فعالیت/تازه‌وارد → نمره از جدول کانفیگ
df["recency_score"] = df["recency_label"].map(R_SCORES).fillna(0).astype(float).round(2)

# --- بسامد: صدکی (مثل مالی) — صدک فقط بین کاربرانِ دارای آپلود ---
# چرا صدک به‌جای سقف ثابت؟ سقف ثابت (مثلاً ۱۲ آپلود) عددی دلبخواه است که باید
# با رشد محصول دستی بازتنظیم شود؛ صدک خودش را با توزیع واقعی جمعیت تنظیم می‌کند
# و پرت‌ها (کاربران غول‌آسا) هم بیش از نمره ۵ نمی‌گیرند.
_up_all = df.loc[df["upload_count"] > 0, "upload_count"]
_up_win = df.loc[df["uploads_last_window"] > 0, "uploads_last_window"]
p90f_all = _up_all.quantile(F_P90_PCT / 100) if len(_up_all) else np.nan
p90f_win = _up_win.quantile(F_P90_PCT / 100) if len(_up_win) else np.nan
if (pd.isna(p90f_all) or p90f_all <= 0) and (pd.isna(p90f_win) or p90f_win <= 0):
    log.warning("  frequency: no uploaders at all — all frequency scores will be 0")
else:
    if pd.isna(p90f_all) or p90f_all <= 0:
        log.warning("  frequency: lifetime percentile dead — lifetime component will be 0 for everyone")
    if pd.isna(p90f_win) or p90f_win <= 0:
        log.warning("  frequency: window percentile dead (no uploads within window?) — window component will be 0 for everyone")

if pd.notna(p90f_all) and p90f_all > 0:
    df["freq_lifetime_component"] = (df["upload_count"].clip(upper=p90f_all) / p90f_all * 5).clip(0, 5).round(2)
else:
    df["freq_lifetime_component"] = 0.0

if pd.notna(p90f_win) and p90f_win > 0:
    df["freq_window_component"] = (df["uploads_last_window"].clip(upper=p90f_win) / p90f_win * 5).clip(0, 5).round(2)
else:
    df["freq_window_component"] = 0.0

df["frequency_score"] = (W_W * df["freq_window_component"] + L_W * df["freq_lifetime_component"]).round(2)

# --- Monetary: P90 lifetime vs P90 window ---
# صفرِ غیرپرداخت‌کننده‌ها صدک را نابود نکند و بُعد مالی از کار نیفتد.
# اگر باز هم صدک=0 (هیچ پرداختی نیست) → همه صفر، با لاگ هشدار.
_payer_all = df.loc[df["total_payment_rial"] > 0, "total_payment_rial"]
_payer_win = df.loc[df["total_payment_window_rial"] > 0, "total_payment_window_rial"]
p90m_all = _payer_all.quantile(MON_P90_PCT / 100) if len(_payer_all) else np.nan
p90m_win = _payer_win.quantile(MON_P90_PCT / 100) if len(_payer_win) else np.nan
if (pd.isna(p90m_all) or p90m_all <= 0) and (pd.isna(p90m_win) or p90m_win <= 0):
    log.warning("  monetary: no payers at all — all monetary scores will be 0")
else:
    # مرگ تک صدک (مثلاً هیچ پرداختی در پنجره) بی‌صدا سقف نمره مالی را می‌بُرید
    if pd.isna(p90m_all) or p90m_all <= 0:
        log.warning("  monetary: lifetime P90 dead — lifetime monetary component will be 0 for everyone")
    if pd.isna(p90m_win) or p90m_win <= 0:
        log.warning("  monetary: window P90 dead (no payments within window?) — window monetary component will be 0 for everyone")

if pd.notna(p90m_all) and p90m_all > 0:
    _mon_all = (df["total_payment_rial"].clip(upper=p90m_all) / p90m_all * 5).clip(0, 5)
else:
    _mon_all = pd.Series(0.0, index=df.index)

if pd.notna(p90m_win) and p90m_win > 0:
    _mon_win = (df["total_payment_window_rial"].clip(upper=p90m_win) / p90m_win * 5).clip(0, 5)
else:
    _mon_win = pd.Series(0.0, index=df.index)

df["monetary_score"] = (MON_W_W * _mon_win + MON_L_W * _mon_all).round(2)
# برای سازگاری ستون پرداختِ پنجره‌ای نگه داشته می‌شود ولی در خروجی اصلی نمی‌آید مگر نیاز باشد

df["recency_band"] = df["recency_score"].apply(lambda s: SEG.rfm_band(s, R_BAND_T))
df["frequency_band"] = df["frequency_score"].apply(lambda s: SEG.rfm_band(s, FREQ["band_thresholds"]))
# باند مالی از امتیاز مطلق — از ماژول مشترک سگمنت
df["monetary_band"] = df["monetary_score"].apply(lambda s: SEG.rfm_band(s, M_BAND_T))
# has_ever_paid
df["has_ever_paid"] = (df["total_payment_rial"] > 0).map({True:"yes", False:"no"})

df["rfm_segment"] = df.apply(SEG.segment, axis=1)

# ── Health ──────────────────────────────────────────
log.info("Health score (weights from config)")
def health(row):
    severity=row.get("severity_score"); ratio=row.get("normalized_abnormal_ratio")
    age=row.get("age"); bmi=row.get("bmi")
    if pd.isna(severity) or pd.isna(ratio):
        _miss = [n for n,v in (("severity",severity),("count",ratio)) if pd.isna(v)]
        return pd.Series({"health_score":np.nan,"health_tier":np.nan,"health_status":"Unknown","health_missing_factors":",".join(_miss) or "severity,count"})
    # هر شاخص ۰ تا ۱۰۰ — وزن از کانفیگ
    penalties={}
    # شدت ۰..۱۰۰ → * وزن شدت
    penalties["severity"] = min(max(float(severity)/100,0),1)*HEALTH_WEIGHTS["severity"]
    # نمره غیرعادی ۰ تا ۱ → × وزن شمارش
    penalties["count"] = min(max(float(ratio),0),1)*HEALTH_WEIGHTS["count"]
    # دموگرافیک: هر کدام ۰ تا ۱۰۰ از براکت، بعد میانگین → ۰ تا ۱۰۰ → × وزن دموگرافیک
    age_brackets = HEALTH_DEMO_CFG.get("age", {}).get("brackets", [])
    bmi_brackets = HEALTH_DEMO_CFG.get("bmi", {}).get("brackets", [])
    missing=[]
    if pd.notna(age) and age>0 and pd.notna(bmi) and bmi>0 and age_brackets and bmi_brackets:
        age_s = bracket_score(age, age_brackets)
        bmi_s = bracket_score(bmi, bmi_brackets)
        demo_score = (age_s + bmi_s) / 2  # هر دو ۰..۱۰۰ → میانگین ۰..۱۰۰
        penalties["demographics"] = demo_score/100 * HEALTH_WEIGHTS["demographics"]
        # جمع وزن‌ها ۱۰۰ است — بدون مخرج اضافه
        score = max(0, min(100, 100 - sum(penalties.values())))
    else:
        missing.append("demographics")
        # اگر دموگرافیک نداشت، دو وزن باقی را به ۱۰۰ بازنرمال کن
        w_s = HEALTH_WEIGHTS["severity"]; w_c = HEALTH_WEIGHTS["count"]
        tot = w_s + w_c
        # نرمالایز: هر وزن سهمش از ۱۰۰
        penalties_n = {
            "severity": min(max(float(severity)/100,0),1) * (w_s/tot*100),
            "count": min(max(float(ratio),0),1) * (w_c/tot*100),
        }
        score = max(0, min(100, 100 - sum(penalties_n.values())))
    if score>=80: tier,status=5,"Routine"
    elif score>=60: tier,status=4,"Monitor"
    elif score>=40: tier,status=3,"Follow-up"
    elif score>=20: tier,status=2,"Priority follow-up"
    else: tier,status=1,"Human review"
    return pd.Series({"health_score":round(score),"health_tier":tier,"health_status":status,"health_missing_factors":",".join(missing) or "none"})

if not df.empty:
    df = pd.concat([df, df.apply(health, axis=1)], axis=1)

# ── روند سلامت (نسخه چهار — مطابق سند پیشنهاد روند) ──
# اگر روند در کانفیگ خاموش بود، بدون ترند ادامه می‌دهد.
HEALTH_TREND_CFG = CFG.get("health_trend", {})
health_history = {}
trend_rows = []
if HEALTH_TREND_CFG.get("enabled"):
    log.info("Health trend — building per-file history")
    # — ساختار جدید: ۵ نقطه بدون روز در اسم، همه از کانفیگ —
    _pts_cfg = HEALTH_TREND_CFG.get("points", [])
    # فیلتر روشن‌ها و مرتب‌سازی بر اساس بیشینه روز
    _enabled_pts = [p for p in _pts_cfg if p.get("enabled")]
    _enabled_pts = sorted(_enabled_pts, key=lambda x: int(x.get("max_days", 0)))
    # ستون‌ها: هر نقطه ستون خودش را دارد (دلتای نقطه ۱، ۲، ...)
    _point_cols = [(p.get("name"), p.get("column"), int(p.get("max_days"))) for p in _enabled_pts]
    # برای بازه: هر نقطه بازهٔ (سقف قبلی + ۱ تا سقف فعلی) از آخرین آزمایش
    _intervals = []
    _prev_max = 0
    for _name, _col, _max in _point_cols:
        _intervals.append((_name, _col, _prev_max + 1, _max))
        _prev_max = _max
    HT_STABLE = float(HEALTH_TREND_CFG.get("stable_band", 5))
    HT_SLOPE_N = int(HEALTH_TREND_CFG.get("slope_points", 3))
    HT_MIN_POINTS = max(2, int(HEALTH_TREND_CFG.get("min_points", 2)))
    _col_cfg = HEALTH_TREND_CFG.get("column_names", {})
    COL_SLOPE = _col_cfg.get("slope", "health_slope_3pt")
    # برای ترند دموگرافیک لازم نیست — فقط آزمایش
    # برای هر فایل کامل‌شده یک سلامت محاسبه می‌کنیم
    for _, report in completed_files.sort_values(["userId","createdAt","_id"]).iterrows():
        uid, fid = str(report["userId"]), str(report["_id"])
        scores = score_lab_report(fid)
        if scores is None:
            continue
        tested_count, abnormal_count, findings = scores
        ratio=abnormal_count/tested_count if tested_count else np.nan
        peak = max((score for score, _ in findings.values()), default=0.0)
        severity_score=min(peak/MAX_FINDING,1.0)*100
        if pd.isna(severity_score) or pd.isna(ratio):
            h_score=np.nan
        else:
            # هر شاخص ۰ تا ۱۰۰ — وزن ترند جدا از هلث اصلی
            w_s = HEALTH_TREND_WEIGHTS.get("severity", 73)
            w_c = HEALTH_TREND_WEIGHTS.get("count", 27)
            # نرمالایز اگر جمع ۱۰۰ نبود
            _tot = w_s + w_c
            if abs(_tot - 100) > 1e-6:
                w_s = w_s/_tot*100
                w_c = w_c/_tot*100
            penalties = {
                "severity": min(max(float(severity_score)/100,0),1) * w_s,
                "count": min(max(float(ratio),0),1) * w_c,
            }
            h_score = max(0, min(100, 100 - sum(penalties.values())))
            h_score = round(h_score)
        report_date = lab_report_date(report)
        if uid not in health_history:
            health_history[uid]=[]
        health_history[uid].append({
            "date": report_date.date().isoformat() if pd.notna(report_date) else None,
            "health_score": float(h_score) if pd.notna(h_score) else None,
        })
    # حالا برای هر کاربر روند را از تاریخچه استخراج می‌کنیم
    # برای هر کاربر، مرتب‌سازی و محاسبه Δ و شیب
    for uid, hist in health_history.items():
        # در آن روز برنده است (مثل قاعده قاعده آخرین نسخه هر فایل). آپلود/تصحیح هم‌روز نباید در
        # شیب دوبار شمرده نشود یا شمارش نقاط و سطح اطمینان متورم نشود
        valid=[h for h in hist if h["health_score"] is not None and h["date"] is not None]
        valid=sorted(valid, key=lambda x: x["date"])  # stable: ترتیب (createdAt,_id) حلقهٔ ساخت حفظ می‌شود
        _by_date={}
        for h in valid:
            _by_date[h["date"]]=h
        valid=sorted(_by_date.values(), key=lambda x: x["date"])
        health_history[uid]=valid
        if len(valid) < HT_MIN_POINTS:
            _row = {"userId": uid, COL_SLOPE: np.nan, "health_trend_label": "insufficient_history", "health_trend_confidence": "low", "health_history_points": len(valid)}
            for _, _col, _, _ in _intervals:
                _row[_col] = np.nan
            trend_rows.append(_row)
            continue
        # — هر نقطه یک بازهٔ مجزا از آخرین آزمایش —
        last=valid[-1]
        last_date=pd.to_datetime(last["date"])
        last_score=last["health_score"]
        _deltas = {}
        for _name, _col, _mn, _mx in _intervals:
            best=None
            for h in reversed(valid[:-1]):
                d=(last_date - pd.to_datetime(h["date"])).days
                if _mn <= d <= _mx:
                    best=h
                    break
            _deltas[_col]=round(float(last_score - best["health_score"]),2) if best is not None else np.nan
        # آستانه حداقل گستره برای اعتبار شیب — کمتر از این، شیب ناپایدار است → تهی (نه عدد گمراه‌کننده)
        _min_span = float(HEALTH_TREND_CFG.get("min_slope_span_days", 7))
        pts=valid[-HT_SLOPE_N:]
        if len(pts)>=2:
            xs=np.array([(pd.to_datetime(p["date"]) - pd.to_datetime(pts[0]["date"])).days for p in pts], dtype=float)
            ys=np.array([p["health_score"] for p in pts], dtype=float)
            # رگرسیون خطی ساده روی نقاط اخیر → شیب روزانه ×۳۰ (روند ماهانه)
            # پرهیز از شیب ساختگی: گستره کوتاه‌تر از حد مجاز → تهی، نه صفر ساختگی با برچسب ثبات
            if xs[-1] < _min_span:
                slope_per_30=np.nan
            else:
                try:
                    slope_per_day=np.polyfit(xs, ys, 1)[0]
                    slope_per_30=round(float(slope_per_day*30),2)
                except (np.linalg.LinAlgError, ValueError) as _slope_err:
                    log.warning(f"  slope polyfit failed for {uid}: {_slope_err} → fallback delta/days")
                    slope_per_30=round(float((ys[-1]-ys[0])/(xs[-1] or 1)*30),2)
        else:
            slope_per_30=np.nan
        # انتخاب مرجع برچسب: اولویت با شیب (وقتی حداقل ۳ نقطه و معتبر)، وگرنه اولین دلتای موجود
        _first_col = _intervals[0][1] if _intervals else None
        _ref = _deltas.get(_first_col, np.nan) if _first_col else np.nan
        if len(valid) >= 3 and pd.notna(slope_per_30):
            ref_val = slope_per_30
        else:
            ref_val = _ref if pd.notna(_ref) else (slope_per_30 if pd.notna(slope_per_30) else np.nan)
        # برچسب نهایی با باند ثبات (از کانفیگ)؛ بدون مرجع → خنثی: ثبات
        if pd.isna(ref_val):
            label="stable"
        else:
            if ref_val >= HT_STABLE: label="improving"
            elif ref_val <= -HT_STABLE: label="worsening"
            else: label="stable"
        # اطمینان برچسب: حداقل ۳ نقطه → بالا؛ وگرنه متوسط (فقط یک دلتا پشتیبانش است)
        # (چک نوسان سطح دو هنوز پیاده نشده — رزرو شده برای نسخه بعد)
        conf="high" if len(valid)>=3 else "medium"
        _row = {"userId": uid, COL_SLOPE: slope_per_30, "health_trend_label": label, "health_trend_confidence": conf, "health_history_points": len(valid)}
        _row.update(_deltas)
        trend_rows.append(_row)
    # کاربران بدون هیچ گزارش کامل → ردیف روند خالی («تاریخچه ناکافی») تا
    # تعداد ردیف‌های خروجی با جدول ویژگی‌ها یکی بماند
    all_uids=set(df["userId"].astype(str)) if not df.empty else set()
    trended=set(t["userId"] for t in trend_rows)
    for uid in all_uids - trended:
        _row = {"userId": uid, COL_SLOPE: np.nan, "health_trend_label": "insufficient_history", "health_trend_confidence": "low", "health_history_points": 0}
        for _, _col, _, _ in _intervals:
            _row[_col] = np.nan
        trend_rows.append(_row)
    # ترکیب به جدول اصلی
    _point_col_names = [_col for _, _col, _, _ in _intervals]
    TREND_COLS=["userId"] + _point_col_names + [COL_SLOPE,"health_trend_label","health_trend_confidence","health_history_points"]
    if trend_rows:
        DF_TREND=pd.DataFrame(trend_rows)
        # اطمینان از وجود همه ستون‌ها حتی اگر بعضی تهی
        for c in TREND_COLS:
            if c not in DF_TREND.columns:
                DF_TREND[c]=np.nan
        DF_TREND=DF_TREND[TREND_COLS]
    else:
        DF_TREND=pd.DataFrame(columns=TREND_COLS)
    if not df.empty:
        df=df.merge(DF_TREND, on="userId", how="left")
    else:
        # جدول خالی است — ستون‌ها را اضافه کن ولی ردیفی اضافه نشود
        for c in TREND_COLS:
            if c != "userId" and c not in df.columns:
                df[c]=pd.Series(dtype=float) if c not in ("health_trend_label","health_trend_confidence") else pd.Series(dtype=object)
    log.info(f"  health_trend: {len([t for t in trend_rows if t['health_trend_label']!='insufficient_history'])} users with trend, points={len(_intervals)} {[_c for _,_c,_,_ in _intervals]} stable=±{HT_STABLE}")
else:
    log.info("  health_trend disabled — no trend columns")

# ── خروجی ───────────────────────────────────────────
df_crm = df.reindex(columns=OUTPUT_COLUMNS).sort_values(OUTPUT_COLUMNS)
severity_summary = Counter(df_crm.loc[df_crm.severity_score.notna(), "severity_tier"])

OUT.mkdir(parents=True, exist_ok=True)
df_crm.to_csv(OUT / "crm_features.csv", index=False, encoding="utf-8-sig")
with open(OUT / "severity_breakdown.json","w",encoding="utf-8") as f: json.dump(severity_summary,f,indent=2)
with open(OUT / "data_quality_report.json","w",encoding="utf-8") as f: json.dump(diagnostics,f,ensure_ascii=False,indent=2)

metadata={
    "as_of_date": TODAY.date().isoformat(),
    "eligible_user_count": eligible_user_count,
    "output_user_count": len(df_crm),
    "severity_audit_user_count": sum(severity_summary.values()),
    "health_weights": HEALTH_WEIGHTS,
    "health_trend": {"enabled": bool(HEALTH_TREND_CFG.get("enabled")), "users_with_history": len(health_history) if HEALTH_TREND_CFG.get("enabled") else 0, "users_with_trend": len([t for t in trend_rows if t.get("health_trend_label") not in (None, "insufficient_history")]) if HEALTH_TREND_CFG.get("enabled") else 0, "total_points": sum(len(v) for v in health_history.values()) if HEALTH_TREND_CFG.get("enabled") else 0},
    "dictionary_version": dictionary_payload.get("meta",{}).get("version") or dictionary_payload.get("dictionary_version"),
}
with open(OUT / "run_metadata.json","w",encoding="utf-8") as f: json.dump(metadata,f,ensure_ascii=False,indent=2)
log.info("Done")

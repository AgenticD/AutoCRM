# AutoCRM

AutoCRM turns local JSON records into one CRM feature row per eligible user. It combines upload, usage, and payment activity into recency, frequency, and monetary (RFM) segments. It also summarizes lab findings and checks the generated outputs against the inputs and scoring rules.

## Run the demo

With Python 3.11 or later, run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python run.py --base examples --out output --as-of-date 2026-07-22
```

The included records are synthetic. The demo produces two rows of derived features in `output/crm_features.csv`, aggregate severity counts in `output/severity_breakdown.json`, and a validation summary. The sample scoring rules are for demonstration and have no clinical validation.

## Use your own files

Point `--base` at a folder with the five JSON arrays illustrated in `examples/`. Use an empty array when an input has no records.

| File | Records |
| --- | --- |
| `users.json` | Temporary join keys, account status, and signup date |
| `files.json` | Uploaded files linked to users |
| `labresults.json` | Extracted and classified lab findings linked to files |
| `usagetokens.json` | Usage events linked to users and files |
| `paymenttransactions.json` | Payment events linked to users |

Replace user and file identifiers with consistent opaque keys before preparing inputs. Omit names, contact details, dates of birth, and other personal fields. The pipeline uses only the fields it needs and accepts optional precomputed `age` and `bmi` for scoring. Dates use Gregorian ISO formats. The same input files, scoring settings, and `--as-of-date` reproduce the same feature values. When no date is set, the run uses the current date.

`config/config.yaml` contains scoring settings. To override it, pass `--config PATH`.

The CSV has no user keys, contact details, demographics, exact dates or amounts, or free-text lab values. It still contains individual behavioral and health features, so share only the synthetic demo. Use a fresh output folder when upgrading from an earlier version that exported individual lab and history reports.

## Project layout

| Path | Purpose |
| --- | --- |
| `run.py` | Runs generation and validation in order |
| `autocrm/` | Input contract, feature generation, scoring, and validation |
| `config/` | Default settings and biomarker aliases |
| `examples/` | Synthetic demo input |
| `tests/` | Pipeline and end-to-end regression checks |

```sh
python -m unittest discover -s tests -v
```

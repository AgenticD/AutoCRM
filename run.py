"""Generate and validate CRM features from local JSON files."""

import argparse
import subprocess
import sys
from pathlib import Path

from autocrm.inputs import CONFIG, DICTIONARY, INPUT_DIR, OUTPUT_DIR, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=INPUT_DIR, help="Input JSON folder")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output folder")
    parser.add_argument("--config", type=Path, default=CONFIG, help="YAML or JSON settings")
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY, help="Biomarker dictionary")
    parser.add_argument("--as-of-date", help="Reference date (YYYY-MM-DD)")
    args = parser.parse_args()

    paths = [
        "--base", str(args.base.resolve()),
        "--out", str(args.out.resolve()),
        "--config", str(args.config.resolve()),
        "--dictionary", str(args.dictionary.resolve()),
    ]
    date = ["--as-of-date", args.as_of_date] if args.as_of_date else []
    try:
        subprocess.run(
            [sys.executable, "-m", "autocrm.pipeline", *paths, *date],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "autocrm.validate", *paths],
            cwd=ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

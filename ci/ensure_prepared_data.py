"""If DVC did not populate data/prepared, copy committed CI fixtures so train.py can run."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared"
TRAIN = PREPARED / "train.csv"
TEST = PREPARED / "test.csv"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "ci_prepared"


def main() -> None:
    if TRAIN.exists() and TEST.exists():
        print("Prepared data already present:", TRAIN)
        return
    fix_train = FIXTURE_DIR / "train.csv"
    fix_test = FIXTURE_DIR / "test.csv"
    if not fix_train.exists() or not fix_test.exists():
        raise SystemExit(
            f"Missing {TRAIN} and CI fixtures under {FIXTURE_DIR}. "
            "Run: python ci/build_ci_fixtures.py (needs local data/prepared/train.csv)."
        )
    PREPARED.mkdir(parents=True, exist_ok=True)
    shutil.copy(fix_train, TRAIN)
    shutil.copy(fix_test, TEST)
    print("Bootstrapped prepared data from CI fixtures ->", PREPARED)


if __name__ == "__main__":
    main()

"""One-off: build tests/fixtures/ci_prepared from local data/prepared/train.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "prepared" / "train.csv"
OUT = ROOT / "tests" / "fixtures" / "ci_prepared"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}; run prepare locally first.")
    df = pd.read_csv(SRC, nrows=3500)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT / "train.csv", index=False)
    test_df.to_csv(OUT / "test.csv", index=False)
    print("Wrote", OUT / "train.csv", train_df.shape)
    print("Wrote", OUT / "test.csv", test_df.shape)


if __name__ == "__main__":
    main()

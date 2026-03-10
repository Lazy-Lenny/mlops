import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare raw data for training")
    parser.add_argument(
        "--input_file",
        type=str,
        default="../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to raw CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/prepared",
        help="Directory to save prepared train/test CSV files",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split size",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_and_clean_data(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(input_file)

    # customerID не використовуємо як ознаку
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # приховані пропуски в TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # simple feature engineering
    if {"tenure", "MonthlyCharges"}.issubset(df.columns):
        df["AvgMonthlySpend"] = df["MonthlyCharges"]

    if {"TotalCharges", "tenure"}.issubset(df.columns):
        df["ChargesPerMonth_calc"] = df["TotalCharges"] / df["tenure"].replace(0, pd.NA)

    # заповнення пропусків у числових колонках
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # заповнення пропусків у категоріальних колонках
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_clean_data(args.input_file)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["Churn"] if "Churn" in df.columns else None,
    )

    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Prepared train saved to: {train_path}")
    print(f"Prepared test saved to: {test_path}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    main()
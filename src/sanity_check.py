#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys
from data_lib import load_data


def analyze_combinations(df):
    """Analyze unique identity combinations"""
    cols = ["personName", "lastName", "birthDate", "gender"]
    unique = df.select(cols).unique()
    print(f"\nUnique combinations: {len(unique):,}")
    print(f"Unique names: {df['personName'].n_unique():,}")
    return unique


def find_inconsistencies(df):
    """Find people with inconsistent data"""
    grouped = (
        df.select(["personName", "lastName", "birthDate", "gender"])
        .unique()
        .group_by(["personName", "lastName"])
        .agg(
            [
                pl.col("birthDate").n_unique().alias("bd_count"),
                pl.col("gender").n_unique().alias("gender_count"),
            ]
        )
    )
    inconsistent = grouped.filter(
        (pl.col("bd_count") > 1) | (pl.col("gender_count") > 1)
    )

    print(f"\nInconsistent people: {len(inconsistent)}")
    if len(inconsistent) > 0:
        print(inconsistent)
    return inconsistent


def check_missing(df):
    """Check missing data"""
    print("\nMissing data:")
    results = {}
    for col in ["personName", "lastName", "birthDate", "gender"]:
        nulls = df[col].null_count()
        empties = (
            df.filter(pl.col(col) == "").height
            if df[col].dtype in [pl.Utf8, pl.Categorical]
            else 0
        )
        total = nulls + empties
        results[col] = total
        print(f"{col}: {total} missing ({total/len(df)*100:.1f}%)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Dataset sanity check")
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-report", help="Output file")
    args = parser.parse_args()

    try:
        df = load_data(Path(args.parquet_dir) / "billionaires.parquet", "billionaires")
        print(f"Total records: {len(df):,}")

        unique = analyze_combinations(df)
        inconsistencies = find_inconsistencies(df)
        missing = check_missing(df)

        print(f"\nFirst {args.limit} combinations:")
        print(unique.head(args.limit))

        if args.output_report:
            path = Path(args.output_report)
            if path.suffix == ".csv":
                unique.write_csv(path)
            else:
                unique.write_parquet(path)
            print(f"\nSaved to {path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

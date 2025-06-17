#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys
from data_lib import load_billionaires_data


def analyze_combinations(df):
    """Analyze unique identity combinations"""
    identity_cols = ["personName", "lastName", "birthDate", "gender"]
    unique_combs = df.select(identity_cols).unique()
    print(f"\nUnique identity combinations: {len(unique_combs):,}")
    print(f"Unique person names: {df['personName'].n_unique():,}")
    return unique_combs


def find_inconsistencies(df):
    """Find people with inconsistent identity data"""
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

    print(f"\nPeople with inconsistent data: {len(inconsistent)}")
    if len(inconsistent) > 0:
        print(inconsistent)

    return inconsistent


def check_missing(df):
    """Check for missing data in identity fields"""
    print("\nMissing data analysis:")
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
    parser = argparse.ArgumentParser(description="Billionaires dataset sanity check")
    parser.add_argument("--parquet-dir", default="data", help="Data directory")
    parser.add_argument("--limit", type=int, default=50, help="Display limit")
    parser.add_argument("--output-report", help="Output file path")
    args = parser.parse_args()

    try:
        df = load_billionaires_data(Path(args.parquet_dir) / "billionaires.parquet")
        print(f"Total records: {len(df):,}")

        # Core analysis
        unique_combs = analyze_combinations(df)
        inconsistencies = find_inconsistencies(df)
        missing_data = check_missing(df)

        # Display results
        print(f"\nFirst {args.limit} identity combinations:")
        print(unique_combs.head(args.limit))

        # Save report if requested
        if args.output_report:
            out_path = Path(args.output_report)
            if out_path.suffix == ".csv":
                unique_combs.write_csv(out_path)
            else:
                unique_combs.write_parquet(out_path)
            print(f"\nReport saved to {out_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

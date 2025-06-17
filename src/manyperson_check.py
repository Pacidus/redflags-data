#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys
from data_lib import load_data

pl.enable_string_cache()


def find_conflicts(df):
    print(f"\n🔍 Analyzing personName → lastName relationships...")

    # Clean and aggregate
    lastname_dates = (
        df.with_columns(
            pl.when(pl.col("lastName") == "").then(None).otherwise(pl.col("lastName"))
        )
        .group_by(["personName", "lastName"])
        .agg(
            [
                pl.col("date").min().alias("first_appearance"),
                pl.col("date").max().alias("last_appearance"),
                pl.col("date").count().alias("appearance_count"),
            ]
        )
    )

    # Find conflicts
    conflicts = (
        lastname_dates.group_by("personName")
        .agg(pl.col("lastName").n_unique().alias("unique_lastnames"))
        .filter(pl.col("unique_lastnames") > 1)
        .sort("unique_lastnames", descending=True)
    )

    print(f"📊 Found {len(conflicts):,} conflicts")
    return conflicts, lastname_dates


def display_conflicts(conflicts, lastname_dates, limit=20):
    if len(conflicts) == 0:
        print("✅ No conflicts found!")
        return

    print(f"\n📋 PERSONNAME → LASTNAME CONFLICTS")
    print("=" * 80)

    for row in conflicts.head(limit).iter_rows(named=True):
        print(f"👤 {row['personName']} → {row['unique_lastnames']} lastNames:")

        details = lastname_dates.filter(pl.col("personName") == row["personName"]).sort(
            "first_appearance"
        )
        for d in details.iter_rows(named=True):
            print(
                f"   - {d['lastName']}: {d['first_appearance']} to {d['last_appearance']} ({d['appearance_count']}x)"
            )


def main():
    parser = argparse.ArgumentParser(description="Find personName/lastName conflicts")
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-report", type=str, help="Output file")
    args = parser.parse_args()

    data_path = Path(args.parquet_dir) / "billionaires.parquet"
    print(f"🔍 Processing: {data_path}")

    try:
        df = load_data(data_path, "billionaires")
        conflicts, lastname_dates = find_conflicts(df)
        display_conflicts(conflicts, lastname_dates, args.limit)

        print("\n" + "=" * 80)
        print(
            f"📊 SUMMARY: {len(conflicts):,} conflicts out of {df['personName'].n_unique():,} names"
        )
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    with pl.StringCache():
        sys.exit(0 if main() else 1)

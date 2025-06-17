#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys
from data_lib import load_data, save_data
from repairs_lib import (
    clean_whitespace_and_unknowns,
    count_0th_order_issues,
    analyze_repair_impact,
)

pl.enable_string_cache()


def process_dataset(input_path, output_path, name, dry_run=False):
    """Process one dataset using the repairs library"""
    if not input_path.exists():
        print(f"❌ {name} not found: {input_path}")
        return None

    # Load
    df = load_data(input_path, name)

    # Count issues before
    issues_before = count_0th_order_issues(df)

    # Clean using library function
    cleaned = clean_whitespace_and_unknowns(df, name)

    # Count issues after
    issues_after = count_0th_order_issues(cleaned)

    # Report changes
    ws_fixed = issues_before["whitespace"] - issues_after["whitespace"]
    unk_fixed = issues_before["unknown"] - issues_after["unknown"]
    print(f"📊 Fixed: {ws_fixed:,} whitespace, {unk_fixed:,} unknown variations")

    # Save
    if not dry_run:
        save_data(cleaned, output_path, name)
    else:
        print(f"🔍 DRY RUN - would save to {output_path}")

    return {"whitespace": ws_fixed, "unknown": unk_fixed}


def main():
    parser = argparse.ArgumentParser(
        description="0th order cleaning using repairs library"
    )
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument(
        "--dataset", choices=["billionaires", "assets", "both"], default="both"
    )
    parser.add_argument("--output-suffix", default="_0th_cleaned")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dir = Path(args.parquet_dir)

    print("🧹 0TH ORDER REPAIR (Using Repairs Library)")
    print("=" * 50)
    print("Action: Strip whitespace + unknown variations → NULL")
    print(f"Processing: {args.dataset}")
    if args.dry_run:
        print("🔒 DRY RUN MODE")

    total_ws = total_unk = 0

    # Process billionaires
    if args.dataset in ["billionaires", "both"]:
        input = dir / "billionaires.parquet"
        output = dir / f"billionaires{args.output_suffix}.parquet"
        result = process_dataset(input, output, "billionaires", args.dry_run)
        if result:
            total_ws += result["whitespace"]
            total_unk += result["unknown"]

    # Process assets
    if args.dataset in ["assets", "both"]:
        input = dir / "assets.parquet"
        output = dir / f"assets{args.output_suffix}.parquet"
        result = process_dataset(input, output, "assets", args.dry_run)
        if result:
            total_ws += result["whitespace"]
            total_unk += result["unknown"]

    # Summary
    print(f"\n📊 TOTAL FIXES")
    print("-" * 20)
    print(f"Whitespace: {total_ws:,}")
    print(f"Unknown variations→NULL: {total_unk:,}")
    print(f"Total: {total_ws + total_unk:,}")

    if not args.dry_run:
        print(f"\n✅ Files saved with '{args.output_suffix}' suffix")

    print("🧹 Data cleaned: whitespace trimmed, 'unknown' variations → NULL")


if __name__ == "__main__":
    with pl.StringCache():
        main()

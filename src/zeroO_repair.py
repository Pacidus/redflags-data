#!/usr/bin/env python3
import polars as pl
import argparse
import re
from pathlib import Path
import sys
from data_lib import load_data, save_data

pl.enable_string_cache()


def clean_dataset(df, name):
    """Clean whitespace and unknown values"""
    print(f"\n🧹 CLEANING {name.upper()}")
    print("-" * 30)

    # Get string columns
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in [pl.Utf8, pl.Categorical]
    ]

    if not string_cols:
        print("No string columns")
        return df

    print(f"Cleaning {len(string_cols)} columns")

    # Unknown patterns - only match "unknown" and "unknown_123" style
    patterns = [r"(?i)^unknown$", r"(?i)^unknown_-?\d+$"]

    def clean_value(val):
        if val is None:
            return None
        cleaned = val.strip()
        if cleaned == "":
            return None
        for pat in patterns:
            if re.search(pat, cleaned):
                return None
        return cleaned

    # Clean each column
    exprs = []
    for col in string_cols:
        dtype = df.schema[col]
        expr = (
            pl.col(col)
            .cast(pl.Utf8)
            .map_elements(clean_value, return_dtype=pl.Utf8)
            .cast(dtype)
            .alias(col)
        )
        exprs.append(expr)
        print(f"  ✓ {col}")

    # Add non-string columns
    non_string = [col for col in df.columns if col not in string_cols]
    all_exprs = [pl.col(col) for col in non_string] + exprs

    cleaned = df.select(all_exprs)
    print(f"✅ Cleaned {len(string_cols)} columns")
    return cleaned


def count_changes(original, cleaned, string_cols):
    """Count changes"""
    ws_fixes = unk_fixes = 0

    for col in string_cols:
        for orig, clean in zip(
            original.select(col).iter_rows(), cleaned.select(col).iter_rows()
        ):
            if orig[0] != clean[0]:
                if orig[0] is not None and clean[0] is None:
                    unk_fixes += 1
                elif orig[0] is not None and clean[0] is not None:
                    ws_fixes += 1

    return ws_fixes, unk_fixes


def process_dataset(input_path, output_path, name, dry_run=False):
    """Process one dataset"""
    if not input_path.exists():
        print(f"❌ {name} not found: {input_path}")
        return None

    # Load
    df = load_data(input_path, name)

    # Get string columns
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in [pl.Utf8, pl.Categorical]
    ]

    # Clean
    cleaned = clean_dataset(df, name)

    # Count changes
    ws, unk = 0, 0
    if string_cols:
        ws, unk = count_changes(df, cleaned, string_cols)
        print(f"📊 Fixed: {ws:,} whitespace, {unk:,} unknown variations")

    # Save
    if not dry_run:
        save_data(cleaned, output_path, name)
    else:
        print(f"🔍 DRY RUN - would save to {output_path}")

    return {"whitespace": ws, "unknown": unk}


def main():
    parser = argparse.ArgumentParser(description="0th order cleaning")
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument(
        "--dataset", choices=["billionaires", "assets", "both"], default="both"
    )
    parser.add_argument("--output-suffix", default="_0th_cleaned")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dir = Path(args.parquet_dir)

    print("🧹 0TH ORDER REPAIR")
    print("=" * 40)
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

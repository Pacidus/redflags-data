#!/usr/bin/env python3
import polars as pl
import argparse
import re
from pathlib import Path
import sys

pl.enable_string_cache()


def clean_dataset(df, dataset_name):
    """Clean whitespace and unknown values from a dataset"""

    print(f"\n🧹 CLEANING {dataset_name.upper()}")
    print("-" * 30)

    # Get string columns
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in [pl.Utf8, pl.Categorical]
    ]

    if not string_cols:
        print("No string columns to clean")
        return df

    print(f"Cleaning {len(string_cols)} columns")

    # Unknown patterns (exact + floating)
    unknown_patterns = [
        r"(?i)^unknown$",
        r"(?i)^unknown_-?\d+$",
        r"(?i)^n/?a$",
        r"(?i)^unk$",
        r"(?i)^\?\?\?+$",
        r"(?i)^--+$",
        r"(?i)^none$",
        r"(?i)^null$",
        r"(?i)\bunknown\b",
        r"(?i)\bunknown_-?\d+\b",
        r"(?i)\bn/?a\b",
        r"(?i)\bunk\b",
    ]

    def clean_value(value):
        """Clean a single value"""
        if value is None:
            return None

        # Strip whitespace
        cleaned = value.strip()

        # Check for unknown patterns
        if cleaned == "":
            return None

        for pattern in unknown_patterns:
            if re.search(pattern, cleaned):
                return None

        return cleaned

    # Create cleaning expressions
    cleaning_exprs = []

    for col in string_cols:
        original_dtype = df.schema[col]

        cleaned_expr = (
            pl.col(col)
            .cast(pl.Utf8)
            .map_elements(clean_value, return_dtype=pl.Utf8)
            .cast(original_dtype)
            .alias(col)
        )
        cleaning_exprs.append(cleaned_expr)
        print(f"  ✓ {col}")

    # Add non-string columns unchanged
    non_string_cols = [col for col in df.columns if col not in string_cols]
    all_exprs = [pl.col(col) for col in non_string_cols] + cleaning_exprs

    # Apply cleaning
    cleaned_df = df.select(all_exprs)

    print(f"✅ Cleaned {len(string_cols)} columns")
    return cleaned_df


def count_changes(original_df, cleaned_df, string_cols):
    """Count how many values were changed"""

    whitespace_fixes = 0
    unknown_fixes = 0

    for col in string_cols:
        for orig_row, clean_row in zip(
            original_df.select(col).iter_rows(), cleaned_df.select(col).iter_rows()
        ):
            orig_val = orig_row[0]
            clean_val = clean_row[0]

            if orig_val != clean_val:
                if orig_val is not None and clean_val is None:
                    unknown_fixes += 1
                elif orig_val is not None and clean_val is not None:
                    whitespace_fixes += 1

    return whitespace_fixes, unknown_fixes


def process_dataset(input_path, output_path, dataset_name, dry_run=False):
    """Process one dataset"""

    if not input_path.exists():
        print(f"❌ {dataset_name} not found: {input_path}")
        return None

    # Load data
    df = pl.read_parquet(input_path)
    print(f"📖 Loaded {len(df):,} records")

    # Get string columns for change counting
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in [pl.Utf8, pl.Categorical]
    ]

    # Clean data
    cleaned_df = clean_dataset(df, dataset_name)

    # Count changes
    if string_cols:
        ws_fixes, unk_fixes = count_changes(df, cleaned_df, string_cols)
        print(f"📊 Fixed: {ws_fixes:,} whitespace, {unk_fixes:,} unknown")

    # Save result
    if not dry_run:
        cleaned_df.write_parquet(
            output_path, compression="brotli", compression_level=11
        )
        print(f"💾 Saved to {output_path}")
    else:
        print(f"🔍 DRY RUN - would save to {output_path}")

    return {
        "whitespace": ws_fixes if string_cols else 0,
        "unknown": unk_fixes if string_cols else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="0th order data cleaning")
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument(
        "--dataset", choices=["billionaires", "assets", "both"], default="both"
    )
    parser.add_argument("--output-suffix", default="_0th_cleaned")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    parquet_dir = Path(args.parquet_dir)

    print("🧹 0TH ORDER REPAIR")
    print("=" * 40)
    print(f"Action: Strip whitespace + unknown → NULL")
    print(f"Processing: {args.dataset}")
    if args.dry_run:
        print("🔒 DRY RUN MODE")

    total_ws = 0
    total_unk = 0

    # Process billionaires
    if args.dataset in ["billionaires", "both"]:
        input_path = parquet_dir / "billionaires.parquet"
        output_path = parquet_dir / f"billionaires{args.output_suffix}.parquet"

        result = process_dataset(input_path, output_path, "billionaires", args.dry_run)
        if result:
            total_ws += result["whitespace"]
            total_unk += result["unknown"]

    # Process assets
    if args.dataset in ["assets", "both"]:
        input_path = parquet_dir / "assets.parquet"
        output_path = parquet_dir / f"assets{args.output_suffix}.parquet"

        result = process_dataset(input_path, output_path, "assets", args.dry_run)
        if result:
            total_ws += result["whitespace"]
            total_unk += result["unknown"]

    # Summary
    print(f"\n📊 TOTAL FIXES")
    print("-" * 20)
    print(f"Whitespace: {total_ws:,}")
    print(f"Unknown→NULL: {total_unk:,}")
    print(f"Total: {total_ws + total_unk:,}")

    if not args.dry_run:
        print(f"\n✅ Files saved with '{args.output_suffix}' suffix")

    print(f"🧹 Data cleaned: whitespace trimmed, unknown patterns → NULL")


if __name__ == "__main__":
    with pl.StringCache():
        main()

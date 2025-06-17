#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
from data_lib import load_data


def check_dataset(path, name):
    """Check dataset for 0th order issues"""
    print(f"\n {name.upper()}")
    print("-" * 30)

    df = load_data(path, name)

    # Get string columns
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in (pl.Utf8, pl.Categorical)
    ]

    if not string_cols:
        print("No string columns")
        return {"whitespace": 0, "unknown": 0, "examples": []}

    print(f"String columns: {len(string_cols)}")

    # Pattern for unknown values - only match "unknown" and "unknown_123" style
    unk_pattern = r"(?i)^(unknown|unknown_-?\d+)$"

    issues = {"whitespace": 0, "unknown": 0, "examples": []}

    for col in string_cols:
        col_str = pl.col(col).cast(pl.Utf8)

        # Whitespace
        ws = df.filter(
            col_str.is_not_null() & (col_str != col_str.str.strip_chars())
        ).height

        # Unknown variations
        unk = df.filter(
            col_str.is_not_null() & col_str.str.contains(unk_pattern)
        ).height

        if ws or unk:
            print(f"  {col}: {ws} whitespace, {unk} unknown variations")

        issues["whitespace"] += ws
        issues["unknown"] += unk

        # Examples
        if ws:
            val = (
                df.filter(col_str != col_str.str.strip_chars())
                .select(col)
                .get_column(col)
                .head(1)[0]
            )
            issues["examples"].append(f"{name}.{col}: '{val}' → '{val.strip()}'")

        if unk:
            val = (
                df.filter(col_str.str.contains(unk_pattern))
                .select(col)
                .get_column(col)
                .head(1)[0]
            )
            issues["examples"].append(
                f"{name}.{col}: '{val}' → NULL (unknown variation)"
            )

    print(f"Total issues: {issues['whitespace'] + issues['unknown']:,}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="0th order check")
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument(
        "--dataset", choices=["billionaires", "assets", "both"], default="both"
    )
    args = parser.parse_args()

    dir = Path(args.parquet_dir)
    datasets = {
        "billionaires": dir / "billionaires.parquet",
        "assets": dir / "assets.parquet",
    }

    print("0TH ORDER CHECK")
    print("=" * 40)

    results = []
    for name, path in datasets.items():
        if args.dataset in (name, "both") and path.exists():
            results.append(check_dataset(path, name))
        elif args.dataset in (name, "both"):
            print(f"\n{name.upper()} not found")
            results.append({"whitespace": 0, "unknown": 0, "examples": []})

    # Summary
    total_ws = sum(r["whitespace"] for r in results)
    total_unk = sum(r["unknown"] for r in results)
    examples = [ex for r in results for ex in r["examples"]]

    if examples:
        print("\nEXAMPLES")
        print("-" * 30)
        for ex in examples[:6]:
            print(f"  {ex}")

    print("\nSUMMARY")
    print("-" * 20)
    print(f"Whitespace: {total_ws:,}")
    print(f"Unknown variations: {total_unk:,}")
    print(f"Total: {total_ws + total_unk:,}")

    if total_ws + total_unk:
        print("\nRun: python src/zeroO_repair.py")
    else:
        print("\nClean!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
from data_lib import load_data
from repairs_lib import count_0th_order_issues

pl.enable_string_cache()


def check_dataset(path, name):
    """Check dataset for 0th order issues using repairs library"""
    print(f"\n {name.upper()}")
    print("-" * 30)

    df = load_data(path, name)

    # Use library function to count issues
    issues = count_0th_order_issues(df)

    # Get string columns for reporting
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in (pl.Utf8, pl.Categorical)
    ]

    print(f"String columns: {len(string_cols)}")
    print(f"Total issues: {issues['whitespace'] + issues['unknown']:,}")

    # Get examples
    examples = []
    if issues["whitespace"] > 0:
        # Find a whitespace example
        for col in string_cols:
            col_str = pl.col(col).cast(pl.Utf8)
            ws_rows = df.filter(
                col_str.is_not_null() & (col_str != col_str.str.strip_chars())
            ).head(1)
            if len(ws_rows) > 0:
                val = ws_rows.get_column(col)[0]
                examples.append(f"{name}.{col}: '{val}' → '{val.strip()}'")
                break

    if issues["unknown"] > 0:
        # Find an unknown example
        import re

        unk_pattern = r"(?i)^(unknown|unknown_-?\d+)$"
        for col in string_cols:
            col_str = pl.col(col).cast(pl.Utf8)
            unk_rows = df.filter(
                col_str.is_not_null() & col_str.str.contains(unk_pattern)
            ).head(1)
            if len(unk_rows) > 0:
                val = unk_rows.get_column(col)[0]
                examples.append(f"{name}.{col}: '{val}' → NULL (unknown variation)")
                break

    return {
        "whitespace": issues["whitespace"],
        "unknown": issues["unknown"],
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="0th order check using repairs library"
    )
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

    print("0TH ORDER CHECK (Using Repairs Library)")
    print("=" * 50)

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
    with pl.StringCache():
        main()

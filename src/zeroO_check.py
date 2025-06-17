#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path


def check_dataset(parquet_path, dataset_name):
    """Check one dataset for 0th order issues"""
    print(f"\n {dataset_name.upper()}")
    print("-" * 30)

    df = pl.read_parquet(parquet_path)
    print(f"Records: {len(df):,}")

    # Identify string columns
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in (pl.Utf8, pl.Categorical)
    ]

    if not string_cols:
        print("No string columns")
        return {"whitespace": 0, "unknown": 0, "examples": []}

    print(f"String columns: {len(string_cols)}")

    # Patterns for unknown values
    unknown_pattern = r"(?i)(^unknown$|^unknown_-?\d+$|^n/?a$|^unk$|^\?\?\?+$|^--+$|^none$|^null$|\bunknown\b|\bunknown_-?\d+\b|\bn/?a\b|\bunk\b)"

    issues = {"whitespace": 0, "unknown": 0, "examples": []}

    for col in string_cols:
        col_str = pl.col(col).cast(pl.Utf8)

        # Whitespace check
        ws_count = df.filter(
            col_str.is_not_null() & (col_str != col_str.str.strip_chars())
        ).height

        # Unknown check
        unk_count = df.filter(
            col_str.is_not_null() & col_str.str.contains(unknown_pattern)
        ).height

        if ws_count or unk_count:
            print(f"  {col}: {ws_count} whitespace, {unk_count} unknown")

        issues["whitespace"] += ws_count
        issues["unknown"] += unk_count

        # Collect examples
        if ws_count:
            val = (
                df.filter(col_str != col_str.str.strip_chars())
                .select(col)
                .get_column(col)
                .head(1)[0]
            )
            issues["examples"].append(
                f"{dataset_name}.{col}: '{val}' â†’ '{val.strip()}' (whitespace)"
            )

        if unk_count:
            val = (
                df.filter(col_str.str.contains(unknown_pattern))
                .select(col)
                .get_column(col)
                .head(1)[0]
            )
            issues["examples"].append(
                f"{dataset_name}.{col}: '{val}' â†’ NULL (unknown)"
            )

    print(f"Total issues: {issues['whitespace'] + issues['unknown']:,}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="0th order data quality check")
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument(
        "--dataset", choices=["billionaires", "assets", "both"], default="both"
    )
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    datasets = {
        "billionaires": parquet_dir / "billionaires.parquet",
        "assets": parquet_dir / "assets.parquet",
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

    # Combine results
    total_ws = sum(r["whitespace"] for r in results)
    total_unk = sum(r["unknown"] for r in results)
    all_examples = [ex for r in results for ex in r["examples"]]

    # Show examples
    if all_examples:
        print("\nEXAMPLES")
        print("-" * 30)
        for ex in all_examples[:6]:
            print(f"  {ex}")

    # Summary
    print("\nSUMMARY")
    print("-" * 20)
    print(f"Whitespace: {total_ws:,}")
    print(f"Unknown: {total_unk:,}")
    print(f"Total: {total_ws + total_unk:,}")

    print("\nRun: python src/zerothO_repair.py" if total_ws + total_unk else "\nClean!")


if __name__ == "__main__":
    main()

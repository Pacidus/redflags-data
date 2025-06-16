#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys

pl.enable_string_cache()


def check_dataset(parquet_path, dataset_name):
    """Check one dataset for 0th order issues"""
    
    if not parquet_path.exists():
        return None
    
    print(f"\n📊 {dataset_name.upper()}")
    print("-" * 30)
    
    df = pl.read_parquet(parquet_path)
    print(f"Records: {len(df):,}")
    
    # Get string columns
    string_cols = [col for col, dtype in df.schema.items() 
                   if dtype in [pl.Utf8, pl.Categorical]]
    
    if not string_cols:
        print("No string columns")
        return {'whitespace': 0, 'unknown': 0, 'examples': []}
    
    print(f"String columns: {len(string_cols)}")
    
    # Patterns for unknown values (exact + floating)
    unknown_pattern = r'(?i)(^unknown$|^unknown_-?\d+$|^n/?a$|^unk$|^\?\?\?+$|^--+$|^none$|^null$|\bunknown\b|\bunknown_-?\d+\b|\bn/?a\b|\bunk\b)'
    
    whitespace_total = 0
    unknown_total = 0
    examples = []
    
    for col in string_cols:
        col_expr = pl.col(col).cast(pl.Utf8)
        
        # Count whitespace
        ws = df.select(
            (col_expr.is_not_null() & (col_expr != col_expr.str.strip_chars())).sum()
        ).item()
        
        # Count unknown
        unk = df.select(
            (col_expr.is_not_null() & col_expr.str.contains(unknown_pattern)).sum()
        ).item()
        
        if ws > 0 or unk > 0:
            print(f"  {col}: {ws} whitespace, {unk} unknown")
        
        whitespace_total += ws
        unknown_total += unk
        
        # Get examples
        if ws > 0:
            ws_ex = df.filter(
                col_expr.is_not_null() & (col_expr != col_expr.str.strip_chars())
            ).select(col_expr.alias(col)).unique().head(1)
            for row in ws_ex.iter_rows():
                examples.append(f"{dataset_name}.{col}: '{row[0]}' → '{row[0].strip()}' (whitespace)")
        
        if unk > 0:
            unk_ex = df.filter(
                col_expr.is_not_null() & col_expr.str.contains(unknown_pattern)
            ).select(col_expr.alias(col)).unique().head(1)
            for row in unk_ex.iter_rows():
                examples.append(f"{dataset_name}.{col}: '{row[0]}' → NULL (unknown)")
    
    print(f"Total issues: {whitespace_total + unknown_total:,}")
    
    return {
        'whitespace': whitespace_total,
        'unknown': unknown_total, 
        'examples': examples
    }


def main():
    parser = argparse.ArgumentParser(description="0th order data quality check")
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument("--dataset", choices=["billionaires", "assets", "both"], default="both")
    
    args = parser.parse_args()
    parquet_dir = Path(args.parquet_dir)
    
    print("🔍 0TH ORDER CHECK")
    print("=" * 40)
    
    all_examples = []
    total_ws = 0
    total_unk = 0
    
    # Check datasets
    if args.dataset in ["billionaires", "both"]:
        result = check_dataset(parquet_dir / "billionaires.parquet", "billionaires")
        if result:
            total_ws += result['whitespace']
            total_unk += result['unknown']
            all_examples.extend(result['examples'])
    
    if args.dataset in ["assets", "both"]:
        result = check_dataset(parquet_dir / "assets.parquet", "assets")
        if result:
            total_ws += result['whitespace']
            total_unk += result['unknown']
            all_examples.extend(result['examples'])
    
    # Show examples
    if all_examples:
        print(f"\n🔍 EXAMPLES")
        print("-" * 30)
        for ex in all_examples[:6]:
            print(f"  {ex}")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("-" * 20)
    print(f"Whitespace: {total_ws:,}")
    print(f"Unknown: {total_unk:,}")
    print(f"Total: {total_ws + total_unk:,}")
    
    if total_ws + total_unk > 0:
        print(f"\n💡 Run: python src/zerothO_repair.py")
    else:
        print(f"\n✅ Clean!")


if __name__ == "__main__":
    with pl.StringCache():
        main()

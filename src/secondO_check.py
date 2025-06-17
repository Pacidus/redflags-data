#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys

pl.enable_string_cache()


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    print(f"📖 Loading dataset from {path}")
    df = pl.read_parquet(path)
    print(
        f"✅ Loaded {len(df):,} records | 👥 Names: {df['personName'].n_unique():,} | 📅 Dates: {df['date'].min()} to {df['date'].max()}"
    )
    return df


def analyze_fields(df):
    fields = ["countryOfCitizenship", "city", "state", "source", "industries"]
    print("\n📊 SECOND ORDER FIELDS ANALYSIS\n" + "=" * 80)
    print(
        f"🔍 Analyzing: {', '.join(fields)}\n💡 Fill with: 1. Last non-null 2. Future non-null"
    )

    # Clean data
    df_clean = df.with_columns(
        [
            pl.when(pl.col(f) == "").then(None).otherwise(pl.col(f)).alias(f)
            for f in fields
            if f in df.columns
        ]
    )

    # Missing stats
    print("\n📋 Missing Data Summary:\n" + "-" * 75)
    print(f"{'Field':<25} {'Null':<10} {'Empty':<10} {'Total Missing':<15} {'%':<8}")
    stats = {}
    for f in fields:
        if f not in df.columns:
            print(f"{f:<25} {'N/A':<10} {'N/A':<10} {'N/A':<15} {'N/A':<8}")
            continue

        nulls = df_clean[f].is_null().sum()
        empties = (
            (df_clean[f] == "").sum()
            if df_clean[f].dtype in [pl.Utf8, pl.Categorical]
            else 0
        )
        total_missing = nulls + empties
        pct = total_missing / len(df) * 100

        stats[f] = {"nulls": nulls, "empties": empties}
        print(f"{f:<25} {nulls:<10} {empties:<10} {total_missing:<15} {pct:<7.1f}%")

    return stats, df_clean, fields


def analyze_changes(df, fields):
    print("\n📈 VALUE CHANGES OVER TIME ANALYSIS\n" + "=" * 80)
    for f in fields:
        if f not in df.columns:
            continue

        # Get people with multiple values
        changed = (
            df.filter(pl.col(f).is_not_null())
            .select(["personName", f])
            .unique()
            .group_by("personName")
            .agg(pl.col(f).n_unique().alias("n"))
            .filter(pl.col("n") > 1)
        )

        total = df.filter(pl.col(f).is_not_null())["personName"].n_unique()
        n_changed = len(changed)

        print(f"\n🔍 {f}:\n   👥 With data: {total:,} | 🔄 Changed: {n_changed:,}")
        if n_changed:
            pct = n_changed / total * 100
            dist = changed.group_by("n").agg(pl.len()).sort("n")
            print(f"   📊 Changed: {pct:.1f}% | 📋 Distribution:")
            for row in dist.iter_rows():
                print(f"      {row[1]:,} people → {row[0]} values")


def show_opportunities(df, fields, n=5):
    print("\n🔧 FILLING OPPORTUNITIES ANALYSIS\n" + "=" * 80)
    print(f"🎯 Examples where filling would help")

    for f in fields:
        if f not in df.columns:
            continue

        print(f"\n📝 Field: {f}\n" + "-" * 40)
        # Fixed: Added aliases to prevent duplicate column names
        mixed = (
            df.group_by("personName")
            .agg(
                [
                    pl.col(f).is_null().any().alias("has_nulls"),
                    pl.col(f).is_not_null().any().alias("has_values"),
                ]
            )
            .filter(pl.col("has_nulls") & pl.col("has_values"))
            .head(n)
            .get_column("personName")
        )

        if mixed.is_empty():
            print("   ✅ No filling opportunities found")
            continue

        for name in mixed:
            timeline = (
                df.filter(pl.col("personName") == name).select(["date", f]).sort("date")
            )

            print(f"\n   👤 {name}:")
            vals = [
                f"{row[0]}: {row[1]}"
                for row in timeline.iter_rows()
                if row[1] is not None
            ]
            nulls = [str(row[0]) for row in timeline.iter_rows() if row[1] is None]
            print(
                f"      📅 Values: {', '.join(vals[:3])}{'...' if len(vals)>3 else ''}"
            )
            print(
                f"      ❌ Nulls: {', '.join(nulls[:3])}{'...' if len(nulls)>3 else ''}"
            )

            # Find filling examples
            last_val = None
            for row in timeline.iter_rows():
                if row[1] is not None:
                    last_val = row[1]
                elif last_val:
                    print(f"      🔄 Forward fill: {row[0]} → {last_val}")
                    break


def estimate_impact(df, fields):
    print("\n📊 REPAIR IMPACT ESTIMATION\n" + "=" * 80)
    total_before = total_fillable = 0

    for f in fields:
        if f not in df.columns:
            continue

        nulls = df[f].is_null().sum()
        # Fixed: Added aliases to prevent duplicate column names
        mixed = (
            df.group_by("personName")
            .agg(
                [
                    pl.col(f).is_null().any().alias("has_nulls"),
                    pl.col(f).is_not_null().any().alias("has_values"),
                ]
            )
            .filter(pl.col("has_nulls") & pl.col("has_values"))
        )

        # FIXED: Use implode() to resolve the deprecation warning
        person_list = mixed.get_column("personName").implode()
        fillable = df.filter(
            pl.col("personName").is_in(person_list) & pl.col(f).is_null()
        ).height

        remains = nulls - fillable
        pct = fillable / nulls * 100 if nulls else 0

        print(
            f"📝 {f}:\n   Current: {nulls:,} | Fillable: {fillable:,} ({pct:.1f}%) | Remains: {remains:,}"
        )
        total_before += nulls
        total_fillable += fillable

    print(f"\n📊 Overall:\n   Before: {total_before:,} | Fillable: {total_fillable:,}")
    if total_before:
        print(f"   Improvement: {total_fillable/total_before*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze second-order fields")
    parser.add_argument("--parquet-dir", default="data", help="Data directory")
    parser.add_argument("--sample-size", type=int, default=5, help="Example count")
    args = parser.parse_args()

    path = Path(args.parquet_dir) / "billionaires.parquet"
    print("🔍 SECOND ORDER ANALYSIS\n" + "=" * 80)
    print(
        f"📁 Dataset: {path}\n🎯 Fields: countryOfCitizenship, city, state, source, industries"
    )

    try:
        df = load_data(path)
        stats, df_clean, fields = analyze_fields(df)
        analyze_changes(df_clean, fields)
        show_opportunities(df_clean, fields, args.sample_size)
        estimate_impact(df_clean, fields)

        print("\n" + "=" * 80 + "\n📊 SUMMARY\n" + "=" * 80)
        total_missing = sum(s["nulls"] + s["empties"] for s in stats.values())
        print(f"✅ Analyzed {len(fields)} fields | Total missing: {total_missing:,}")
        print("💡 Next step: Run repair script for forward/backward filling")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    with pl.StringCache():
        success = main()
    sys.exit(0 if success else 1)

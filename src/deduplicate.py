#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys
from data_lib import load_data, save_data, get_schema

pl.enable_string_cache()


def clean_and_prepare_billionaires(df):
    """
    Clean billionaires data and prepare for deduplication.
    Remove records with missing essential identifiers.
    """
    print("🧹 Cleaning billionaires data...")

    # Remove records where BOTH personName and lastName are missing/empty
    condition = (pl.col("personName").is_null()) & (pl.col("lastName").is_null())
    df_clean = df.filter(~condition)

    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"   ⚠️  Removed {removed:,} records with missing names")

    return df_clean


def clean_and_prepare_assets(df):
    """
    Clean assets data and prepare for deduplication.
    Remove records with missing personName.
    """
    print("🧹 Cleaning assets data...")

    # Remove records with missing personName
    condition = pl.col("personName").is_null()
    df_clean = df.filter(~condition)

    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"   ⚠️  Removed {removed:,} records with missing personName")

    return df_clean


def deduplicate_billionaires(df):
    """
    Deduplicate billionaires data keeping the record with highest finalWorth.

    Deduplication key: date|personName|lastName
    Sort criterion: finalWorth (highest first)
    """
    print("🔄 Deduplicating billionaires...")

    # Create deduplication key
    df_keyed = df.with_columns(
        pl.concat_str(
            [
                pl.col("date").cast(pl.Utf8),
                pl.col("personName").fill_null(""),
            ],
            separator="|",
        ).alias("dedup_key")
    )

    # Convert finalWorth to decimal for proper sorting
    df_keyed = df_keyed.with_columns(
        pl.when(pl.col("finalWorth").is_null())
        .then(pl.lit(0).cast(pl.Decimal(precision=18, scale=8)))
        .otherwise(pl.col("finalWorth"))
        .alias("finalWorth_for_sort")
    )

    # Sort by dedup_key (ascending) and finalWorth (descending - highest first)
    df_sorted = df_keyed.sort(
        ["dedup_key", "finalWorth_for_sort"], descending=[False, True]
    )

    # Keep first record for each dedup_key (which is the one with highest finalWorth)
    df_deduped = df_sorted.unique(subset=["dedup_key"], keep="first")

    # Remove temporary columns
    df_final = df_deduped.drop(["dedup_key", "finalWorth_for_sort"])

    removed = len(df) - len(df_final)
    print(f"   ✅ Removed {removed:,} duplicate records")

    return df_final


def deduplicate_assets(df):
    """
    Deduplicate assets data keeping the record with highest numberOfShares.

    Deduplication key: date|personName|ticker|companyName|currencyCode|exchange|interactive|exchangeRate|exerciseOptionPrice
    Sort criterion: numberOfShares (highest first)
    """
    print("🔄 Deduplicating assets...")

    # Create comprehensive deduplication key
    df_keyed = df.with_columns(
        pl.concat_str(
            [
                pl.col("date").cast(pl.Utf8),
                pl.col("personName").fill_null(""),
                pl.col("ticker").fill_null(""),
                pl.col("companyName").fill_null(""),
                pl.col("currencyCode").fill_null(""),
                pl.col("exchange").fill_null(""),
                pl.col("interactive").cast(pl.Utf8).fill_null(""),
                pl.col("exchangeRate").cast(pl.Utf8).fill_null(""),
                pl.col("exerciseOptionPrice").cast(pl.Utf8).fill_null(""),
            ],
            separator="|",
        ).alias("dedup_key")
    )

    # Convert numberOfShares to decimal for proper sorting
    df_keyed = df_keyed.with_columns(
        pl.when(pl.col("numberOfShares").is_null())
        .then(pl.lit(0).cast(pl.Decimal(precision=18, scale=2)))
        .otherwise(pl.col("numberOfShares"))
        .alias("numberOfShares_for_sort")
    )

    # Sort by dedup_key (ascending) and numberOfShares (descending - highest first)
    df_sorted = df_keyed.sort(
        ["dedup_key", "numberOfShares_for_sort"], descending=[False, True]
    )

    # Keep first record for each dedup_key (which is the one with highest numberOfShares)
    df_deduped = df_sorted.unique(subset=["dedup_key"], keep="first")

    # Remove temporary columns
    df_final = df_deduped.drop(["dedup_key", "numberOfShares_for_sort"])

    removed = len(df) - len(df_final)
    print(f"   ✅ Removed {removed:,} duplicate records")

    return df_final


def process_dataset(input_path, output_path, dataset_type, dry_run=False):
    """Process one dataset with deduplication"""
    if not input_path.exists():
        print(f"❌ {dataset_type} not found: {input_path}")
        return False

    print(f"\n📊 PROCESSING {dataset_type.upper()}")
    print("=" * 60)

    # Load data using library function
    df = load_data(input_path, dataset_type)
    original_count = len(df)

    # Clean and deduplicate based on dataset type
    if dataset_type == "billionaires":
        df_clean = clean_and_prepare_billionaires(df)
        df_final = deduplicate_billionaires(df_clean)
    elif dataset_type == "assets":
        df_clean = clean_and_prepare_assets(df)
        df_final = deduplicate_assets(df_clean)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    final_count = len(df_final)
    total_removed = original_count - final_count

    print(
        f"📈 Summary: {original_count:,} → {final_count:,} records ({total_removed:,} removed)"
    )

    # Save using library function (applies proper schema and sorting)
    if not dry_run:
        save_data(df_final, output_path, dataset_type)
        print(f"💾 Saved to: {output_path}")
    else:
        print(f"🔍 DRY RUN - Would save to: {output_path}")

    return True


def analyze_duplicates(df, dataset_type):
    """Analyze duplicate patterns before deduplication"""
    print(f"\n🔍 DUPLICATE ANALYSIS - {dataset_type.upper()}")
    print("=" * 50)

    if dataset_type == "billionaires":
        # Group by dedup key to find duplicates
        duplicates = (
            df.with_columns(
                pl.concat_str(
                    [
                        pl.col("date").cast(pl.Utf8),
                        pl.col("personName").fill_null(""),
                    ],
                    separator="|",
                ).alias("dedup_key")
            )
            .group_by("dedup_key")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("personName").first().alias("person"),
                    pl.col("finalWorth").min().alias("min_worth"),
                    pl.col("finalWorth").max().alias("max_worth"),
                ]
            )
            .filter(pl.col("count") > 1)
            .sort("count", descending=True)
        )

        if len(duplicates) > 0:
            print(f"Found {len(duplicates):,} duplicate groups")
            print("Top duplicate examples:")
            for row in duplicates.head(5).iter_rows(named=True):
                print(
                    f"  👤 {row['person']}: {row['count']} records, worth {row['min_worth']} - {row['max_worth']}"
                )
        else:
            print("✅ No duplicates found")

    elif dataset_type == "assets":
        # Group by dedup key to find duplicates
        duplicates = (
            df.with_columns(
                pl.concat_str(
                    [
                        pl.col("date").cast(pl.Utf8),
                        pl.col("personName").fill_null(""),
                        pl.col("ticker").fill_null(""),
                        pl.col("companyName").fill_null(""),
                        pl.col("currencyCode").fill_null(""),
                        pl.col("exchange").fill_null(""),
                        pl.col("interactive").cast(pl.Utf8).fill_null(""),
                        pl.col("exchangeRate").cast(pl.Utf8).fill_null(""),
                        pl.col("exerciseOptionPrice").cast(pl.Utf8).fill_null(""),
                    ],
                    separator="|",
                ).alias("dedup_key")
            )
            .group_by("dedup_key")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("personName").first().alias("person"),
                    pl.col("ticker").first().alias("ticker"),
                    pl.col("numberOfShares").min().alias("min_shares"),
                    pl.col("numberOfShares").max().alias("max_shares"),
                ]
            )
            .filter(pl.col("count") > 1)
            .sort("count", descending=True)
        )

        if len(duplicates) > 0:
            print(f"Found {len(duplicates):,} duplicate groups")
            print("Top duplicate examples:")
            for row in duplicates.head(5).iter_rows(named=True):
                print(
                    f"  💰 {row['person']} - {row['ticker']}: {row['count']} records, {row['min_shares']} - {row['max_shares']} shares"
                )
        else:
            print("✅ No duplicates found")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate Forbes dataset using integrated library functions"
    )
    parser.add_argument("--parquet-dir", default="data", help="Data directory")
    parser.add_argument(
        "--dataset",
        choices=["billionaires", "assets", "both"],
        default="both",
        help="Which dataset to process",
    )
    parser.add_argument(
        "--output-suffix", default="_deduped", help="Suffix for output files"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze duplicates, don't process",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without saving"
    )
    args = parser.parse_args()

    data_dir = Path(args.parquet_dir)

    print("🔄 FORBES DATASET DEDUPLICATION")
    print("=" * 80)
    print(f"📁 Directory: {data_dir.absolute()}")
    print(f"🎯 Processing: {args.dataset}")
    print(f"🔍 Mode: {'Analysis only' if args.analyze_only else 'Full processing'}")
    if args.dry_run and not args.analyze_only:
        print("🔒 DRY RUN: Will not save files")

    # Strategy explanation
    print(f"\n📋 DEDUPLICATION STRATEGY:")
    print("🏦 Billionaires: Keep record with highest finalWorth per (date, personName)")
    print("💰 Assets: Keep record with highest numberOfShares per comprehensive key")

    success = True

    # Process billionaires
    if args.dataset in ["billionaires", "both"]:
        billionaires_input = data_dir / "billionaires.parquet"
        billionaires_output = data_dir / f"billionaires{args.output_suffix}.parquet"

        if billionaires_input.exists():
            if args.analyze_only:
                df = load_data(billionaires_input, "billionaires")
                analyze_duplicates(df, "billionaires")
            else:
                if not process_dataset(
                    billionaires_input,
                    billionaires_output,
                    "billionaires",
                    args.dry_run,
                ):
                    success = False
        else:
            print(f"❌ Billionaires file not found: {billionaires_input}")
            success = False

    # Process assets
    if args.dataset in ["assets", "both"]:
        assets_input = data_dir / "assets.parquet"
        assets_output = data_dir / f"assets{args.output_suffix}.parquet"

        if assets_input.exists():
            if args.analyze_only:
                df = load_data(assets_input, "assets")
                analyze_duplicates(df, "assets")
            else:
                if not process_dataset(
                    assets_input, assets_output, "assets", args.dry_run
                ):
                    success = False
        else:
            print(f"❌ Assets file not found: {assets_input}")
            success = False

    # Final summary
    print(f"\n{'=' * 80}")
    if args.analyze_only:
        print("✅ DUPLICATE ANALYSIS COMPLETED")
    elif success:
        print("✅ DEDUPLICATION COMPLETED")
        if not args.dry_run:
            print(f"📁 Output files created with '{args.output_suffix}' suffix")
        print(
            "🎯 Strategy: Intelligent value-based deduplication (not just exact matches)"
        )
    else:
        print("❌ DEDUPLICATION FAILED")

    return success


if __name__ == "__main__":
    with pl.StringCache():
        success = main()
    sys.exit(0 if success else 1)

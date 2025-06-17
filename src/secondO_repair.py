#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys

# Import our data library
from data_lib import load_billionaires_data, save_billionaires_data

# Enable string cache to handle categorical comparisons
pl.enable_string_cache()


def clean_empty_strings(df):
    """Convert empty strings to nulls for second order fields"""

    second_order_fields = [
        "countryOfCitizenship",
        "city",
        "state",
        "source",
        "industries",
    ]

    print(f"🧹 Converting empty strings to nulls for: {', '.join(second_order_fields)}")

    # Only process fields that exist in the dataframe
    existing_fields = [field for field in second_order_fields if field in df.columns]

    if not existing_fields:
        print(f"⚠️  No second order fields found in dataset")
        return df, []

    # Convert empty strings to nulls
    df_clean = df.with_columns(
        [
            pl.when(pl.col(field) == "")
            .then(None)
            .otherwise(pl.col(field))
            .alias(field)
            for field in existing_fields
        ]
    )

    # Count how many empty strings were converted
    for field in existing_fields:
        empty_count = df.select((pl.col(field) == "").sum()).item()
        if empty_count > 0:
            print(f"   🔄 {field}: converted {empty_count:,} empty strings to nulls")

    return df_clean, existing_fields


def apply_forward_backward_fill(df, field):
    """Apply forward and backward fill to a single field using window functions"""

    print(f"   🪟 Processing {field}...")

    # Step 1: Forward fill within each person's timeline
    df_forward = df.with_columns(
        [
            pl.col(field)
            .fill_null(strategy="forward")
            .over("personName")
            .alias(f"{field}_temp")
        ]
    )

    # Step 2: Backward fill for any remaining nulls
    df_filled = df_forward.with_columns(
        [
            pl.col(f"{field}_temp")
            .fill_null(strategy="backward")
            .over("personName")
            .alias(field)
        ]
    ).drop(f"{field}_temp")

    return df_filled


def repair_second_order_fields(df_clean, fields_to_repair):
    """Apply forward/backward fill to all second order fields"""

    print(f"\n🔧 APPLYING FORWARD/BACKWARD FILL")
    print("=" * 60)
    print(
        f"📊 Processing {len(fields_to_repair)} fields: {', '.join(fields_to_repair)}"
    )

    # Sort by personName and date first - crucial for window functions
    print(f"📊 Sorting data by personName and date...")
    df_sorted = df_clean.sort(["personName", "date"])

    # Process each field one by one
    repaired_df = df_sorted

    for i, field in enumerate(fields_to_repair, 1):
        print(f"🔄 Field {i}/{len(fields_to_repair)}: {field}")
        repaired_df = apply_forward_backward_fill(repaired_df, field)
        print(f"   ✅ Completed {field}")

    print(
        f"✅ Successfully applied forward/backward fill to all {len(fields_to_repair)} fields"
    )

    return repaired_df


def analyze_repair_results(original_df, repaired_df, fields_repaired):
    """Analyze the impact of the repair process"""

    print(f"\n📊 REPAIR IMPACT ANALYSIS")
    print("=" * 60)

    total_nulls_before = 0
    total_nulls_after = 0

    print(f"{'Field':<25} {'Before':<10} {'After':<10} {'Filled':<10} {'%Filled':<10}")
    print("-" * 70)

    for field in fields_repaired:
        nulls_before = original_df.select(pl.col(field).is_null().sum()).item()
        nulls_after = repaired_df.select(pl.col(field).is_null().sum()).item()
        filled = nulls_before - nulls_after

        percentage_filled = (filled / nulls_before * 100) if nulls_before > 0 else 0

        print(
            f"{field:<25} {nulls_before:<10,} {nulls_after:<10,} {filled:<10,} {percentage_filled:<9.1f}%"
        )

        total_nulls_before += nulls_before
        total_nulls_after += nulls_after

    total_filled = total_nulls_before - total_nulls_after
    overall_percentage = (
        (total_filled / total_nulls_before * 100) if total_nulls_before > 0 else 0
    )

    print("-" * 70)
    print(
        f"{'TOTAL':<25} {total_nulls_before:<10,} {total_nulls_after:<10,} {total_filled:<10,} {overall_percentage:<9.1f}%"
    )

    return {
        "total_nulls_before": total_nulls_before,
        "total_nulls_after": total_nulls_after,
        "total_filled": total_filled,
        "percentage_filled": overall_percentage,
    }


def show_examples(original_df, repaired_df, fields_repaired, num_examples=2):
    """Show examples of repairs made (memory-efficient version)"""

    print(f"\n🔍 REPAIR EXAMPLES")
    print("=" * 60)

    # Get a small sample of people to check for examples (exclude nulls)
    sample_people = (
        original_df.select("personName")
        .filter(pl.col("personName").is_not_null())
        .unique()
        .head(200)
        .to_series()
        .to_list()
    )

    examples_shown = 0
    max_total_examples = 5  # Limit total examples to prevent memory issues

    for field in fields_repaired:
        if examples_shown >= max_total_examples:
            break

        print(f"\n📝 Examples for {field}:")
        field_examples = 0

        # Check each person in our sample
        for person_name in sample_people:
            if field_examples >= num_examples or examples_shown >= max_total_examples:
                break

            # Skip if person_name is None to avoid comparison warnings
            if person_name is None:
                continue

            # Get just this person's data using proper filtering
            person_original = (
                original_df.filter(pl.col("personName") == person_name)
                .select(["date", field])
                .sort("date")
            )

            person_repaired = (
                repaired_df.filter(pl.col("personName") == person_name)
                .select(["date", field])
                .sort("date")
            )

            # Check if this person had any repairs
            had_repair = False
            changes = []

            for orig_row, rep_row in zip(
                person_original.iter_rows(named=True),
                person_repaired.iter_rows(named=True),
            ):
                orig_val = orig_row[field]
                rep_val = rep_row[field]

                if orig_val is None and rep_val is not None:
                    had_repair = True
                    changes.append((orig_row["date"], "NULL", rep_val))
                    if len(changes) >= 3:  # Limit changes shown per person
                        break

            # Show this person's repairs if any
            if had_repair:
                print(f"\n   👤 {person_name}:")
                for date, before, after in changes:
                    print(f"      📅 {date}: {before} → {after} ✅")

                field_examples += 1
                examples_shown += 1

        if field_examples == 0:
            print(f"   ℹ️  No examples found in sample")

    if examples_shown == 0:
        print(f"\nℹ️  No repair examples found in sample of {len(sample_people)} people")

    print(
        f"\n💡 Note: Checked sample of {len(sample_people)} people to avoid memory issues"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply second order repair using window functions (forward/backward fill)"
    )
    parser.add_argument(
        "--parquet-dir",
        default="data",
        help="Directory containing parquet files (default: data)",
    )
    parser.add_argument(
        "--output",
        default="billionaires_second_order_repaired",
        help="Output filename (without extension, default: billionaires_second_order_repaired)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=2,
        help="Number of repair examples to show per field (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without saving files",
    )

    args = parser.parse_args()

    # Setup paths
    parquet_dir = Path(args.parquet_dir)
    billionaires_path = parquet_dir / "billionaires.parquet"

    output_extension = "parquet" if args.format == "parquet" else "csv"
    output_path = parquet_dir / f"{args.output}.{output_extension}"

    print("🔧 SECOND ORDER FIELDS REPAIR (Clean Implementation)")
    print("=" * 80)
    print(f"📁 Input dataset: {billionaires_path}")
    print(f"🛠️  Target fields: countryOfCitizenship, city, state, source, industries")
    print(f"💾 Output file: {output_path}")
    print(f"🔒 Dry run: {args.dry_run}")
    print(f"🪟 Method: Window functions with forward → backward fill")
    print(f"📋 Strategy: Fill nulls with past values, then future values if needed")

    try:
        # Step 1: Load the dataset using our library
        df = load_billionaires_data(billionaires_path)

        # Step 2: Clean empty strings
        df_clean, fields_to_repair = clean_empty_strings(df)

        if not fields_to_repair:
            print("❌ No second order fields found to repair")
            return False

        # Step 3: Apply forward/backward fill
        repaired_df = repair_second_order_fields(df_clean, fields_to_repair)

        # Step 4: Analyze results
        impact_stats = analyze_repair_results(df_clean, repaired_df, fields_to_repair)

        # Step 5: Show examples
        show_examples(df_clean, repaired_df, fields_to_repair, args.examples)

        # Step 6: Save results (unless dry run)
        if not args.dry_run:
            if args.format == "parquet":
                # Use our library function for parquet (with proper sorting and compression)
                save_billionaires_data(repaired_df, output_path)
            else:
                # For CSV, write directly
                repaired_df.write_csv(output_path)
                print(f"✅ Saved {len(repaired_df):,} records to {output_path}")
        else:
            print(
                f"\n🔍 DRY RUN - Would save {len(repaired_df):,} records to {output_path}"
            )

        # Step 7: Final summary
        print("\n" + "=" * 80)
        print("✅ SECOND ORDER REPAIR COMPLETED")
        print("=" * 80)
        print(f"📊 Total records: {len(repaired_df):,} (unchanged)")
        print(
            f"🔧 Fields repaired: {len(fields_to_repair)} ({', '.join(fields_to_repair)})"
        )
        print(f"📈 Total nulls filled: {impact_stats['total_filled']:,}")
        print(f"📊 Overall improvement: {impact_stats['percentage_filled']:.1f}%")
        print(f"✅ All time-series and identity data preserved")

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

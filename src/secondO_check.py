#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys

# Enable string cache to handle categorical comparisons
pl.enable_string_cache()


def load_billionaires_data(parquet_path):
    """Load billionaires dataset from parquet file"""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    print(f"📖 Loading billionaires dataset from {parquet_path}")
    df = pl.read_parquet(parquet_path)
    print(f"✅ Loaded {len(df):,} records")
    print(f"👥 Unique person names: {df['personName'].n_unique():,}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def analyze_second_order_fields(df):
    """Analyze missing data patterns in second order fields"""

    # Second order fields that can change through time
    second_order_fields = [
        "countryOfCitizenship",
        "city",
        "state",
        "source",
        "industries",
    ]

    print(f"\n📊 SECOND ORDER FIELDS ANALYSIS")
    print("=" * 80)
    print(f"🔍 Analyzing fields: {', '.join(second_order_fields)}")
    print(f"💡 These fields can change over time, so nulls should be filled with:")
    print(f"   1. Last past non-null value (forward fill)")
    print(f"   2. If no past value, first future non-null value (backward fill)")

    # Clean up empty strings first
    df_clean = df.with_columns(
        [
            pl.when(pl.col(field) == "")
            .then(None)
            .otherwise(pl.col(field))
            .alias(field)
            for field in second_order_fields
            if field in df.columns
        ]
    )

    missing_stats = {}

    print(f"\n📋 Missing Data Summary:")
    print(f"{'Field':<25} {'Null':<10} {'Empty':<10} {'Total Missing':<15} {'%':<8}")
    print("-" * 75)

    for field in second_order_fields:
        if field not in df.columns:
            print(f"{field:<25} {'N/A':<10} {'N/A':<10} {'N/A':<15} {'N/A':<8}")
            continue

        null_count = df.select(pl.col(field).is_null().sum()).item()

        # Check for empty strings
        col_dtype = df.schema[field]
        if col_dtype in [pl.Utf8, pl.Categorical]:
            empty_count = df.select((pl.col(field) == "").sum()).item()
        else:
            empty_count = 0

        total_missing = null_count + empty_count
        percentage = (total_missing / len(df)) * 100

        missing_stats[field] = {
            "null": null_count,
            "empty": empty_count,
            "total_missing": total_missing,
            "percentage": percentage,
        }

        print(
            f"{field:<25} {null_count:<10} {empty_count:<10} {total_missing:<15} {percentage:<7.1f}%"
        )

    return missing_stats, df_clean, second_order_fields


def analyze_value_changes_over_time(df_clean, second_order_fields):
    """Analyze how second order fields change over time for people"""

    print(f"\n📈 VALUE CHANGES OVER TIME ANALYSIS")
    print("=" * 80)

    change_stats = {}

    for field in second_order_fields:
        if field not in df_clean.columns:
            continue

        print(f"\n🔍 Analyzing {field}:")

        # Find people who have multiple different values for this field
        field_variations = (
            df_clean.select(["personName", field])
            .filter(pl.col(field).is_not_null())  # Only non-null values
            .unique()
            .group_by("personName")
            .agg(
                [
                    pl.col(field).n_unique().alias("unique_values"),
                    pl.col(field).alias("all_values"),
                ]
            )
            .filter(pl.col("unique_values") > 1)
            .sort("unique_values", descending=True)
        )

        total_people_with_data = (
            df_clean.select(["personName", field])
            .filter(pl.col(field).is_not_null())
            .select("personName")
            .n_unique()
        )

        people_with_changes = len(field_variations)

        print(f"   👥 People with non-null {field}: {total_people_with_data:,}")
        print(f"   🔄 People with multiple values: {people_with_changes:,}")

        if people_with_changes > 0:
            percentage = (people_with_changes / total_people_with_data) * 100
            print(f"   📊 Percentage with changes: {percentage:.1f}%")

            # Show distribution of change counts
            change_distribution = (
                field_variations.group_by("unique_values")
                .agg(pl.len().alias("count"))
                .sort("unique_values")
            )

            print(f"   📋 Change distribution:")
            for row in change_distribution.iter_rows(named=True):
                print(
                    f"      {row['count']:,} people have {row['unique_values']} different values"
                )

        change_stats[field] = {
            "total_people": total_people_with_data,
            "people_with_changes": people_with_changes,
            "field_variations": field_variations,
        }

    return change_stats


def show_filling_opportunities(df_clean, second_order_fields, sample_size=5):
    """Show examples of records that would benefit from forward/backward filling"""

    print(f"\n🔧 FILLING OPPORTUNITIES ANALYSIS")
    print("=" * 80)
    print(f"🎯 Showing examples where forward/backward fill would help")

    for field in second_order_fields:
        if field not in df_clean.columns:
            continue

        print(f"\n📝 Field: {field}")
        print("-" * 40)

        # Find people who have null values but also have non-null values
        people_with_mixed_data = (
            df_clean.select(["personName", "date", field])
            .group_by("personName")
            .agg(
                [
                    pl.col(field).is_null().any().alias("has_nulls"),
                    pl.col(field).is_not_null().any().alias("has_values"),
                ]
            )
            .filter(pl.col("has_nulls") & pl.col("has_values"))
            .select("personName")
            .head(sample_size)
        )

        if len(people_with_mixed_data) == 0:
            print(
                "   ✅ No filling opportunities found (all people have consistent data)"
            )
            continue

        for person_row in people_with_mixed_data.iter_rows(named=True):
            person_name = person_row["personName"]

            # Get this person's timeline for this field
            person_timeline = (
                df_clean.filter(pl.col("personName") == person_name)
                .select(["date", field])
                .sort("date")
            )

            print(f"\n   👤 {person_name}:")

            # Show timeline
            null_dates = []
            value_dates = []

            for row in person_timeline.iter_rows(named=True):
                date = row["date"]
                value = row[field]

                if value is None:
                    null_dates.append(str(date))
                else:
                    value_dates.append(f"{date}: {value}")

            print(
                f"      📅 Dates with values: {', '.join(value_dates[:3])}{'...' if len(value_dates) > 3 else ''}"
            )
            print(
                f"      ❌ Dates with nulls: {', '.join(null_dates[:3])}{'...' if len(null_dates) > 3 else ''}"
            )

            # Show what forward/backward fill would do
            forward_fill_example = None
            backward_fill_example = None

            timeline_data = person_timeline.to_dicts()

            # Simulate forward fill
            last_value = None
            for i, record in enumerate(timeline_data):
                if record[field] is not None:
                    last_value = record[field]
                elif last_value is not None and forward_fill_example is None:
                    forward_fill_example = (
                        f"{record['date']}: NULL → {last_value} (forward fill)"
                    )

            # Simulate backward fill
            next_value = None
            for i in range(len(timeline_data) - 1, -1, -1):
                record = timeline_data[i]
                if record[field] is not None:
                    next_value = record[field]
                elif next_value is not None and backward_fill_example is None:
                    backward_fill_example = (
                        f"{record['date']}: NULL → {next_value} (backward fill)"
                    )

            if forward_fill_example:
                print(f"      🔄 Forward fill example: {forward_fill_example}")
            if backward_fill_example:
                print(f"      🔙 Backward fill example: {backward_fill_example}")


def estimate_repair_impact(df_clean, second_order_fields):
    """Estimate how many nulls would be filled by the repair process"""

    print(f"\n📊 REPAIR IMPACT ESTIMATION")
    print("=" * 80)

    total_nulls_before = 0
    total_nulls_fillable = 0

    for field in second_order_fields:
        if field not in df_clean.columns:
            continue

        # Count current nulls
        current_nulls = df_clean.select(pl.col(field).is_null().sum()).item()

        # Estimate how many could be filled
        people_with_mixed_data = (
            df_clean.select(["personName", field])
            .group_by("personName")
            .agg(
                [
                    pl.col(field).is_null().any().alias("has_nulls"),
                    pl.col(field).is_not_null().any().alias("has_values"),
                ]
            )
            .filter(pl.col("has_nulls") & pl.col("has_values"))
        )

        # Count fillable nulls (rough estimate)
        fillable_nulls = 0
        for person_row in people_with_mixed_data.iter_rows(named=True):
            person_name = person_row["personName"]
            person_nulls = (
                df_clean.filter(pl.col("personName") == person_name)
                .select(pl.col(field).is_null().sum())
                .item()
            )
            fillable_nulls += person_nulls

        percentage_fillable = (
            (fillable_nulls / current_nulls * 100) if current_nulls > 0 else 0
        )

        print(f"📝 {field}:")
        print(f"   Current nulls: {current_nulls:,}")
        print(
            f"   Potentially fillable: {fillable_nulls:,} ({percentage_fillable:.1f}%)"
        )
        print(f"   Would remain null: {current_nulls - fillable_nulls:,}")

        total_nulls_before += current_nulls
        total_nulls_fillable += fillable_nulls

    print(f"\n📊 Overall Impact:")
    print(f"   Total nulls before repair: {total_nulls_before:,}")
    print(f"   Estimated fillable nulls: {total_nulls_fillable:,}")
    if total_nulls_before > 0:
        print(
            f"   Overall improvement: {(total_nulls_fillable / total_nulls_before * 100):.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze second order fields for forward/backward fill repair opportunities"
    )
    parser.add_argument(
        "--parquet-dir",
        default="data",
        help="Directory containing parquet files (default: data)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of examples to show per field (default: 5)",
    )

    args = parser.parse_args()

    # Setup paths
    parquet_dir = Path(args.parquet_dir)
    billionaires_path = parquet_dir / "billionaires.parquet"

    print("🔍 SECOND ORDER FIELDS ANALYSIS")
    print("=" * 80)
    print(f"📁 Dataset path: {billionaires_path}")
    print(f"🎯 Analyzing fields that can change over time")
    print(f"📋 Fields: countryOfCitizenship, city, state, source, industries")

    try:
        # Load the dataset
        df = load_billionaires_data(billionaires_path)

        # Analyze missing data
        missing_stats, df_clean, second_order_fields = analyze_second_order_fields(df)

        # Analyze value changes over time
        change_stats = analyze_value_changes_over_time(df_clean, second_order_fields)

        # Show filling opportunities
        show_filling_opportunities(df_clean, second_order_fields, args.sample_size)

        # Estimate repair impact
        estimate_repair_impact(df_clean, second_order_fields)

        # Summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(
            f"✅ Analysis completed for {len(second_order_fields)} second order fields"
        )

        total_missing = sum(stats["total_missing"] for stats in missing_stats.values())
        print(f"📊 Total missing values across all fields: {total_missing:,}")

        fields_with_changes = sum(
            1
            for field, stats in change_stats.items()
            if stats["people_with_changes"] > 0
        )
        print(
            f"🔄 Fields showing value changes over time: {fields_with_changes}/{len(second_order_fields)}"
        )

        print(f"\n💡 Recommended next step:")
        print(
            f"   Run the second order repair script to apply forward/backward filling"
        )

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

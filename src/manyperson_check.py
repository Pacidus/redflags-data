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


def find_personname_conflicts(df):
    """Find personName values that have multiple lastName values with first appearance dates"""

    print(f"\n🔍 Analyzing personName → lastName relationships...")

    # Clean up empty strings first
    df_clean = df.with_columns(
        [
            pl.when(pl.col("lastName") == "")
            .then(None)
            .otherwise(pl.col("lastName"))
            .alias("lastName")
        ]
    )

    # Get unique combinations with first appearance date for each lastName
    lastname_first_dates = df_clean.group_by(["personName", "lastName"]).agg(
        [
            pl.col("date").min().alias("first_appearance"),
            pl.col("date").max().alias("last_appearance"),
            pl.col("date").count().alias("appearance_count"),
        ]
    )

    # Group by personName and check for multiple lastName values
    conflicts = (
        lastname_first_dates.group_by("personName")
        .agg(
            [
                pl.col("lastName").n_unique().alias("unique_lastnames"),
                pl.col("lastName").alias("all_lastnames"),
                pl.col("first_appearance").alias("all_first_dates"),
                pl.col("last_appearance").alias("all_last_dates"),
                pl.col("appearance_count").alias("all_appearance_counts"),
            ]
        )
        .filter(pl.col("unique_lastnames") > 1)
        .sort("unique_lastnames", descending=True)
    )

    print(
        f"📊 Found {len(conflicts):,} personName values with multiple lastName values"
    )

    return conflicts, lastname_first_dates


def analyze_conflicts_detail(df, conflicts):
    """Provide detailed analysis of the conflicts"""

    print(f"\n📋 CONFLICT DETAILS:")
    print("=" * 80)

    if len(conflicts) == 0:
        print(
            "✅ No conflicts found! All personName values have consistent lastName values."
        )
        return

    # Show statistics
    max_lastnames = conflicts["unique_lastnames"].max()
    avg_lastnames = conflicts["unique_lastnames"].mean()

    print(f"📊 Conflict Statistics:")
    print(f"   Maximum lastName values for one personName: {max_lastnames}")
    print(f"   Average lastName values per conflicted personName: {avg_lastnames:.1f}")

    # Show distribution
    distribution = (
        conflicts.group_by("unique_lastnames")
        .agg(pl.count().alias("count"))
        .sort("unique_lastnames")
    )

    print(f"\n📊 Distribution of conflicts:")
    for row in distribution.iter_rows(named=True):
        print(
            f"   {row['count']:,} personName(s) have {row['unique_lastnames']} different lastName values"
        )


def display_conflicts(
    df, conflicts, lastname_first_dates, limit=20, show_examples=True
):
    """Display the conflicts in a readable format with chronological information"""

    if len(conflicts) == 0:
        return

    print(f"\n📋 PERSONNAME → LASTNAME CONFLICTS (with chronological data)")
    print("=" * 80)

    display_conflicts_df = conflicts.head(limit) if limit else conflicts

    print(
        f"Showing {'first ' + str(limit) if limit and len(conflicts) > limit else 'all'} "
        f"{len(display_conflicts_df):,} conflicts:\n"
    )

    for i, row in enumerate(display_conflicts_df.iter_rows(named=True), 1):
        person_name = row["personName"] if row["personName"] else "NULL"
        lastname_count = row["unique_lastnames"]

        print(f"{i:2d}. 👤 {person_name}")
        print(f"    📝 {lastname_count} different lastName values:")

        # Get detailed chronological data for this personName
        person_filter = (
            pl.col("personName") == row["personName"]
            if row["personName"]
            else pl.col("personName").is_null()
        )
        person_lastname_details = lastname_first_dates.filter(person_filter).sort(
            "first_appearance"
        )  # Sort by first appearance date

        print(f"    📅 Chronological lastName usage:")
        for detail in person_lastname_details.iter_rows(named=True):
            last_name = detail["lastName"] if detail["lastName"] else "NULL"
            first_date = detail["first_appearance"]
            last_date = detail["last_appearance"]
            count = detail["appearance_count"]

            if first_date == last_date:
                date_range = f"{first_date}"
            else:
                date_range = f"{first_date} to {last_date}"

            print(
                f"       → {last_name}: first used {first_date}, last used {last_date} ({count:,} times)"
            )

        # Calculate the time gap between first lastName usage
        first_dates = [
            detail["first_appearance"]
            for detail in person_lastname_details.iter_rows(named=True)
        ]
        if len(first_dates) >= 2:
            time_gap = (first_dates[1] - first_dates[0]).days
            print(f"    ⏱️  Time gap between first two lastName usages: {time_gap} days")

            if time_gap == 0:
                print(
                    f"    🚨 Multiple lastName values used on the same first date - likely different people!"
                )
            elif time_gap < 30:
                print(
                    f"    ⚠️  Very short gap ({time_gap} days) - possible data entry error or rapid name change"
                )
            elif time_gap > 365:
                print(
                    f"    ℹ️  Long gap ({time_gap} days) - possible legitimate name change over time"
                )

        if show_examples:
            # Show a few example records for this personName
            examples = (
                df.filter(person_filter)
                .select(
                    [
                        "personName",
                        "lastName",
                        "date",
                        "finalWorth",
                        "countryOfCitizenship",
                        "birthDate",
                        "gender",
                    ]
                )
                .unique(subset=["lastName"])
                .sort("lastName")
                .head(3)  # Show max 3 examples per lastName
            )

            print("    🔍 Sample records:")
            for example in examples.iter_rows(named=True):
                last_name = example["lastName"] if example["lastName"] else "NULL"
                country = (
                    example["countryOfCitizenship"]
                    if example["countryOfCitizenship"]
                    else "Unknown"
                )
                worth = (
                    f"${example['finalWorth']:,}M"
                    if example["finalWorth"]
                    else "Unknown"
                )
                birth_date = example["birthDate"] if example["birthDate"] else "Unknown"
                gender = example["gender"] if example["gender"] else "Unknown"
                print(
                    f"       → {person_name} {last_name} (born: {birth_date}, {gender}, {country}, {worth})"
                )

        print()

    if limit and len(conflicts) > limit:
        print(f"... and {len(conflicts) - limit:,} more conflicts")


def save_conflicts_report(conflicts, lastname_first_dates, df, output_path):
    """Save a detailed report of all conflicts with chronological information"""

    if len(conflicts) == 0:
        print("📄 No conflicts to save.")
        return

    print(f"\n💾 Generating detailed conflicts report with chronological data...")

    # Create a detailed report with chronological information
    detailed_report = []

    for conflict in conflicts.iter_rows(named=True):
        person_name = conflict["personName"]

        # Get chronological data for this personName
        person_filter = (
            pl.col("personName") == person_name
            if person_name
            else pl.col("personName").is_null()
        )
        person_lastname_details = lastname_first_dates.filter(person_filter).sort(
            "first_appearance"
        )

        # Get sample records for additional context
        person_records = (
            df.filter(person_filter)
            .select(
                [
                    "personName",
                    "lastName",
                    "birthDate",
                    "gender",
                    "date",
                    "finalWorth",
                    "countryOfCitizenship",
                ]
            )
            .unique(
                subset=["lastName", "birthDate", "gender"]
            )  # Remove exact duplicates
            .sort(["lastName", "date"])
        )

        for detail, record in zip(
            person_lastname_details.iter_rows(named=True),
            person_records.iter_rows(named=True),
        ):
            detailed_report.append(
                {
                    "personName": person_name,
                    "lastName": detail["lastName"],
                    "first_appearance_date": detail["first_appearance"],
                    "last_appearance_date": detail["last_appearance"],
                    "appearance_count": detail["appearance_count"],
                    "birthDate": record.get("birthDate"),
                    "gender": record.get("gender"),
                    "countryOfCitizenship": record.get("countryOfCitizenship"),
                    "sample_finalWorth": record.get("finalWorth"),
                    "sample_date": record.get("date"),
                    "total_lastnames_for_this_person": conflict["unique_lastnames"],
                }
            )

    # Convert to DataFrame and save
    report_df = pl.DataFrame(detailed_report).sort(
        ["personName", "first_appearance_date"]
    )

    if str(output_path).endswith(".csv"):
        report_df.write_csv(output_path)
    else:
        report_df.write_parquet(output_path)

    print(
        f"✅ Saved detailed chronological report with {len(report_df):,} records to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Find personName values that have multiple lastName values with chronological analysis"
    )
    parser.add_argument(
        "--parquet-dir",
        default="data",
        help="Directory containing parquet files (default: data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of conflicts to display (0 for all, default: 20)",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Don't show example records for each conflict",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        help="Save detailed report to file (CSV or parquet)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all conflicts (equivalent to --limit 0)",
    )

    args = parser.parse_args()

    if args.show_all:
        args.limit = 0

    # Setup paths
    parquet_dir = Path(args.parquet_dir)
    billionaires_path = parquet_dir / "billionaires.parquet"

    print("🔍 PERSONNAME → LASTNAME CONFLICTS CHECKER (with chronological analysis)")
    print("=" * 80)
    print(f"📁 Dataset path: {billionaires_path}")
    print(f"🎯 Looking for personName values with multiple lastName values")
    print(f"📅 Including first/last usage dates for each lastName")

    try:
        # Load the dataset
        df = load_billionaires_data(billionaires_path)

        # Find conflicts
        conflicts, lastname_first_dates = find_personname_conflicts(df)

        # Analyze conflicts
        analyze_conflicts_detail(df, conflicts)

        # Display conflicts
        display_conflicts(
            df,
            conflicts,
            lastname_first_dates,
            limit=args.limit,
            show_examples=not args.no_examples,
        )

        # Generate detailed report if requested
        if args.output_report:
            save_conflicts_report(
                conflicts, lastname_first_dates, df, Path(args.output_report)
            )

        # Summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"✅ Total unique personName values: {df['personName'].n_unique():,}")
        print(f"⚠️  PersonName values with multiple lastName values: {len(conflicts):,}")

        if len(conflicts) > 0:
            percentage = (len(conflicts) / df["personName"].n_unique()) * 100
            print(
                f"📊 Percentage of personName values with conflicts: {percentage:.2f}%"
            )
            print(f"🔍 Chronological analysis helps identify:")
            print(
                f"   • Same-day conflicts: Likely different people with same first name"
            )
            print(f"   • Short gaps (<30 days): Possible data entry errors")
            print(f"   • Long gaps (>1 year): Possible legitimate name changes")
            print(f"   • Sequential usage: May indicate name changes over time")
            print(f"   • Overlapping usage: May indicate different people")
        else:
            print(
                "✅ No conflicts found - all personName values have consistent lastName values!"
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

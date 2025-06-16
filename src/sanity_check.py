#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys


def load_billionaires_data(parquet_path):
    """Load billionaires dataset from parquet file"""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    print(f"📖 Loading billionaires dataset from {parquet_path}")
    df = pl.read_parquet(parquet_path)
    print(f"✅ Loaded {len(df):,} records from {df['date'].n_unique()} unique dates")

    return df


def analyze_person_combinations(df):
    """Analyze unique combinations of personName, lastName, birthDate, gender"""

    print("\n" + "=" * 80)
    print("🔍 ANALYZING PERSON IDENTITY COMBINATIONS")
    print("=" * 80)

    # Get unique combinations of the 4 key fields
    identity_columns = ["personName", "lastName", "birthDate", "gender"]

    unique_combinations = (
        df.select(identity_columns)
        .unique()
        .sort(["personName", "lastName", "birthDate", "gender"])
    )

    print(f"📊 Found {len(unique_combinations):,} unique identity combinations")
    print(f"📊 From {df['personName'].n_unique():,} unique person names in dataset")

    return unique_combinations


def find_inconsistent_identities(df):
    """Find people with inconsistent birthDate or gender across records"""

    print("\n" + "=" * 80)
    print("⚠️  CHECKING FOR IDENTITY INCONSISTENCIES")
    print("=" * 80)

    # Group by personName + lastName and check for multiple birthDates or genders
    person_variations = (
        df.select(["personName", "lastName", "birthDate", "gender"])
        .unique()
        .group_by(["personName", "lastName"])
        .agg(
            [
                pl.col("birthDate").n_unique().alias("unique_birthdates"),
                pl.col("gender").n_unique().alias("unique_genders"),
                pl.col("birthDate").alias("all_birthdates"),
                pl.col("gender").alias("all_genders"),
            ]
        )
        .filter((pl.col("unique_birthdates") > 1) | (pl.col("unique_genders") > 1))
        .sort("personName")
    )

    if len(person_variations) > 0:
        print(
            f"🚨 Found {len(person_variations)} people with inconsistent identity data:"
        )
        print()

        for row in person_variations.iter_rows(named=True):
            print(f"👤 {row['personName']} {row['lastName']}")
            if row["unique_birthdates"] > 1:
                dates = [str(d) if d else "NULL" for d in row["all_birthdates"]]
                print(f"   📅 Multiple birth dates: {', '.join(dates)}")
            if row["unique_genders"] > 1:
                genders = [str(g) if g else "NULL" for g in row["all_genders"]]
                print(f"   ⚧️  Multiple genders: {', '.join(genders)}")
            print()
    else:
        print("✅ No identity inconsistencies found!")

    return person_variations


def analyze_missing_data(df):
    """Analyze missing data patterns in identity fields"""

    print("\n" + "=" * 80)
    print("📋 MISSING DATA ANALYSIS")
    print("=" * 80)

    identity_columns = ["personName", "lastName", "birthDate", "gender"]

    missing_stats = {}

    for col in identity_columns:
        null_count = df.select(pl.col(col).is_null().sum()).item()

        # Only check for empty strings on string/categorical columns
        col_dtype = df.schema[col]
        if col_dtype in [pl.Utf8, pl.Categorical]:
            empty_count = df.select((pl.col(col) == "").sum()).item()
        else:
            # For date and other non-string types, empty strings don't apply
            empty_count = 0

        total_missing = null_count + empty_count
        percentage = (total_missing / len(df)) * 100

        missing_stats[col] = {
            "null": null_count,
            "empty": empty_count,
            "total_missing": total_missing,
            "percentage": percentage,
        }

    print("Missing data summary:")
    print(f"{'Field':<20} {'Null':<8} {'Empty':<8} {'Total':<8} {'%':<8}")
    print("-" * 55)

    for col, stats in missing_stats.items():
        print(
            f"{col:<20} {stats['null']:<8} {stats['empty']:<8} "
            f"{stats['total_missing']:<8} {stats['percentage']:<7.2f}%"
        )

    return missing_stats


def display_unique_combinations(unique_combinations, limit=50):
    """Display the unique combinations in a readable format"""

    print("\n" + "=" * 80)
    print("📋 UNIQUE IDENTITY COMBINATIONS")
    print("=" * 80)

    if limit and len(unique_combinations) > limit:
        print(f"Showing first {limit} of {len(unique_combinations):,} combinations:")
        display_df = unique_combinations.head(limit)
    else:
        print(f"Showing all {len(unique_combinations):,} combinations:")
        display_df = unique_combinations

    print()
    print(f"{'Person Name':<25} {'Last Name':<15} {'Birth Date':<12} {'Gender':<8}")
    print("-" * 65)

    for row in display_df.iter_rows(named=True):
        person_name = str(row["personName"])[:24] if row["personName"] else "NULL"
        last_name = str(row["lastName"])[:14] if row["lastName"] else "NULL"
        birth_date = str(row["birthDate"]) if row["birthDate"] else "NULL"
        gender = str(row["gender"]) if row["gender"] else "NULL"

        print(f"{person_name:<25} {last_name:<15} {birth_date:<12} {gender:<8}")

    if limit and len(unique_combinations) > limit:
        print(f"\n... and {len(unique_combinations) - limit:,} more combinations")


def generate_detailed_report(df, output_file=None):
    """Generate a detailed report of all unique combinations"""

    if not output_file:
        return

    print(f"\n💾 Generating detailed report: {output_file}")

    # Create comprehensive analysis
    unique_combinations = (
        df.select(["personName", "lastName", "birthDate", "gender"])
        .unique()
        .sort(["personName", "lastName", "birthDate", "gender"])
    )

    # Add some analysis columns
    report_df = unique_combinations.with_columns(
        [
            pl.col("birthDate").is_null().alias("missing_birthdate"),
            pl.col("gender").is_null().alias("missing_gender"),
            (pl.col("birthDate").is_null() | pl.col("gender").is_null()).alias(
                "has_missing_data"
            ),
        ]
    )

    # Save to CSV for easy viewing
    if str(output_file).endswith(".csv"):
        report_df.write_csv(output_file)
    else:
        # Default to parquet
        report_df.write_parquet(output_file)

    print(f"✅ Report saved with {len(report_df):,} unique identity combinations")


def main():
    parser = argparse.ArgumentParser(
        description="Check data sanity of billionaires dataset - analyze person identity consistency"
    )
    parser.add_argument(
        "--parquet-dir",
        default="data",
        help="Directory containing parquet files (default: data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit number of combinations to display (0 for all, default: 50)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        help="Save detailed report to file (CSV or parquet)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all combinations (equivalent to --limit 0)",
    )

    args = parser.parse_args()

    if args.show_all:
        args.limit = 0

    # Setup paths
    parquet_dir = Path(args.parquet_dir)
    billionaires_path = parquet_dir / "billionaires.parquet"

    print("🕵️ BILLIONAIRES DATASET SANITY CHECKER")
    print("=" * 80)
    print(f"📁 Dataset path: {billionaires_path}")

    try:
        # Load the dataset
        df = load_billionaires_data(billionaires_path)

        # Analyze unique combinations
        unique_combinations = analyze_person_combinations(df)

        # Check for inconsistencies
        inconsistencies = find_inconsistent_identities(df)

        # Analyze missing data
        missing_stats = analyze_missing_data(df)

        # Display combinations
        display_unique_combinations(unique_combinations, limit=args.limit)

        # Generate detailed report if requested
        if args.output_report:
            generate_detailed_report(df, Path(args.output_report))

        # Summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"✅ Total records in dataset: {len(df):,}")
        print(f"✅ Unique person names: {df['personName'].n_unique():,}")
        print(f"✅ Unique identity combinations: {len(unique_combinations):,}")
        print(f"⚠️  People with inconsistent data: {len(inconsistencies)}")

        # Check if we have more combinations than people (indicates inconsistencies)
        unique_names = df["personName"].n_unique()
        if len(unique_combinations) > unique_names:
            print(
                f"🚨 WARNING: More identity combinations ({len(unique_combinations):,}) than unique names ({unique_names:,})"
            )
            print("   This suggests some people have inconsistent identity data!")
        else:
            print("✅ Identity combinations match unique person names")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

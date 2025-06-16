#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys
from datetime import datetime

# Enable string cache to handle categorical comparisons
pl.enable_string_cache()


def load_billionaires_data(parquet_path):
    """Load billionaires dataset from parquet file"""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    print(f"📖 Loading billionaires dataset from {parquet_path}")
    df = pl.read_parquet(parquet_path)
    print(f"✅ Loaded {len(df):,} records from {df['date'].n_unique()} unique dates")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def find_canonical_identity_values(df, identity_key_columns=None):
    """Find the most recent non-null lastName, birthDate and gender for each person"""

    if identity_key_columns is None:
        identity_key_columns = ["personName"]

    print(
        f"\n🔍 Finding canonical identity values using key: {', '.join(identity_key_columns)}"
    )

    # Identity fields to consolidate (now includes lastName)
    identity_fields_to_fix = ["lastName", "birthDate", "gender"]

    # Clean up empty strings by converting them to nulls for identity fields
    print("🧹 Converting empty strings to nulls for identity fields...")
    df_clean = df.with_columns(
        [
            pl.when(pl.col("lastName") == "")
            .then(None)
            .otherwise(pl.col("lastName"))
            .alias("lastName"),
            pl.when(pl.col("gender") == "")
            .then(None)
            .otherwise(pl.col("gender"))
            .alias("gender"),
            # Note: birthDate is already Date type, so no empty strings possible
        ]
    )

    # Get unique identities
    unique_identities = df_clean.select(identity_key_columns).unique()
    print(f"👥 Found {len(unique_identities):,} unique identities")

    canonical_values = []

    for i, identity_row in enumerate(unique_identities.iter_rows(named=True)):
        if (i + 1) % 100 == 0:
            print(f"   Processing identity {i + 1:,}/{len(unique_identities):,}...")

        # Filter data for this specific identity - handle null values properly
        identity_filter = None
        for key_col in identity_key_columns:
            key_value = identity_row[key_col]

            if key_value is None:
                condition = pl.col(key_col).is_null()
            else:
                condition = pl.col(key_col) == key_value

            if identity_filter is None:
                identity_filter = condition
            else:
                identity_filter = identity_filter & condition

        person_data = df_clean.filter(identity_filter).sort("date", descending=True)

        if len(person_data) == 0:
            continue

        # Start with the identity key values
        canonical_record = {col: identity_row[col] for col in identity_key_columns}

        # For each identity field, get the most recent non-null value
        for field in identity_fields_to_fix:
            non_null_values = person_data.filter(pl.col(field).is_not_null())
            if len(non_null_values) > 0:
                canonical_record[field] = non_null_values[field][0]
            else:
                canonical_record[field] = None

        canonical_values.append(canonical_record)

    # Convert to DataFrame with matching schema
    canonical_df = pl.DataFrame(canonical_values)

    # Ensure the canonical_df has the same data types as the original df for join columns
    for col in identity_key_columns + identity_fields_to_fix:
        if col in df.columns and col in canonical_df.columns:
            original_dtype = df.schema[col]
            canonical_df = canonical_df.with_columns(pl.col(col).cast(original_dtype))

    print(f"✅ Found canonical values for {len(canonical_df):,} unique identities")

    return canonical_df


def apply_canonical_identity_fixes(df, canonical_df, identity_key_columns):
    """Apply the canonical identity values to all records"""

    print(f"\n🔧 Applying canonical identity fixes to all {len(df):,} records...")

    # First, clean up empty strings in the original data
    print("🧹 Converting empty strings to nulls in original data...")
    df_clean = df.with_columns(
        [
            pl.when(pl.col("lastName") == "")
            .then(None)
            .otherwise(pl.col("lastName"))
            .alias("lastName"),
            pl.when(pl.col("gender") == "")
            .then(None)
            .otherwise(pl.col("gender"))
            .alias("gender"),
        ]
    )

    # Join the canonical values back to the cleaned data
    fixed_df = df_clean.join(
        canonical_df.select(
            identity_key_columns + ["lastName", "birthDate", "gender"]
        ).rename(
            {
                "lastName": "canonical_lastName",
                "birthDate": "canonical_birthDate",
                "gender": "canonical_gender",
            }
        ),
        on=identity_key_columns,
        how="left",
    )

    # Replace the original lastName, birthDate and gender with canonical values
    fixed_df = fixed_df.with_columns(
        [
            pl.col("canonical_lastName").alias("lastName"),
            pl.col("canonical_birthDate").alias("birthDate"),
            pl.col("canonical_gender").alias("gender"),
        ]
    ).drop(["canonical_lastName", "canonical_birthDate", "canonical_gender"])

    print(f"✅ Applied canonical identity fixes to all records")
    print(f"✅ Converted empty strings to nulls for cleaner data representation")

    return fixed_df


def analyze_identity_fixes(original_df, fixed_df, identity_key_columns):
    """Analyze what identity inconsistencies were fixed"""

    print("\n" + "=" * 80)
    print("📊 IDENTITY FIXES ANALYSIS")
    print("=" * 80)

    # Clean the original data for comparison (convert empty strings to nulls)
    original_df_clean = original_df.with_columns(
        [
            pl.when(pl.col("lastName") == "")
            .then(None)
            .otherwise(pl.col("lastName"))
            .alias("lastName"),
            pl.when(pl.col("gender") == "")
            .then(None)
            .otherwise(pl.col("gender"))
            .alias("gender"),
        ]
    )

    # Check for people who had inconsistent values before
    lastname_inconsistencies_before = (
        original_df_clean.select(identity_key_columns + ["lastName"])
        .unique()
        .group_by(identity_key_columns)
        .agg(pl.col("lastName").n_unique().alias("unique_lastnames"))
        .filter(pl.col("unique_lastnames") > 1)
    )

    birthdate_inconsistencies_before = (
        original_df_clean.select(identity_key_columns + ["birthDate"])
        .unique()
        .group_by(identity_key_columns)
        .agg(pl.col("birthDate").n_unique().alias("unique_birthdates"))
        .filter(pl.col("unique_birthdates") > 1)
    )

    gender_inconsistencies_before = (
        original_df_clean.select(identity_key_columns + ["gender"])
        .unique()
        .group_by(identity_key_columns)
        .agg(pl.col("gender").n_unique().alias("unique_genders"))
        .filter(pl.col("unique_genders") > 1)
    )

    # Check after fixes (should be 0)
    lastname_inconsistencies_after = (
        fixed_df.select(identity_key_columns + ["lastName"])
        .unique()
        .group_by(identity_key_columns)
        .agg(pl.col("lastName").n_unique().alias("unique_lastnames"))
        .filter(pl.col("unique_lastnames") > 1)
    )

    birthdate_inconsistencies_after = (
        fixed_df.select(identity_key_columns + ["birthDate"])
        .unique()
        .group_by(identity_key_columns)
        .agg(pl.col("birthDate").n_unique().alias("unique_birthdates"))
        .filter(pl.col("unique_birthdates") > 1)
    )

    gender_inconsistencies_after = (
        fixed_df.select(identity_key_columns + ["gender"])
        .unique()
        .group_by(identity_key_columns)
        .agg(pl.col("gender").n_unique().alias("unique_genders"))
        .filter(pl.col("unique_genders") > 1)
    )

    print(f"🔍 Last Name Inconsistencies:")
    print(
        f"   Before: {len(lastname_inconsistencies_before):,} people with inconsistent last names"
    )
    print(
        f"   After:  {len(lastname_inconsistencies_after):,} people with inconsistent last names"
    )

    print(f"\n🔍 Birth Date Inconsistencies:")
    print(
        f"   Before: {len(birthdate_inconsistencies_before):,} people with inconsistent birth dates"
    )
    print(
        f"   After:  {len(birthdate_inconsistencies_after):,} people with inconsistent birth dates"
    )

    print(f"\n🔍 Gender Inconsistencies:")
    print(
        f"   Before: {len(gender_inconsistencies_before):,} people with inconsistent genders"
    )
    print(
        f"   After:  {len(gender_inconsistencies_after):,} people with inconsistent genders"
    )

    # Analyze data completeness (using cleaned original data for fair comparison)
    print(f"\n📋 Data Completeness (after converting empty strings to nulls):")
    for field in ["lastName", "birthDate", "gender"]:
        original_completeness = (
            original_df_clean.select(pl.col(field).is_not_null().sum()).item()
            / len(original_df_clean)
        ) * 100
        fixed_completeness = (
            fixed_df.select(pl.col(field).is_not_null().sum()).item() / len(fixed_df)
        ) * 100
        improvement = fixed_completeness - original_completeness

        print(
            f"   {field:<12}: {original_completeness:6.1f}% → {fixed_completeness:6.1f}% ({improvement:+5.1f}%)"
        )

    return {
        "lastname_fixes": len(lastname_inconsistencies_before),
        "birthdate_fixes": len(birthdate_inconsistencies_before),
        "gender_fixes": len(gender_inconsistencies_before),
    }


def show_example_fixes(original_df, fixed_df, identity_key_columns, num_examples=3):
    """Show specific examples of people whose identity data was fixed"""

    print(f"\n" + "=" * 80)
    print(f"🔍 EXAMPLE IDENTITY FIXES (showing up to {num_examples} examples)")
    print("=" * 80)

    # Clean the original data first (convert empty strings to nulls)
    original_df_clean = original_df.with_columns(
        [
            pl.when(pl.col("lastName") == "")
            .then(None)
            .otherwise(pl.col("lastName"))
            .alias("lastName"),
            pl.when(pl.col("gender") == "")
            .then(None)
            .otherwise(pl.col("gender"))
            .alias("gender"),
        ]
    )

    # Find people who had inconsistent lastName, birthDates or genders in original data
    inconsistent_people = (
        original_df_clean.select(
            identity_key_columns + ["lastName", "birthDate", "gender"]
        )
        .unique()
        .group_by(identity_key_columns)
        .agg(
            [
                pl.col("lastName").n_unique().alias("unique_lastnames"),
                pl.col("birthDate").n_unique().alias("unique_birthdates"),
                pl.col("gender").n_unique().alias("unique_genders"),
                pl.col("lastName").alias("all_lastnames"),
                pl.col("birthDate").alias("all_birthdates"),
                pl.col("gender").alias("all_genders"),
            ]
        )
        .filter(
            (pl.col("unique_lastnames") > 1)
            | (pl.col("unique_birthdates") > 1)
            | (pl.col("unique_genders") > 1)
        )
        .head(num_examples)
    )

    if len(inconsistent_people) == 0:
        print("✅ No identity inconsistencies found to show examples of!")
        return

    for person in inconsistent_people.iter_rows(named=True):
        print(f"\n👤 {person['personName']}")

        if person["unique_lastnames"] > 1:
            names = [str(n) if n else "NULL" for n in person["all_lastnames"]]
            print(f"   📝 Had multiple last names: {', '.join(set(names))}")

        if person["unique_birthdates"] > 1:
            dates = [str(d) if d else "NULL" for d in person["all_birthdates"]]
            print(f"   📅 Had multiple birth dates: {', '.join(set(dates))}")

        if person["unique_genders"] > 1:
            genders = [str(g) if g else "NULL" for g in person["all_genders"]]
            print(f"   ⚧️  Had multiple genders: {', '.join(set(genders))}")

        # Show the fixed values
        identity_filter = None
        for key_col in identity_key_columns:
            key_value = person[key_col]
            if key_value is None:
                condition = pl.col(key_col).is_null()
            else:
                condition = pl.col(key_col) == key_value

            if identity_filter is None:
                identity_filter = condition
            else:
                identity_filter = identity_filter & condition

        fixed_person = (
            fixed_df.filter(identity_filter)
            .select(["lastName", "birthDate", "gender"])
            .unique()
        )
        if len(fixed_person) > 0:
            fixed_row = fixed_person.row(0, named=True)
            print(
                f"   ✅ Fixed to: lastName={fixed_row['lastName']}, birthDate={fixed_row['birthDate']}, gender={fixed_row['gender']}"
            )
            print(
                f"   🧹 Note: Empty strings have been converted to nulls for cleaner data"
            )


def save_fixed_data(fixed_df, output_path, format_type="parquet"):
    """Save the dataset with fixed identity data"""

    print(f"\n💾 Saving fixed dataset to {output_path}")

    if format_type.lower() == "csv":
        fixed_df.write_csv(output_path)
    elif format_type.lower() == "parquet":
        fixed_df.write_parquet(output_path, compression="brotli", compression_level=11)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    print(f"✅ Saved {len(fixed_df):,} records with fixed identity data")


def main():
    parser = argparse.ArgumentParser(
        description="Fix billionaire identity inconsistencies (lastName, birthDate and gender) while preserving all time-series data"
    )
    parser.add_argument(
        "--parquet-dir",
        default="data",
        help="Directory containing parquet files (default: data)",
    )
    parser.add_argument(
        "--output",
        default="billionaires_fixed_identities",
        help="Output filename (without extension, default: billionaires_fixed_identities)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--identity-keys",
        nargs="+",
        default=["personName"],
        help="Columns to use as identity keys (default: personName)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of fix examples to show (default: 5)",
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

    print("🔧 BILLIONAIRE IDENTITY REPAIR (lastName, birthDate & gender)")
    print("=" * 80)
    print(f"📁 Input dataset: {billionaires_path}")
    print(f"🔑 Identity keys: {', '.join(args.identity_keys)}")
    print(f"🛠️  Fields to fix: lastName, birthDate, gender")
    print(f"💾 Output file: {output_path}")
    print(f"🔒 Dry run: {args.dry_run}")
    print(
        f"⚠️  All other fields (wealth, location, etc.) will be preserved as time-series data"
    )

    try:
        # Load the dataset
        df = load_billionaires_data(billionaires_path)

        # Find canonical identity values
        canonical_df = find_canonical_identity_values(df, args.identity_keys)

        # Apply the fixes to all records
        fixed_df = apply_canonical_identity_fixes(df, canonical_df, args.identity_keys)

        # Analyze what was fixed
        fix_stats = analyze_identity_fixes(df, fixed_df, args.identity_keys)

        # Show examples of fixes
        show_example_fixes(df, fixed_df, args.identity_keys, args.examples)

        # Save results (unless dry run)
        if not args.dry_run:
            save_fixed_data(fixed_df, output_path, args.format)
        else:
            print(
                f"\n🔍 DRY RUN - Would save {len(fixed_df):,} records with fixed identity data to {output_path}"
            )

        # Final summary
        print("\n" + "=" * 80)
        print("✅ IDENTITY REPAIR COMPLETED")
        print("=" * 80)
        print(f"📊 Total records: {len(fixed_df):,} (unchanged)")
        print(f"🔧 People with fixed last names: {fix_stats['lastname_fixes']:,}")
        print(f"🔧 People with fixed birth dates: {fix_stats['birthdate_fixes']:,}")
        print(f"🔧 People with fixed genders: {fix_stats['gender_fixes']:,}")
        print(f"✅ All time-series data preserved")
        print(f"✅ Only lastName, birthDate and gender inconsistencies were fixed")

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

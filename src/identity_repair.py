#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys
from data_lib import load_data, save_data

pl.enable_string_cache()


def find_canonical_values(df, id_keys=None):
    """Find most recent non-null identity values for each person"""
    if id_keys is None:
        id_keys = ["personName"]

    print(f"\n🔍 Finding canonical values using: {', '.join(id_keys)}")

    # Fields to fix
    fix_fields = ["lastName", "birthDate", "gender"]

    # Clean empty strings
    print("🧹 Converting empty strings to nulls...")
    df_clean = df.with_columns(
        [
            pl.when(pl.col("lastName") == "").then(None).otherwise(pl.col("lastName")),
            pl.when(pl.col("gender") == "").then(None).otherwise(pl.col("gender")),
        ]
    )

    # Get unique identities
    unique_ids = df_clean.select(id_keys).unique()
    print(f"👥 Found {len(unique_ids):,} unique identities")

    canonical = []

    for i, id_row in enumerate(unique_ids.iter_rows(named=True)):
        if (i + 1) % 100 == 0:
            print(f"   Processing {i + 1:,}/{len(unique_ids):,}...")

        # Build filter
        filter_expr = None
        for key in id_keys:
            val = id_row[key]
            cond = pl.col(key).is_null() if val is None else pl.col(key) == val
            filter_expr = cond if filter_expr is None else filter_expr & cond

        person_data = df_clean.filter(filter_expr).sort("date", descending=True)
        if len(person_data) == 0:
            continue

        # Build canonical record
        rec = {col: id_row[col] for col in id_keys}

        # Get most recent non-null value for each field
        for field in fix_fields:
            non_null = person_data.filter(pl.col(field).is_not_null())
            rec[field] = non_null[field][0] if len(non_null) > 0 else None

        canonical.append(rec)

    # Convert to dataframe with matching types
    canonical_df = pl.DataFrame(canonical)
    for col in id_keys + fix_fields:
        if col in df.columns and col in canonical_df.columns:
            canonical_df = canonical_df.with_columns(pl.col(col).cast(df.schema[col]))

    print(f"✅ Found canonical values for {len(canonical_df):,} identities")
    return canonical_df


def apply_fixes(df, canonical_df, id_keys):
    """Apply canonical values to all records"""
    print(f"\n🔧 Applying fixes to {len(df):,} records...")

    # Clean empty strings
    df_clean = df.with_columns(
        [
            pl.when(pl.col("lastName") == "").then(None).otherwise(pl.col("lastName")),
            pl.when(pl.col("gender") == "").then(None).otherwise(pl.col("gender")),
        ]
    )

    # Join with canonical values
    fixed = df_clean.join(
        canonical_df.select(id_keys + ["lastName", "birthDate", "gender"]).rename(
            {
                "lastName": "new_lastName",
                "birthDate": "new_birthDate",
                "gender": "new_gender",
            }
        ),
        on=id_keys,
        how="left",
    )

    # Replace with canonical values
    fixed = fixed.with_columns(
        [
            pl.col("new_lastName").alias("lastName"),
            pl.col("new_birthDate").alias("birthDate"),
            pl.col("new_gender").alias("gender"),
        ]
    ).drop(["new_lastName", "new_birthDate", "new_gender"])

    print("✅ Applied fixes")
    return fixed


def analyze_fixes(original, fixed, id_keys):
    """Analyze what was fixed"""
    print("\n" + "=" * 80)
    print("📊 IDENTITY FIXES ANALYSIS")
    print("=" * 80)

    # Clean original for comparison
    orig_clean = original.with_columns(
        [
            pl.when(pl.col("lastName") == "").then(None).otherwise(pl.col("lastName")),
            pl.when(pl.col("gender") == "").then(None).otherwise(pl.col("gender")),
        ]
    )

    stats = {}
    for field in ["lastName", "birthDate", "gender"]:
        # Count inconsistencies before
        before = (
            orig_clean.select(id_keys + [field])
            .unique()
            .group_by(id_keys)
            .agg(pl.col(field).n_unique().alias("n"))
            .filter(pl.col("n") > 1)
        )

        # Count inconsistencies after
        after = (
            fixed.select(id_keys + [field])
            .unique()
            .group_by(id_keys)
            .agg(pl.col(field).n_unique().alias("n"))
            .filter(pl.col("n") > 1)
        )

        stats[field] = len(before)
        print(f"🔍 {field}: {len(before):,} → {len(after):,} inconsistencies")

    # Completeness
    print(f"\n📋 Data Completeness:")
    for field in ["lastName", "birthDate", "gender"]:
        orig_pct = (orig_clean[field].is_not_null().sum() / len(orig_clean)) * 100
        fixed_pct = (fixed[field].is_not_null().sum() / len(fixed)) * 100
        print(
            f"   {field:<12}: {orig_pct:6.1f}% → {fixed_pct:6.1f}% ({fixed_pct-orig_pct:+5.1f}%)"
        )

    return stats


def show_examples(original, fixed, id_keys, n=3):
    """Show example fixes"""
    print(f"\n🔍 EXAMPLE FIXES (up to {n})")
    print("=" * 80)

    # Clean original
    orig_clean = original.with_columns(
        [
            pl.when(pl.col("lastName") == "").then(None).otherwise(pl.col("lastName")),
            pl.when(pl.col("gender") == "").then(None).otherwise(pl.col("gender")),
        ]
    )

    # Find people with inconsistencies
    inconsistent = (
        orig_clean.select(id_keys + ["lastName", "birthDate", "gender"])
        .unique()
        .group_by(id_keys)
        .agg(
            [
                pl.col("lastName").n_unique().alias("n_ln"),
                pl.col("birthDate").n_unique().alias("n_bd"),
                pl.col("gender").n_unique().alias("n_g"),
                pl.col("lastName").alias("all_ln"),
                pl.col("birthDate").alias("all_bd"),
                pl.col("gender").alias("all_g"),
            ]
        )
        .filter((pl.col("n_ln") > 1) | (pl.col("n_bd") > 1) | (pl.col("n_g") > 1))
        .head(n)
    )

    if len(inconsistent) == 0:
        print("✅ No inconsistencies found!")
        return

    for person in inconsistent.iter_rows(named=True):
        print(f"\n👤 {person['personName']}")

        if person["n_ln"] > 1:
            vals = [str(v) if v else "NULL" for v in person["all_ln"]]
            print(f"   📝 Last names: {', '.join(set(vals))}")

        if person["n_bd"] > 1:
            vals = [str(v) if v else "NULL" for v in person["all_bd"]]
            print(f"   📅 Birth dates: {', '.join(set(vals))}")

        if person["n_g"] > 1:
            vals = [str(v) if v else "NULL" for v in person["all_g"]]
            print(f"   ⚧️  Genders: {', '.join(set(vals))}")

        # Show fixed values
        filter_expr = None
        for key in id_keys:
            val = person[key]
            cond = pl.col(key).is_null() if val is None else pl.col(key) == val
            filter_expr = cond if filter_expr is None else filter_expr & cond

        fixed_person = (
            fixed.filter(filter_expr)
            .select(["lastName", "birthDate", "gender"])
            .unique()
        )
        if len(fixed_person) > 0:
            row = fixed_person.row(0, named=True)
            print(
                f"   ✅ Fixed to: lastName={row['lastName']}, birthDate={row['birthDate']}, gender={row['gender']}"
            )


def main():
    parser = argparse.ArgumentParser(description="Fix identity inconsistencies")
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument("--output", default="billionaires_fixed_identities")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--identity-keys", nargs="+", default=["personName"])
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.parquet_dir) / "billionaires.parquet"
    output_path = Path(args.parquet_dir) / f"{args.output}.{args.format}"

    print("🔧 BILLIONAIRE IDENTITY REPAIR")
    print("=" * 80)
    print(f"📁 Input: {input_path}")
    print(f"🔑 Keys: {', '.join(args.identity_keys)}")
    print(f"🛠️  Fields: lastName, birthDate, gender")
    print(f"💾 Output: {output_path}")
    print(f"🔒 Dry run: {args.dry_run}")

    try:
        # Load data
        df = load_data(input_path, "billionaires")

        # Find canonical values
        canonical = find_canonical_values(df, args.identity_keys)

        # Apply fixes
        fixed = apply_fixes(df, canonical, args.identity_keys)

        # Analyze
        stats = analyze_fixes(df, fixed, args.identity_keys)

        # Show examples
        show_examples(df, fixed, args.identity_keys, args.examples)

        # Save
        if not args.dry_run:
            if args.format == "parquet":
                save_data(fixed, output_path, "billionaires")
            else:
                fixed.write_csv(output_path)
                print(f"✅ Saved {len(fixed):,} records to {output_path}")
        else:
            print(f"\n🔍 DRY RUN - Would save {len(fixed):,} records")

        # Summary
        print("\n" + "=" * 80)
        print("✅ IDENTITY REPAIR COMPLETED")
        print("=" * 80)
        print(f"📊 Total records: {len(fixed):,}")
        print(f"🔧 Fixed last names: {stats['lastName']:,}")
        print(f"🔧 Fixed birth dates: {stats['birthDate']:,}")
        print(f"🔧 Fixed genders: {stats['gender']:,}")

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

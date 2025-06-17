#!/usr/bin/env python3
import polars as pl
import argparse
from pathlib import Path
import sys
from data_lib import load_data, save_data
from repairs_lib import repair_second_order_fields, clean_second_order_empty_strings

pl.enable_string_cache()


def analyze_results(original, repaired, fields):
    """Analyze repair impact"""
    print(f"\n📊 REPAIR IMPACT ANALYSIS")
    print("=" * 60)

    total_before = total_after = 0

    print(f"{'Field':<25} {'Before':<10} {'After':<10} {'Filled':<10} {'%':<10}")
    print("-" * 70)

    for field in fields:
        before = original[field].is_null().sum()
        after = repaired[field].is_null().sum()
        filled = before - after
        pct = (filled / before * 100) if before > 0 else 0

        print(f"{field:<25} {before:<10,} {after:<10,} {filled:<10,} {pct:<9.1f}%")
        total_before += before
        total_after += after

    total_filled = total_before - total_after
    overall_pct = (total_filled / total_before * 100) if total_before > 0 else 0

    print("-" * 70)
    print(
        f"{'TOTAL':<25} {total_before:<10,} {total_after:<10,} {total_filled:<10,} {overall_pct:<9.1f}%"
    )

    return {"total_filled": total_filled, "percentage": overall_pct}


def show_examples(original, repaired, fields, n=2):
    """Show repair examples"""
    print(f"\n🔍 REPAIR EXAMPLES")
    print("=" * 60)

    # Sample people
    sample = (
        original.select("personName")
        .filter(pl.col("personName").is_not_null())
        .unique()
        .head(200)
        .to_series()
        .to_list()
    )

    shown = 0
    max_total = 5

    for field in fields:
        if shown >= max_total:
            break

        print(f"\n📝 Examples for {field}:")
        field_examples = 0

        for person in sample:
            if field_examples >= n or shown >= max_total:
                break

            if person is None:
                continue

            # Get person data
            orig = (
                original.filter(pl.col("personName") == person)
                .select(["date", field])
                .sort("date")
            )
            rep = (
                repaired.filter(pl.col("personName") == person)
                .select(["date", field])
                .sort("date")
            )

            # Check for repairs
            had_repair = False
            changes = []

            for o, r in zip(orig.iter_rows(named=True), rep.iter_rows(named=True)):
                if o[field] is None and r[field] is not None:
                    had_repair = True
                    changes.append((o["date"], "NULL", r[field]))
                    if len(changes) >= 3:
                        break

            if had_repair:
                print(f"\n   👤 {person}:")
                for date, before, after in changes:
                    print(f"      📅 {date}: {before} → {after} ✅")
                field_examples += 1
                shown += 1

        if field_examples == 0:
            print("   ℹ️  No examples in sample")

    print(f"\n💡 Checked sample of {len(sample)} people")


def main():
    parser = argparse.ArgumentParser(
        description="Second order repair with forward/backward fill using repairs library"
    )
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument("--output", default="billionaires_second_order_repaired")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.parquet_dir) / "billionaires.parquet"
    output_path = Path(args.parquet_dir) / f"{args.output}.{args.format}"

    print("🔧 SECOND ORDER FIELDS REPAIR (Using Repairs Library)")
    print("=" * 80)
    print(f"📁 Input: {input_path}")
    print(f"🛠️  Fields: countryOfCitizenship, city, state, source, industries")
    print(f"💾 Output: {output_path}")
    print(f"🔒 Dry run: {args.dry_run}")
    print(f"🪟 Method: Forward → backward fill")

    try:
        # Load data
        df = load_data(input_path, "billionaires")

        # Get fields that will be repaired (for analysis)
        df_clean, fields = clean_second_order_empty_strings(df)
        if not fields:
            print("❌ No fields to repair")
            return False

        # Apply repair using library function
        repaired = repair_second_order_fields(df)

        # Analyze
        stats = analyze_results(df_clean, repaired, fields)

        # Show examples
        show_examples(df_clean, repaired, fields, args.examples)

        # Save
        if not args.dry_run:
            if args.format == "parquet":
                save_data(repaired, output_path, "billionaires")
            else:
                repaired.write_csv(output_path)
                print(f"✅ Saved {len(repaired):,} records")
        else:
            print(f"\n🔍 DRY RUN - Would save {len(repaired):,} records")

        # Summary
        print("\n" + "=" * 80)
        print("✅ SECOND ORDER REPAIR COMPLETED")
        print("=" * 80)
        print(f"📊 Total records: {len(repaired):,}")
        print(f"🔧 Fields repaired: {len(fields)}")
        print(f"📈 Nulls filled: {stats['total_filled']:,}")
        print(f"📊 Improvement: {stats['percentage']:.1f}%")

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

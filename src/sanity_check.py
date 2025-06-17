#!/usr/bin/env python3
"""Comprehensive dataset analysis script - checks for all order issues and duplicates"""

import polars as pl
import argparse
from pathlib import Path
import sys
from datetime import datetime
import json
from data_lib import load_data
from repairs_lib import (
    count_0th_order_issues,
    analyze_duplicates,
    clean_whitespace_and_unknowns,
    repair_identity_consistency,
    repair_second_order_fields,
    repair_deduplication,
)

pl.enable_string_cache()


def analyze_0th_order_issues(df, dataset_type):
    """Analyze 0th order issues (whitespace, unknowns)"""
    print(f"\n🧹 0TH ORDER ANALYSIS - {dataset_type.upper()}")
    print("=" * 60)
    
    issues = count_0th_order_issues(df)
    
    print(f"Whitespace issues: {issues['whitespace']:,}")
    print(f"Unknown variations: {issues['unknown']:,}")
    
    # Show examples of problematic values
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in (pl.Utf8, pl.Categorical)
    ]
    
    examples = {}
    for col in string_cols[:5]:  # Check first 5 string columns
        col_str = pl.col(col).cast(pl.Utf8)
        
        # Find whitespace issues
        ws_examples = df.filter(
            col_str.is_not_null() & (col_str != col_str.str.strip_chars())
        ).select(col).unique().limit(3)
        
        if len(ws_examples) > 0:
            examples[f"{col}_whitespace"] = ws_examples[col].to_list()
        
        # Find unknown variations
        unk_examples = df.filter(
            col_str.is_not_null() & col_str.str.contains(r"(?i)^(unknown|unknown_-?\d+)$")
        ).select(col).unique().limit(3)
        
        if len(unk_examples) > 0:
            examples[f"{col}_unknown"] = unk_examples[col].to_list()
    
    if examples:
        print("\nExample problematic values:")
        for key, values in examples.items():
            print(f"  {key}: {values}")
    
    return issues


def analyze_1st_order_issues(df, dataset_type):
    """Analyze 1st order issues (identity inconsistencies)"""
    print(f"\n🔍 1ST ORDER ANALYSIS - {dataset_type.upper()}")
    print("=" * 60)
    
    if dataset_type != "billionaires":
        print("⚠️ 1st order analysis only applies to billionaires dataset")
        return {}
    
    # Check for identity inconsistencies
    identity_fields = ["lastName", "birthDate", "gender"]
    inconsistencies = {}
    
    for field in identity_fields:
        if field not in df.columns:
            inconsistencies[field] = 0
            continue
            
        # Clean empty strings first (only for string columns)
        field_dtype = df.schema[field]
        if field_dtype in [pl.Utf8, pl.Categorical]:
            df_clean = df.with_columns(
                pl.when(pl.col(field) == "").then(None).otherwise(pl.col(field)).alias(field)
            )
        else:
            # For non-string columns (like Date), use as-is
            df_clean = df
        
        # Find people with multiple values for this field
        conflicts = (
            df_clean.group_by("personName")
            .agg(pl.col(field).n_unique().alias("unique_count"))
            .filter(pl.col("unique_count") > 1)
        )
        
        inconsistencies[field] = len(conflicts)
        print(f"{field} inconsistencies: {len(conflicts):,} people")
        
        # Show examples with detailed conflict information
        if len(conflicts) > 0:
            print(f"  Examples (showing up to 5):")
            
            # Get the top conflicting person names with their unique counts
            top_conflicts = conflicts.head(5)
            
            for row in top_conflicts.iter_rows(named=True):
                name = row["personName"]
                unique_count = row["unique_count"]
                
                # Get all records for this person to see the actual values
                person_data = df_clean.filter(pl.col("personName") == name).select([field, "date"])
                
                # Get unique values with their first occurrence date
                unique_values = (
                    person_data
                    .group_by(field)
                    .agg([
                        pl.count().alias("count"),
                        pl.col("date").min().alias("first_seen")
                    ])
                    .sort("first_seen")
                )
                
                print(f"    {name} (has {unique_count} different values):")
                for val_row in unique_values.iter_rows(named=True):
                    val = val_row[field]
                    count = val_row["count"]
                    first_seen = val_row["first_seen"]
                    
                    if val is None:
                        val_str = "NULL"
                    else:
                        val_str = f"'{val}'"
                    
                    print(f"      {val_str}: {count} records (first seen: {first_seen})")
    
    return inconsistencies


def analyze_2nd_order_issues(df, dataset_type):
    """Analyze 2nd order issues (forward/backward fillable data)"""
    print(f"\n🔧 2ND ORDER ANALYSIS - {dataset_type.upper()}")
    print("=" * 60)
    
    if dataset_type != "billionaires":
        print("⚠️ 2nd order analysis only applies to billionaires dataset")
        return {}
    
    # Fields that can be forward/backward filled
    fill_fields = ["countryOfCitizenship", "city", "state", "source", "industries"]
    existing_fields = [f for f in fill_fields if f in df.columns]
    
    fillable_stats = {}
    
    for field in existing_fields:
        # Clean empty strings (only for string fields)
        field_dtype = df.schema[field]
        if field_dtype in [pl.Utf8, pl.Categorical]:
            df_clean = df.with_columns(
                pl.when(pl.col(field) == "").then(None).otherwise(pl.col(field)).alias(field)
            )
        else:
            df_clean = df
        
        # Count nulls per person
        person_stats = (
            df_clean.group_by("personName")
            .agg([
                pl.col(field).count().alias("total_records"),
                pl.col(field).null_count().alias("null_records"),
            ])
            .with_columns(
                (pl.col("null_records") / pl.col("total_records") * 100).alias("null_percentage")
            )
        )
        
        # People who have some data but missing some
        partially_fillable = person_stats.filter(
            (pl.col("null_records") > 0) & (pl.col("null_records") < pl.col("total_records"))
        )
        
        # People who have no data at all
        completely_missing = person_stats.filter(
            pl.col("null_records") == pl.col("total_records")
        )
        
        fillable_stats[field] = {
            "total_nulls": df_clean[field].null_count(),
            "partially_fillable_people": len(partially_fillable),
            "completely_missing_people": len(completely_missing),
            "total_people": len(person_stats),
        }
        
        print(f"{field}:")
        print(f"  Total null values: {fillable_stats[field]['total_nulls']:,}")
        print(f"  People with partial data: {fillable_stats[field]['partially_fillable_people']:,}")
        print(f"  People with no data: {fillable_stats[field]['completely_missing_people']:,}")
    
    return fillable_stats


def simulate_repair_impact(df, dataset_type):
    """Simulate what repairs would accomplish"""
    print(f"\n🔮 REPAIR IMPACT SIMULATION - {dataset_type.upper()}")
    print("=" * 60)
    
    original_count = len(df)
    
    # Simulate 0th order
    cleaned = clean_whitespace_and_unknowns(df, None)
    after_0th = len(cleaned)
    
    # Simulate 1st and 2nd order for billionaires
    if dataset_type == "billionaires":
        identity_repaired = repair_identity_consistency(cleaned)
        after_1st = len(identity_repaired)
        
        fill_repaired = repair_second_order_fields(identity_repaired)
        after_2nd = len(fill_repaired)
    else:
        after_1st = after_0th
        after_2nd = after_0th
        fill_repaired = cleaned
    
    # Simulate deduplication
    deduplicated = repair_deduplication(fill_repaired, dataset_type)
    after_3rd = len(deduplicated)
    
    print(f"Original records: {original_count:,}")
    print(f"After 0th order (clean): {after_0th:,} (removed: {original_count - after_0th:,})")
    print(f"After 1st order (identity): {after_1st:,} (removed: {after_0th - after_1st:,})")
    print(f"After 2nd order (fill): {after_2nd:,} (removed: {after_1st - after_2nd:,})")
    print(f"After 3rd order (dedup): {after_3rd:,} (removed: {after_2nd - after_3rd:,})")
    print(f"Total reduction: {original_count - after_3rd:,} records ({(original_count - after_3rd) / original_count * 100:.1f}%)")
    
    return {
        "original": original_count,
        "after_0th": after_0th,
        "after_1st": after_1st,
        "after_2nd": after_2nd,
        "after_3rd": after_3rd,
        "total_reduction": original_count - after_3rd,
    }


def generate_summary_report(results, dataset_type):
    """Generate summary report of all findings"""
    print(f"\n📊 COMPREHENSIVE SUMMARY - {dataset_type.upper()}")
    print("=" * 60)
    
    # 0th order summary
    print("🧹 0th Order Issues:")
    print(f"  Whitespace: {results['0th_order']['whitespace']:,}")
    print(f"  Unknown variations: {results['0th_order']['unknown']:,}")
    
    # 1st order summary
    if dataset_type == "billionaires" and "1st_order" in results:
        print("\n🔍 1st Order Issues:")
        for field, count in results["1st_order"].items():
            print(f"  {field} inconsistencies: {count:,}")
    
    # 2nd order summary
    if dataset_type == "billionaires" and "2nd_order" in results:
        print("\n🔧 2nd Order Issues:")
        for field, stats in results["2nd_order"].items():
            print(f"  {field}: {stats['total_nulls']:,} nulls, {stats['partially_fillable_people']:,} partially fillable people")
    
    # Deduplication summary
    print("\n🔄 Deduplication Issues:")
    print(f"  Duplicate groups: {results['duplicates']['duplicate_groups']:,}")
    print(f"  Total duplicates: {results['duplicates']['total_duplicates']:,}")
    
    # Repair impact
    print("\n🔮 Repair Impact:")
    impact = results["repair_impact"]
    print(f"  Records would be reduced from {impact['original']:,} to {impact['after_3rd']:,}")
    print(f"  Total reduction: {impact['total_reduction']:,} ({impact['total_reduction'] / impact['original'] * 100:.1f}%)")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    total_issues = (
        results['0th_order']['whitespace'] + 
        results['0th_order']['unknown'] + 
        results['duplicates']['total_duplicates']
    )
    
    if total_issues > 0:
        print(f"  ⚠️  Found {total_issues:,} total issues that can be automatically fixed")
        print("  🔧 Run super_repair.py to fix all issues")
        
        if results['duplicates']['total_duplicates'] > 0:
            print(f"  🎯 Deduplication would remove {results['duplicates']['total_duplicates']:,} duplicate records")
    else:
        print("  ✅ Dataset is in excellent condition!")
    
    return results


def process_dataset(file_path, dataset_type, output_dir=None):
    """Process one dataset with comprehensive analysis"""
    print(f"\n📊 PROCESSING {dataset_type.upper()}")
    print("=" * 80)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    # Load data
    df = load_data(file_path, dataset_type)
    print(f"📈 Loaded {len(df):,} records")
    
    results = {"dataset_type": dataset_type, "file_path": str(file_path)}
    
    # Run all analyses
    results["0th_order"] = analyze_0th_order_issues(df, dataset_type)
    
    if dataset_type == "billionaires":
        results["1st_order"] = analyze_1st_order_issues(df, dataset_type)
        results["2nd_order"] = analyze_2nd_order_issues(df, dataset_type)
    
    results["duplicates"] = analyze_duplicates(df, dataset_type)
    results["repair_impact"] = simulate_repair_impact(df, dataset_type)
    
    # Generate summary
    generate_summary_report(results, dataset_type)
    
    # Save detailed report if requested
    if output_dir:
        output_path = Path(output_dir) / f"{dataset_type}_analysis_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_results = json.dumps(results, indent=2, default=str)
        output_path.write_text(json_results)
        print(f"\n💾 Detailed report saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive dataset analysis - checks all order issues and duplicates"
    )
    parser.add_argument("--parquet-dir", default="data", help="Data directory")
    parser.add_argument(
        "--dataset", 
        choices=["billionaires", "assets", "both"], 
        default="both",
        help="Which dataset to analyze"
    )
    parser.add_argument(
        "--output-dir", 
        help="Directory to save detailed JSON reports"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Skip repair simulation for faster analysis"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.parquet_dir)
    
    print("🔍 COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 80)
    print(f"📁 Directory: {data_dir.absolute()}")
    print(f"🎯 Analyzing: {args.dataset}")
    print(f"📅 Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.output_dir:
        print(f"📄 Reports will be saved to: {Path(args.output_dir).absolute()}")
    
    success = True
    all_results = {}
    
    # Process billionaires
    if args.dataset in ["billionaires", "both"]:
        billionaires_path = data_dir / "billionaires.parquet"
        if billionaires_path.exists():
            result = process_dataset(billionaires_path, "billionaires", args.output_dir)
            if result:
                all_results["billionaires"] = result
            else:
                success = False
        else:
            print(f"❌ Billionaires file not found: {billionaires_path}")
            success = False
    
    # Process assets
    if args.dataset in ["assets", "both"]:
        assets_path = data_dir / "assets.parquet"
        if assets_path.exists():
            result = process_dataset(assets_path, "assets", args.output_dir)
            if result:
                all_results["assets"] = result
            else:
                success = False
        else:
            print(f"❌ Assets file not found: {assets_path}")
            success = False
    
    # Final summary
    print(f"\n{'=' * 80}")
    if success and all_results:
        print("✅ COMPREHENSIVE ANALYSIS COMPLETED")
        
        # Cross-dataset summary
        if len(all_results) > 1:
            print("\n🌐 CROSS-DATASET SUMMARY:")
            total_issues = 0
            for dataset, results in all_results.items():
                dataset_issues = (
                    results['0th_order']['whitespace'] + 
                    results['0th_order']['unknown'] + 
                    results['duplicates']['total_duplicates']
                )
                total_issues += dataset_issues
                print(f"  {dataset}: {dataset_issues:,} issues")
            
            print(f"  Total across all datasets: {total_issues:,} issues")
        
        print("\n🎯 NEXT STEPS:")
        print("  • Review the analysis above")
        print("  • Run 'python src/super_repair.py' to fix all identified issues")
        print("  • Use '--dry-run' first to see what would be changed")
        
    else:
        print("❌ ANALYSIS FAILED")
    
    return success


if __name__ == "__main__":
    with pl.StringCache():
        success = main()
    sys.exit(0 if success else 1)

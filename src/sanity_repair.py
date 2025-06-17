#!/usr/bin/env python3
"""Comprehensive dataset repair script - applies all repair orders and deduplication"""

import polars as pl
import argparse
from pathlib import Path
import sys
from datetime import datetime
import json
from data_lib import load_data, save_data
from repairs_lib import (
    repair_all_orders,
    count_0th_order_issues,
    analyze_duplicates,
    analyze_repair_impact,
)

pl.enable_string_cache()


def analyze_before_repair(df, dataset_type):
    """Analyze dataset before repairs to establish baseline"""
    print(f"\n📊 PRE-REPAIR ANALYSIS - {dataset_type.upper()}")
    print("=" * 60)
    
    stats = {
        "total_records": len(df),
        "dataset_type": dataset_type,
    }
    
    # 0th order issues
    issues_0th = count_0th_order_issues(df)
    stats["0th_order_issues"] = issues_0th
    print(f"0th order issues: {issues_0th['whitespace']:,} whitespace, {issues_0th['unknown']:,} unknown")
    
    # Identity inconsistencies (billionaires only)
    if dataset_type == "billionaires":
        identity_issues = analyze_identity_inconsistencies(df)
        stats["1st_order_issues"] = identity_issues
        print(f"1st order issues: {sum(identity_issues.values()):,} identity inconsistencies")
        
        # 2nd order fillable nulls
        fill_issues = analyze_fillable_nulls(df)
        stats["2nd_order_issues"] = fill_issues
        print(f"2nd order issues: {sum(v['total_nulls'] for v in fill_issues.values()):,} fillable nulls")
    
    # Duplicate analysis
    dup_stats = analyze_duplicates(df, dataset_type)
    stats["3rd_order_issues"] = dup_stats
    print(f"3rd order issues: {dup_stats.get('total_duplicates', 0):,} duplicate records")
    
    return stats


def analyze_identity_inconsistencies(df):
    """Analyze identity inconsistencies for 1st order repairs"""
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
    
    return inconsistencies


def analyze_fillable_nulls(df):
    """Analyze fillable nulls for 2nd order repairs"""
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
        
        # Count fillable nulls (people who have some data but missing some)
        person_stats = (
            df_clean.group_by("personName")
            .agg([
                pl.col(field).count().alias("total_records"),
                pl.col(field).null_count().alias("null_records"),
            ])
        )
        
        partially_fillable = person_stats.filter(
            (pl.col("null_records") > 0) & (pl.col("null_records") < pl.col("total_records"))
        )
        
        fillable_stats[field] = {
            "total_nulls": df_clean[field].null_count(),
            "fillable_people": len(partially_fillable),
        }
    
    return fillable_stats


def analyze_after_repair(original_df, repaired_df, dataset_type, original_stats):
    """Analyze dataset after repairs to show impact"""
    print(f"\n📈 POST-REPAIR ANALYSIS - {dataset_type.upper()}")
    print("=" * 60)
    
    stats = {
        "total_records": len(repaired_df),
        "dataset_type": dataset_type,
        "records_removed": len(original_df) - len(repaired_df),
    }
    
    # 0th order improvements
    issues_0th_after = count_0th_order_issues(repaired_df)
    stats["0th_order_fixed"] = {
        "whitespace": original_stats["0th_order_issues"]["whitespace"] - issues_0th_after["whitespace"],
        "unknown": original_stats["0th_order_issues"]["unknown"] - issues_0th_after["unknown"],
    }
    
    print(f"0th order fixed: {stats['0th_order_fixed']['whitespace']:,} whitespace, {stats['0th_order_fixed']['unknown']:,} unknown")
    
    # Identity improvements (billionaires only)
    if dataset_type == "billionaires" and "1st_order_issues" in original_stats:
        identity_after = analyze_identity_inconsistencies(repaired_df)
        stats["1st_order_fixed"] = {}
        total_fixed = 0
        for field in identity_after:
            fixed = original_stats["1st_order_issues"][field] - identity_after[field]
            stats["1st_order_fixed"][field] = fixed
            total_fixed += fixed
        
        print(f"1st order fixed: {total_fixed:,} identity inconsistencies")
        
        # 2nd order improvements
        fill_after = analyze_fillable_nulls(repaired_df)
        stats["2nd_order_fixed"] = {}
        total_filled = 0
        for field in fill_after:
            if field in original_stats["2nd_order_issues"]:
                filled = original_stats["2nd_order_issues"][field]["total_nulls"] - fill_after[field]["total_nulls"]
                stats["2nd_order_fixed"][field] = filled
                total_filled += filled
        
        print(f"2nd order fixed: {total_filled:,} null values filled")
    
    # Deduplication improvements
    dup_after = analyze_duplicates(repaired_df, dataset_type)
    duplicates_removed = original_stats["3rd_order_issues"].get("total_duplicates", 0) - dup_after.get("total_duplicates", 0)
    stats["3rd_order_fixed"] = duplicates_removed
    
    print(f"3rd order fixed: {duplicates_removed:,} duplicate records removed")
    print(f"Total records removed: {stats['records_removed']:,}")
    
    return stats


def create_backup(file_path, backup_dir):
    """Create backup of original file"""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    # Copy file
    import shutil
    shutil.copy2(file_path, backup_path)
    
    print(f"💾 Backup created: {backup_path}")
    return backup_path


def process_dataset(
    file_path, 
    dataset_type, 
    output_path=None, 
    backup_dir=None, 
    dry_run=False,
    repair_orders=None
):
    """Process one dataset with comprehensive repairs"""
    print(f"\n🔧 PROCESSING {dataset_type.upper()}")
    print("=" * 80)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    # Load data
    original_df = load_data(file_path, dataset_type)
    print(f"📊 Loaded {len(original_df):,} records")
    
    if len(original_df) == 0:
        print("⚠️ Dataset is empty, skipping repairs")
        return {"status": "empty"}
    
    # Analyze before repair
    original_stats = analyze_before_repair(original_df, dataset_type)
    
    # Create backup if not dry run
    if not dry_run and backup_dir:
        backup_path = create_backup(file_path, backup_dir)
    
    # Apply repairs
    print(f"\n🔧 APPLYING COMPREHENSIVE REPAIRS")
    print("=" * 60)
    
    if repair_orders is None:
        repair_orders = {"0th": True, "1st": True, "2nd": True, "3rd": True}
    
    repaired_df = repair_all_orders(
        original_df,
        dataset_type=dataset_type,
        people_filter=None,  # Apply to all data
        apply_0th=repair_orders.get("0th", True),
        apply_1st=repair_orders.get("1st", True) and dataset_type == "billionaires",
        apply_2nd=repair_orders.get("2nd", True) and dataset_type == "billionaires",
        apply_3rd=repair_orders.get("3rd", True),
    )
    
    # Analyze after repair
    repair_stats = analyze_after_repair(original_df, repaired_df, dataset_type, original_stats)
    
    # Prepare output path
    if output_path is None:
        output_path = file_path
    
    # Save repaired data
    if dry_run:
        print(f"\n🔍 DRY RUN - Would save {len(repaired_df):,} records to: {output_path}")
    else:
        save_data(repaired_df, output_path, dataset_type)
        print(f"💾 Saved {len(repaired_df):,} records to: {output_path}")
    
    # Compile results
    results = {
        "dataset_type": dataset_type,
        "file_path": str(file_path),
        "output_path": str(output_path),
        "dry_run": dry_run,
        "original_stats": original_stats,
        "repair_stats": repair_stats,
        "repair_orders_applied": repair_orders,
    }
    
    if not dry_run and backup_dir:
        results["backup_path"] = str(backup_path)
    
    return results


def generate_repair_summary(all_results):
    """Generate comprehensive summary of all repairs"""
    print(f"\n📊 COMPREHENSIVE REPAIR SUMMARY")
    print("=" * 80)
    
    total_records_before = 0
    total_records_after = 0
    total_issues_fixed = 0
    
    for dataset, results in all_results.items():
        if results.get("status") == "empty":
            continue
            
        original = results["original_stats"]["total_records"]
        repaired = results["repair_stats"]["total_records"]
        
        total_records_before += original
        total_records_after += repaired
        
        print(f"\n📊 {dataset.upper()} DATASET:")
        print(f"  Records: {original:,} → {repaired:,} (removed: {original - repaired:,})")
        
        # 0th order
        if "0th_order_fixed" in results["repair_stats"]:
            fixed_0th = (
                results["repair_stats"]["0th_order_fixed"]["whitespace"] +
                results["repair_stats"]["0th_order_fixed"]["unknown"]
            )
            total_issues_fixed += fixed_0th
            print(f"  0th order fixed: {fixed_0th:,} (whitespace + unknown)")
        
        # 1st order (billionaires only)
        if "1st_order_fixed" in results["repair_stats"]:
            fixed_1st = sum(results["repair_stats"]["1st_order_fixed"].values())
            total_issues_fixed += fixed_1st
            print(f"  1st order fixed: {fixed_1st:,} (identity inconsistencies)")
        
        # 2nd order (billionaires only)
        if "2nd_order_fixed" in results["repair_stats"]:
            fixed_2nd = sum(results["repair_stats"]["2nd_order_fixed"].values())
            total_issues_fixed += fixed_2nd
            print(f"  2nd order fixed: {fixed_2nd:,} (null values filled)")
        
        # 3rd order
        if "3rd_order_fixed" in results["repair_stats"]:
            fixed_3rd = results["repair_stats"]["3rd_order_fixed"]
            total_issues_fixed += fixed_3rd
            print(f"  3rd order fixed: {fixed_3rd:,} (duplicates removed)")
    
    print(f"\n🌐 OVERALL SUMMARY:")
    print(f"  Total records: {total_records_before:,} → {total_records_after:,}")
    print(f"  Records removed: {total_records_before - total_records_after:,}")
    print(f"  Issues fixed: {total_issues_fixed:,}")
    print(f"  Improvement: {(total_records_before - total_records_after) / total_records_before * 100:.2f}% reduction")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive dataset repair - applies all repair orders and deduplication"
    )
    parser.add_argument("--parquet-dir", default="data", help="Data directory")
    parser.add_argument(
        "--dataset", 
        choices=["billionaires", "assets", "both"], 
        default="both",
        help="Which dataset to repair"
    )
    parser.add_argument(
        "--output-suffix", 
        default="", 
        help="Suffix for output files (default: overwrite originals)"
    )
    parser.add_argument(
        "--backup-dir", 
        default="backups", 
        help="Directory for backups (default: backups/)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true", 
        help="Skip creating backups"
    )
    parser.add_argument(
        "--report-dir", 
        help="Directory to save detailed JSON reports"
    )
    
    # Repair order controls
    parser.add_argument("--no-0th-order", action="store_true", help="Skip 0th order repairs")
    parser.add_argument("--no-1st-order", action="store_true", help="Skip 1st order repairs")
    parser.add_argument("--no-2nd-order", action="store_true", help="Skip 2nd order repairs") 
    parser.add_argument("--no-3rd-order", action="store_true", help="Skip 3rd order repairs")
    
    args = parser.parse_args()
    
    data_dir = Path(args.parquet_dir)
    
    print("🔧 COMPREHENSIVE DATASET REPAIR")
    print("=" * 80)
    print(f"📁 Directory: {data_dir.absolute()}")
    print(f"🎯 Repairing: {args.dataset}")
    print(f"📅 Repair time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔒 Dry run: {args.dry_run}")
    
    if not args.no_backup and not args.dry_run:
        print(f"💾 Backups: {Path(args.backup_dir).absolute()}")
    
    # Determine repair orders
    repair_orders = {
        "0th": not args.no_0th_order,
        "1st": not args.no_1st_order,
        "2nd": not args.no_2nd_order,
        "3rd": not args.no_3rd_order,
    }
    
    enabled_orders = [k for k, v in repair_orders.items() if v]
    print(f"🔧 Repair orders: {', '.join(enabled_orders) if enabled_orders else 'None'}")
    
    if not enabled_orders:
        print("⚠️ No repair orders enabled, nothing to do!")
        return True
    
    success = True
    all_results = {}
    
    # Process billionaires
    if args.dataset in ["billionaires", "both"]:
        billionaires_path = data_dir / "billionaires.parquet"
        if billionaires_path.exists():
            output_path = (
                data_dir / f"billionaires{args.output_suffix}.parquet" 
                if args.output_suffix 
                else billionaires_path
            )
            
            result = process_dataset(
                billionaires_path, 
                "billionaires",
                output_path,
                args.backup_dir if not args.no_backup else None,
                args.dry_run,
                repair_orders
            )
            
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
            output_path = (
                data_dir / f"assets{args.output_suffix}.parquet" 
                if args.output_suffix 
                else assets_path
            )
            
            result = process_dataset(
                assets_path, 
                "assets",
                output_path,
                args.backup_dir if not args.no_backup else None,
                args.dry_run,
                repair_orders
            )
            
            if result:
                all_results["assets"] = result
            else:
                success = False
        else:
            print(f"❌ Assets file not found: {assets_path}")
            success = False
    
    # Generate summary and save reports
    if success and all_results:
        generate_repair_summary(all_results)
        
        # Save detailed reports if requested
        if args.report_dir:
            report_dir = Path(args.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"repair_report_{timestamp}.json"
            
            # Convert to JSON-serializable format
            json_results = json.dumps(all_results, indent=2, default=str)
            report_path.write_text(json_results)
            print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Final status
    print(f"\n{'=' * 80}")
    if success:
        print("✅ COMPREHENSIVE REPAIR COMPLETED")
        if not args.dry_run:
            print("🎯 Your datasets have been successfully repaired!")
            print("📊 Run super_check.py to verify the results")
        else:
            print("🔍 DRY RUN completed - use without --dry-run to apply changes")
    else:
        print("❌ REPAIR FAILED")
    
    return success


if __name__ == "__main__":
    with pl.StringCache():
        success = main()
    sys.exit(0 if success else 1)

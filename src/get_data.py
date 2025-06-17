#!/usr/bin/env python3
import polars as pl
import requests
import json
from datetime import datetime
from pathlib import Path
import argparse
import sys
from data_lib import load_data, save_data, get_schema, create_empty
from repairs_lib import repair_all_orders, get_people_in_new_data


def fetch_forbes_data(session):
    """Fetch current data from Forbes API"""
    urls = [
        "https://www.forbes.com/forbesapi/person/rtb/0/position/true.json",
        "https://www.forbes.com/forbesapi/person/rtb/0/-estWorthPrev/true.json?fields=rank,uri,personName,lastName,gender,source,industries,countryOfCitizenship,birthDate,finalWorth,estWorthPrev,imageExists,squareImage,listUri",
        "https://www.forbes.com/forbesapi/person/rtb/0/-estWorthPrev/true.json",
    ]

    print("🌐 Fetching live data from Forbes API...")

    for i, url in enumerate(urls, 1):
        print(f"📡 Trying URL {i}/{len(urls)}...")
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            records = (
                data.get("personList", {}).get("personsLists")
                or data.get("personList")
                or data.get("data", [])
            )

            if records:
                print(f"✅ Found {len(records)} records")
                return data
            print(f"⚠️  Empty data, trying next...")

        except Exception as e:
            print(f"❌ Failed: {e}")

    print("❌ All URLs failed")
    return None


def process_forbes_data(data, current_date):
    """Process Forbes JSON into dataframes"""
    print("🔄 Processing Forbes data...")

    records = (
        data.get("personList", {}).get("personsLists")
        or data.get("personList")
        or data.get("data", [])
    )

    if not records:
        raise ValueError("No records found")

    billionaires_data = []
    assets_data = []

    for r in records:
        # Billionaire record
        billionaire = {
            "date": current_date,
            "personName": str(r.get("personName", "")),
            "lastName": str(r.get("lastName", "")),
            "birthDate": str(r.get("birthDate", "")),
            "gender": str(r.get("gender", "")),
            "countryOfCitizenship": str(r.get("countryOfCitizenship", "")),
            "city": str(r.get("city", "")),
            "state": str(r.get("state", "")),
            "source": str(r.get("source", "")),
            "industries": str(r.get("industries", "")),
            "finalWorth": (
                str(r.get("finalWorth", "")) if r.get("finalWorth") is not None else ""
            ),
            "estWorthPrev": (
                str(r.get("estWorthPrev", ""))
                if r.get("estWorthPrev") is not None
                else ""
            ),
            "archivedWorth": (
                str(r.get("archivedWorth", ""))
                if r.get("archivedWorth") is not None
                else ""
            ),
            "privateAssetsWorth": (
                str(r.get("privateAssetsWorth", ""))
                if r.get("privateAssetsWorth") is not None
                else ""
            ),
        }
        billionaires_data.append(billionaire)

        # Asset records
        for a in r.get("financialAssets", []):
            asset = {
                "date": current_date,
                "personName": str(r.get("personName", "")),
                "companyName": str(a.get("companyName", "")),
                "currencyCode": str(a.get("currencyCode", "")),
                "currentPrice": (
                    str(a.get("currentPrice", ""))
                    if a.get("currentPrice") is not None
                    else ""
                ),
                "exchange": str(a.get("exchange", "")),
                "exchangeRate": (
                    str(a.get("exchangeRate", ""))
                    if a.get("exchangeRate") is not None
                    else ""
                ),
                "exerciseOptionPrice": (
                    str(a.get("exerciseOptionPrice", ""))
                    if a.get("exerciseOptionPrice") is not None
                    else ""
                ),
                "interactive": (
                    str(a.get("interactive", ""))
                    if a.get("interactive") is not None
                    else ""
                ),
                "numberOfShares": (
                    str(a.get("numberOfShares", ""))
                    if a.get("numberOfShares") is not None
                    else ""
                ),
                "sharePrice": (
                    str(a.get("sharePrice", ""))
                    if a.get("sharePrice") is not None
                    else ""
                ),
                "ticker": str(a.get("ticker", "")),
            }
            assets_data.append(asset)

    print(f"✅ Processed {len(billionaires_data)} billionaires")
    print(f"✅ Processed {len(assets_data)} assets")

    return pl.DataFrame(billionaires_data), (
        pl.DataFrame(assets_data) if assets_data else create_empty("assets")
    )


def apply_schema_transformations(df, schema):
    """Apply schema transformations"""
    exprs = []

    for col, dtype in schema.items():
        if col not in df.columns:
            if dtype == pl.Categorical:
                expr = pl.lit(None).cast(pl.Utf8).cast(pl.Categorical).alias(col)
            else:
                expr = pl.lit(None).cast(dtype).alias(col)
            exprs.append(expr)
            continue

        # Handle specific transformations
        if col == "birthDate":
            expr = (
                pl.when((pl.col(col) == "") | pl.col(col).is_null())
                .then(None)
                .otherwise(
                    pl.col(col)
                    .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                    .fill_null(
                        pl.when(pl.col(col).str.len_chars() > 0)
                        .then(
                            pl.col(col)
                            .cast(pl.Int64, strict=False)
                            .cast(pl.Datetime("ms"))
                            .cast(pl.Date)
                        )
                        .otherwise(None)
                    )
                )
            )
        elif dtype == pl.Date and col == "date":
            expr = pl.col(col).str.strptime(pl.Date, "%Y%m%d", strict=False)
        elif "Decimal" in str(dtype):
            expr = (
                pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).cast(dtype)
            )
        elif dtype == pl.Boolean:
            expr = (
                pl.when(pl.col(col).is_in(["True", "true", "1"]))
                .then(True)
                .when(pl.col(col).is_in(["False", "false", "0"]))
                .then(False)
                .otherwise(None)
            )
        elif dtype == pl.Categorical:
            expr = (
                pl.when(pl.col(col) == "")
                .then(None)
                .otherwise(pl.col(col))
                .cast(pl.Categorical)
            )
        else:
            expr = pl.col(col).cast(dtype)

        exprs.append(expr.alias(col))

    return df.select(exprs).select(list(schema.keys()))


def update_dataset(new_df, existing_df, current_date_obj, dataset_name):
    """Update dataset with new data"""
    if len(existing_df) > 0:
        if current_date_obj in existing_df["date"].unique():
            print(f"⚠️  Removing old {current_date_obj} data...")
            existing_df = existing_df.filter(pl.col("date") != current_date_obj)

    combined = (
        pl.concat([existing_df, new_df], how="vertical_relaxed")
        if len(existing_df) > 0
        else new_df
    )
    print(f"🔄 Total records: {len(combined):,}")
    return combined


def apply_repairs_pipeline(
    combined_df,
    new_df,
    dataset_type,
    enable_repairs=True,
    enable_0th=True,
    enable_1st=True,
    enable_2nd=True,
    enable_3rd=True,
):
    """
    Apply repair pipeline with optimization for incremental updates.

    Args:
        combined_df: Full dataset including new data
        new_df: Just the new data being added
        dataset_type: 'billionaires' or 'assets'
        enable_repairs: Whether to apply any repairs
        enable_0th/1st/2nd/3rd: Whether to apply specific repair orders

    Returns:
        Repaired dataframe
    """
    if not enable_repairs:
        print(f"🔄 Skipping repairs for {dataset_type}")
        return combined_df

    print(f"\n🔧 APPLYING REPAIRS TO {dataset_type.upper()}")
    print("=" * 60)

    # For billionaires, we can optimize by focusing on people in new data
    if dataset_type == "billionaires":
        # Get people who appear in new data
        people_in_new_data = get_people_in_new_data(new_df)

        if people_in_new_data:
            print(
                f"🎯 Optimization: Focusing repairs on {len(people_in_new_data)} people from new data"
            )
            people_filter = people_in_new_data
        else:
            print(f"⚠️  No people found in new data, applying repairs to all data")
            people_filter = None
    else:
        # For assets, no person-based optimization needed
        people_filter = None

    # Apply repair pipeline
    repaired = repair_all_orders(
        combined_df,
        dataset_type=dataset_type,
        people_filter=people_filter,
        apply_0th=enable_0th,
        apply_1st=enable_1st,
        apply_2nd=enable_2nd,
        apply_3rd=enable_3rd,
    )

    return repaired


def main():
    parser = argparse.ArgumentParser(
        description="Update Forbes dataset with integrated repairs and deduplication"
    )
    parser.add_argument("--parquet-dir", default="data")
    parser.add_argument(
        "--user-agent", default="Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N)"
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")

    # Repair control arguments
    parser.add_argument("--no-repairs", action="store_true", help="Skip all repairs")
    parser.add_argument(
        "--no-0th-order",
        action="store_true",
        help="Skip 0th order repairs (whitespace/unknowns)",
    )
    parser.add_argument(
        "--no-1st-order",
        action="store_true",
        help="Skip 1st order repairs (identity consistency)",
    )
    parser.add_argument(
        "--no-2nd-order",
        action="store_true",
        help="Skip 2nd order repairs (forward/backward fill)",
    )
    parser.add_argument(
        "--no-3rd-order",
        action="store_true",
        help="Skip 3rd order repairs (deduplication)",
    )

    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    parquet_dir.mkdir(exist_ok=True, parents=True)

    billionaires_path = parquet_dir / "billionaires.parquet"
    assets_path = parquet_dir / "assets.parquet"
    current_date = datetime.now().strftime("%Y%m%d")

    print("🚀 Forbes Live Data Updater with Integrated Repairs and Deduplication")
    print("=" * 80)
    print(f"📅 Date: {current_date}")
    print(f"📁 Directory: {parquet_dir.absolute()}")
    print(f"🔒 Dry run: {args.dry_run}")

    # Repair settings
    enable_repairs = not args.no_repairs
    enable_0th = not args.no_0th_order
    enable_1st = not args.no_1st_order
    enable_2nd = not args.no_2nd_order
    enable_3rd = not args.no_3rd_order

    print(f"🔧 Repairs enabled: {enable_repairs}")
    if enable_repairs:
        print(f"   0th order (clean): {enable_0th}")
        print(f"   1st order (identity): {enable_1st}")
        print(f"   2nd order (fill): {enable_2nd}")
        print(f"   3rd order (deduplication): {enable_3rd}")

    session = requests.Session()
    session.headers = {"User-Agent": args.user_agent, "Accept": "application/json"}

    try:
        # Fetch data
        forbes_data = fetch_forbes_data(session)
        if not forbes_data:
            return False

        # Process data
        new_billionaires_raw, new_assets_raw = process_forbes_data(
            forbes_data, current_date
        )

        # Apply schemas
        print("🔄 Applying schemas...")
        new_billionaires = apply_schema_transformations(
            new_billionaires_raw, get_schema("billionaires")
        )
        new_assets = apply_schema_transformations(new_assets_raw, get_schema("assets"))

        if args.dry_run:
            print(f"\n🔍 DRY RUN - Would process:")
            print(f"   📊 Billionaires: {len(new_billionaires)} records")
            print(f"   💰 Assets: {len(new_assets)} records")
            print(f"   🔧 Repairs: {enable_repairs}")
            if enable_repairs:
                print(
                    f"      0th={enable_0th}, 1st={enable_1st}, 2nd={enable_2nd}, 3rd={enable_3rd}"
                )
            return True

        # Load existing data
        existing_billionaires = (
            load_data(billionaires_path, "billionaires")
            if billionaires_path.exists()
            else create_empty("billionaires")
        )
        existing_assets = (
            load_data(assets_path, "assets")
            if assets_path.exists()
            else create_empty("assets")
        )

        # Update datasets (combine new with existing)
        current_date_obj = datetime.strptime(current_date, "%Y%m%d").date()

        print("\n" + "=" * 80)
        print("UPDATING BILLIONAIRES")
        combined_billionaires = update_dataset(
            new_billionaires, existing_billionaires, current_date_obj, "billionaires"
        )

        print("\n" + "=" * 80)
        print("UPDATING ASSETS")
        combined_assets = update_dataset(
            new_assets, existing_assets, current_date_obj, "assets"
        )

        # Apply repairs
        print("\n" + "=" * 80)
        print("REPAIR PIPELINE")

        final_billionaires = apply_repairs_pipeline(
            combined_billionaires,
            new_billionaires,
            "billionaires",
            enable_repairs,
            enable_0th,
            enable_1st,
            enable_2nd,
            enable_3rd,
        )

        final_assets = apply_repairs_pipeline(
            combined_assets,
            new_assets,
            "assets",
            enable_repairs,
            enable_0th,
            False,  # No 1st order for assets
            False,  # No 2nd order for assets
            enable_3rd,  # Deduplication for assets
        )

        # Save
        print("\n" + "=" * 80)
        print("SAVING DATASETS")
        save_data(final_billionaires, billionaires_path, "billionaires")
        save_data(final_assets, assets_path, "assets")

        print("\n" + "=" * 80)
        print("✅ UPDATE COMPLETED")
        print(f"📊 Billionaires: {len(final_billionaires):,} records")
        print(f"💰 Assets: {len(final_assets):,} records")
        if enable_repairs:
            print(
                f"🔧 Repairs applied: 0th={enable_0th}, 1st={enable_1st}, 2nd={enable_2nd}, 3rd={enable_3rd}"
            )

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        session.close()


if __name__ == "__main__":
    with pl.StringCache():
        success = main()
    sys.exit(0 if success else 1)

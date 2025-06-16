#!/usr/bin/env python3
import polars as pl
import requests
import json
from datetime import datetime
from pathlib import Path
import argparse
import sys


def get_billionaires_schema():
    """Schema for billionaires data"""
    return {
        "date": pl.Date,
        "personName": pl.Categorical,
        "lastName": pl.Categorical,
        "birthDate": pl.Date,
        "gender": pl.Categorical,
        "countryOfCitizenship": pl.Categorical,
        "city": pl.Categorical,
        "state": pl.Categorical,
        "source": pl.Categorical,
        "industries": pl.Categorical,
        "finalWorth": pl.Decimal(precision=18, scale=8),
        "estWorthPrev": pl.Decimal(precision=18, scale=8),
        "archivedWorth": pl.Decimal(precision=18, scale=8),
        "privateAssetsWorth": pl.Decimal(precision=18, scale=8),
    }


def get_assets_schema():
    """Schema for assets data"""
    return {
        "date": pl.Date,
        "personName": pl.Categorical,
        "companyName": pl.Categorical,
        "currencyCode": pl.Categorical,
        "currentPrice": pl.Decimal(precision=18, scale=11),
        "exchange": pl.Categorical,
        "exchangeRate": pl.Decimal(precision=18, scale=8),
        "exerciseOptionPrice": pl.Decimal(precision=18, scale=11),
        "interactive": pl.Boolean,
        "numberOfShares": pl.Decimal(precision=18, scale=2),
        "sharePrice": pl.Decimal(precision=18, scale=11),
        "ticker": pl.Categorical,
    }


def fetch_forbes_data(session):
    """Fetch current data from Forbes API"""

    forbes_urls = [
        "https://www.forbes.com/forbesapi/person/rtb/0/position/true.json",
        "https://www.forbes.com/forbesapi/person/rtb/0/-estWorthPrev/true.json?fields=rank,uri,personName,lastName,gender,source,industries,countryOfCitizenship,birthDate,finalWorth,estWorthPrev,imageExists,squareImage,listUri",
        "https://www.forbes.com/forbesapi/person/rtb/0/-estWorthPrev/true.json",
    ]

    print("🌐 Fetching live data from Forbes API...")

    for i, url in enumerate(forbes_urls, 1):
        print(f"📡 Trying URL {i}/{len(forbes_urls)}: {url[:80]}...")

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            records = (
                data.get("personList", {}).get("personsLists")
                or data.get("personList")
                or data.get("data", [])
            )

            if records and len(records) > 0:
                print(f"✅ Successfully fetched {len(records)} records from Forbes")
                return data
            else:
                print(f"⚠️  URL {i} returned empty data, trying next...")

        except requests.exceptions.RequestException as e:
            print(f"❌ URL {i} failed: {e}")
            continue
        except json.JSONDecodeError as e:
            print(f"❌ URL {i} returned invalid JSON: {e}")
            continue

    print("❌ All Forbes URLs failed to return valid data")
    return None


def process_forbes_data(data, current_date):
    """Process Forbes JSON data into billionaires and assets DataFrames"""

    print("🔄 Processing Forbes data...")

    records = (
        data.get("personList", {}).get("personsLists")
        or data.get("personList")
        or data.get("data", [])
    )

    if not records:
        raise ValueError("No records found in Forbes data")

    billionaires_data = []
    assets_data = []

    for record in records:
        billionaire = {
            "date": current_date,
            "personName": (
                str(record.get("personName", "")) if record.get("personName") else ""
            ),
            "lastName": (
                str(record.get("lastName", "")) if record.get("lastName") else ""
            ),
            "birthDate": (
                str(record.get("birthDate", "")) if record.get("birthDate") else ""
            ),
            "gender": str(record.get("gender", "")) if record.get("gender") else "",
            "countryOfCitizenship": (
                str(record.get("countryOfCitizenship", ""))
                if record.get("countryOfCitizenship")
                else ""
            ),
            "city": str(record.get("city", "")) if record.get("city") else "",
            "state": str(record.get("state", "")) if record.get("state") else "",
            "source": str(record.get("source", "")) if record.get("source") else "",
            "industries": (
                str(record.get("industries", "")) if record.get("industries") else ""
            ),
            "finalWorth": (
                str(record.get("finalWorth", ""))
                if record.get("finalWorth") is not None
                else ""
            ),
            "estWorthPrev": (
                str(record.get("estWorthPrev", ""))
                if record.get("estWorthPrev") is not None
                else ""
            ),
            "archivedWorth": (
                str(record.get("archivedWorth", ""))
                if record.get("archivedWorth") is not None
                else ""
            ),
            "privateAssetsWorth": (
                str(record.get("privateAssetsWorth", ""))
                if record.get("privateAssetsWorth") is not None
                else ""
            ),
        }
        billionaires_data.append(billionaire)

        for asset in record.get("financialAssets", []):
            asset_record = {
                "date": current_date,
                "personName": (
                    str(record.get("personName", ""))
                    if record.get("personName")
                    else ""
                ),
                "companyName": (
                    str(asset.get("companyName", ""))
                    if asset.get("companyName")
                    else ""
                ),
                "currencyCode": (
                    str(asset.get("currencyCode", ""))
                    if asset.get("currencyCode")
                    else ""
                ),
                "currentPrice": (
                    str(asset.get("currentPrice", ""))
                    if asset.get("currentPrice") is not None
                    else ""
                ),
                "exchange": (
                    str(asset.get("exchange", "")) if asset.get("exchange") else ""
                ),
                "exchangeRate": (
                    str(asset.get("exchangeRate", ""))
                    if asset.get("exchangeRate") is not None
                    else ""
                ),
                "exerciseOptionPrice": (
                    str(asset.get("exerciseOptionPrice", ""))
                    if asset.get("exerciseOptionPrice") is not None
                    else ""
                ),
                "interactive": (
                    str(asset.get("interactive", ""))
                    if asset.get("interactive") is not None
                    else ""
                ),
                "numberOfShares": (
                    str(asset.get("numberOfShares", ""))
                    if asset.get("numberOfShares") is not None
                    else ""
                ),
                "sharePrice": (
                    str(asset.get("sharePrice", ""))
                    if asset.get("sharePrice") is not None
                    else ""
                ),
                "ticker": str(asset.get("ticker", "")) if asset.get("ticker") else "",
            }
            assets_data.append(asset_record)

    billionaires_df = pl.DataFrame(billionaires_data)
    assets_df = (
        pl.DataFrame(assets_data)
        if assets_data
        else pl.DataFrame(schema={k: pl.Utf8 for k in get_assets_schema().keys()})
    )

    print(f"✅ Processed {len(billionaires_df)} billionaire records")
    print(f"✅ Processed {len(assets_df)} asset records")

    return billionaires_df, assets_df


def apply_schema_transformations(df, target_schema):
    """Apply schema transformations without float conversion"""

    column_expressions = []

    for col_name, dtype in target_schema.items():
        if col_name not in df.columns:
            if dtype == pl.Categorical:
                expr = pl.lit(None).cast(pl.Utf8).cast(pl.Categorical)
            else:
                expr = pl.lit(None).cast(dtype)
            column_expressions.append(expr.alias(col_name))
            continue

        if col_name == "birthDate":
            expr = (
                pl.when((pl.col(col_name) == "") | pl.col(col_name).is_null())
                .then(None)
                .otherwise(
                    pl.col(col_name)
                    .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                    .fill_null(
                        pl.when(pl.col(col_name).str.len_chars() > 0)
                        .then(
                            pl.col(col_name)
                            .cast(pl.Int64, strict=False)
                            .cast(pl.Datetime(time_unit="ms"))
                            .cast(pl.Date)
                        )
                        .otherwise(None)
                    )
                )
                .alias(col_name)
            )
        elif dtype == pl.Date:
            if col_name == "date":
                expr = pl.col(col_name).str.strptime(pl.Date, "%Y%m%d", strict=False)
            else:
                expr = pl.col(col_name).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        elif "Decimal" in str(dtype):
            expr = (
                pl.when(pl.col(col_name) == "")
                .then(None)
                .otherwise(pl.col(col_name))
                .cast(dtype)
                .alias(col_name)
            )
        elif dtype == pl.Boolean:
            expr = (
                pl.when(pl.col(col_name).is_in(["True", "true", "1", "TRUE"]))
                .then(True)
                .when(pl.col(col_name).is_in(["False", "false", "0", "FALSE"]))
                .then(False)
                .otherwise(None)
                .alias(col_name)
            )
        elif dtype == pl.Categorical:
            expr = (
                pl.when(pl.col(col_name) == "")
                .then(None)
                .otherwise(pl.col(col_name))
                .cast(pl.Categorical)
                .alias(col_name)
            )
        else:
            expr = pl.col(col_name).cast(dtype).alias(col_name)

        column_expressions.append(expr)

    return df.select(column_expressions).select(list(target_schema.keys()))


def load_or_create_parquet(file_path, schema_func):
    """Load existing parquet file or create empty DataFrame with correct schema"""

    if file_path.exists():
        print(f"📖 Loading existing {file_path.name}...")
        return pl.read_parquet(file_path)
    else:
        print(f"🆕 Creating new {file_path.name} (file doesn't exist)...")
        schema = schema_func()
        return pl.DataFrame(schema=schema).select(list(schema.keys()))


def update_dataset(new_df, existing_df, current_date, dataset_name, sort_columns):
    """Update dataset by removing existing date data and adding new data"""

    current_date_obj = datetime.strptime(current_date, "%Y%m%d").date()

    if len(existing_df) > 0:
        existing_dates = existing_df.select("date").unique().to_series().to_list()
        if current_date_obj in existing_dates:
            print(
                f"⚠️  Date {current_date} already exists in {dataset_name}, removing old data..."
            )
            existing_df = existing_df.filter(pl.col("date") != current_date_obj)
            print(f"✅ Removed old data for {current_date}")

    if len(existing_df) > 0:
        combined_df = pl.concat([existing_df, new_df], how="vertical_relaxed")
        print(
            f"🔄 Combined {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total records"
        )
    else:
        combined_df = new_df
        print(f"🆕 Created new dataset with {len(new_df)} records")

    print(f"🔀 Sorting {dataset_name} by {', '.join(sort_columns)}...")
    sorted_df = combined_df.sort(sort_columns)

    return sorted_df


def main():
    parser = argparse.ArgumentParser(
        description="Update Forbes parquet dataset with live data"
    )
    parser.add_argument(
        "--parquet-dir",
        default="data",
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N)",
        help="User agent for requests",
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Request timeout in seconds"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    parquet_dir.mkdir(exist_ok=True, parents=True)

    billionaires_path = parquet_dir / "billionaires.parquet"
    assets_path = parquet_dir / "assets.parquet"

    current_date = datetime.now().strftime("%Y%m%d")

    print("🚀 Forbes Live Data Updater")
    print("=" * 60)
    print(f"📅 Current date: {current_date}")
    print(f"📁 Parquet directory: {parquet_dir.absolute()}")
    print(f"🔒 Dry run: {args.dry_run}")
    print()

    session = requests.Session()
    session.headers = {
        "User-Agent": args.user_agent,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        forbes_data = fetch_forbes_data(session)
        if not forbes_data:
            print("❌ Failed to fetch Forbes data")
            return False

        new_billionaires_raw, new_assets_raw = process_forbes_data(
            forbes_data, current_date
        )

        print("🔄 Applying schema transformations to billionaires...")
        new_billionaires = apply_schema_transformations(
            new_billionaires_raw, get_billionaires_schema()
        )

        print("🔄 Applying schema transformations to assets...")
        new_assets = apply_schema_transformations(new_assets_raw, get_assets_schema())

        if args.dry_run:
            print("\n🔍 DRY RUN - Would process:")
            print(f"   📊 Billionaires: {len(new_billionaires)} records")
            print(f"   💰 Assets: {len(new_assets)} records")
            print(f"   📅 Date: {current_date}")
            print("   ✅ No files modified (dry run)")
            return True

        existing_billionaires = load_or_create_parquet(
            billionaires_path, get_billionaires_schema
        )
        existing_assets = load_or_create_parquet(assets_path, get_assets_schema)

        print("\n" + "=" * 60)
        print("UPDATING BILLIONAIRES DATASET")
        final_billionaires = update_dataset(
            new_billionaires,
            existing_billionaires,
            current_date,
            "billionaires",
            ["personName", "date"],
        )

        print("\n" + "=" * 60)
        print("UPDATING ASSETS DATASET")
        final_assets = update_dataset(
            new_assets,
            existing_assets,
            current_date,
            "assets",
            ["personName", "companyName", "interactive", "date"],
        )

        print("\n" + "=" * 60)
        print("SAVING UPDATED DATASETS")
        print(f"💾 Saving billionaires to {billionaires_path} (brotli compression)...")
        final_billionaires.write_parquet(
            billionaires_path, compression="brotli", compression_level=11
        )

        print(f"💾 Saving assets to {assets_path} (brotli compression)...")
        final_assets.write_parquet(
            assets_path, compression="brotli", compression_level=11
        )

        print("\n" + "=" * 60)
        print("✅ UPDATE COMPLETED")
        print("=" * 60)
        print(f"📊 Final billionaires: {len(final_billionaires):,} records")
        print(f"💰 Final assets: {len(final_assets):,} records")
        print(f"📅 Data date: {current_date}")
        print(f"📁 Files saved to: {parquet_dir.absolute()}")
        print(f"🗜️  Compression: brotli (level 11)")
        print(f"🔒 Decimal precision preserved (no float conversion)")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        session.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

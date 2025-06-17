#!/usr/bin/env python3
"""Minimal data library for Forbes billionaires dataset"""

import polars as pl
from pathlib import Path

# Schemas
BILLIONAIRES_SCHEMA = {
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
    "finalWorth": pl.Decimal(18, 8),
    "estWorthPrev": pl.Decimal(18, 8),
    "archivedWorth": pl.Decimal(18, 8),
    "privateAssetsWorth": pl.Decimal(18, 8),
}

ASSETS_SCHEMA = {
    "date": pl.Date,
    "personName": pl.Categorical,
    "companyName": pl.Categorical,
    "currencyCode": pl.Categorical,
    "currentPrice": pl.Decimal(18, 11),
    "exchange": pl.Categorical,
    "exchangeRate": pl.Decimal(18, 8),
    "exerciseOptionPrice": pl.Decimal(18, 11),
    "interactive": pl.Boolean,
    "numberOfShares": pl.Decimal(18, 2),
    "sharePrice": pl.Decimal(18, 11),
    "ticker": pl.Categorical,
}

SORT_KEYS = {
    "billionaires": ["personName", "date"],
    "assets": ["personName", "companyName", "interactive", "date"],
}


def get_schema(dataset_type):
    """Get schema for dataset type"""
    if dataset_type == "billionaires":
        return BILLIONAIRES_SCHEMA
    elif dataset_type == "assets":
        return ASSETS_SCHEMA
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")


def load_data(path, dataset_type=None):
    """Load dataset from parquet file"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"📖 Loading {path.name}...")
    df = pl.read_parquet(path)

    # Auto-detect dataset type if not provided
    if dataset_type is None:
        if "billionaires" in path.name:
            dataset_type = "billionaires"
        elif "assets" in path.name:
            dataset_type = "assets"

    # Apply schema if type known
    if dataset_type:
        df = enforce_schema(df, dataset_type)

    print(f"✅ Loaded {len(df):,} records")
    if "date" in df.columns:
        dates = df["date"].n_unique()
        print(f"📅 {dates} dates: {df['date'].min()} to {df['date'].max()}")

    return df


def save_data(df, path, dataset_type=None):
    """Save dataset to parquet with compression"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect dataset type
    if dataset_type is None:
        if "billionaires" in path.name:
            dataset_type = "billionaires"
        elif "assets" in path.name:
            dataset_type = "assets"

    # Apply schema and sort if type known
    if dataset_type:
        df = enforce_schema(df, dataset_type)
        sort_cols = SORT_KEYS.get(dataset_type)
        if sort_cols:
            print(f"🔀 Sorting by {', '.join(sort_cols)}...")
            df = df.sort(sort_cols)

    print(f"💾 Saving to {path.name}...")
    df.write_parquet(path, compression="brotli", compression_level=11)
    print(f"✅ Saved {len(df):,} records")


def enforce_schema(df, dataset_type):
    """Apply schema to dataframe"""
    schema = get_schema(dataset_type)

    for col, dtype in schema.items():
        if col not in df.columns:
            # Add missing column
            if dtype == pl.Categorical:
                df = df.with_columns(
                    pl.lit(None).cast(pl.Utf8).cast(pl.Categorical).alias(col)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            # Fix type if needed
            if df.schema[col] != dtype:
                df = df.with_columns(pl.col(col).cast(dtype))

    # Ensure column order
    return df.select(list(schema.keys()))


def create_empty(dataset_type):
    """Create empty dataset with schema"""
    schema = get_schema(dataset_type)
    return pl.DataFrame(schema=schema)


# Shortcuts for backward compatibility
def load_billionaires_data(path, enforce_schema=True):
    return load_data(path, "billionaires")


def load_assets_data(path, enforce_schema=True):
    return load_data(path, "assets")


def save_billionaires_data(df, path, sort_data=True, enforce_schema=True):
    save_data(df, path, "billionaires")


def save_assets_data(df, path, sort_data=True, enforce_schema=True):
    save_data(df, path, "assets")


def get_billionaires_schema():
    return BILLIONAIRES_SCHEMA


def get_assets_schema():
    return ASSETS_SCHEMA


def create_empty_dataset(dataset_type):
    return create_empty(dataset_type)


def load_dataset(path, dataset_type, enforce_schema=True):
    return load_data(path, dataset_type)


def save_dataset(df, path, dataset_type, sort_data=True, enforce_schema=True):
    save_data(df, path, dataset_type)

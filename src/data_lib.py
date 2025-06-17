#!/usr/bin/env python3
"""
Data Library for Forbes Billionaires Dataset

Provides consistent loading and saving functions with proper:
- Schema enforcement
- Data type consistency
- Sorting before save
- Compression settings
"""

import polars as pl
from pathlib import Path
from typing import Union, Optional


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


def get_sort_columns(dataset_type: str) -> list:
    """Get proper sort columns for each dataset type"""
    if dataset_type == "billionaires":
        return ["personName", "date"]
    elif dataset_type == "assets":
        return ["personName", "companyName", "interactive", "date"]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_billionaires_data(
    parquet_path: Union[str, Path], enforce_schema: bool = True
) -> pl.DataFrame:
    """
    Load billionaires dataset with proper schema enforcement

    Args:
        parquet_path: Path to parquet file
        enforce_schema: Whether to enforce schema (default: True)

    Returns:
        Polars DataFrame with proper schema

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    print(f"📖 Loading billionaires dataset from {parquet_path}")
    df = pl.read_parquet(parquet_path)

    if enforce_schema:
        schema = get_billionaires_schema()

        # Ensure all expected columns exist with proper types
        for col_name, expected_dtype in schema.items():
            if col_name not in df.columns:
                # Add missing column with null values
                if expected_dtype == pl.Categorical:
                    df = df.with_columns(
                        pl.lit(None).cast(pl.Utf8).cast(pl.Categorical).alias(col_name)
                    )
                else:
                    df = df.with_columns(
                        pl.lit(None).cast(expected_dtype).alias(col_name)
                    )
            else:
                # Ensure correct data type
                current_dtype = df.schema[col_name]
                if current_dtype != expected_dtype:
                    df = df.with_columns(pl.col(col_name).cast(expected_dtype))

        # Ensure column order matches schema
        df = df.select(list(schema.keys()))

    print(f"✅ Loaded {len(df):,} records from {df['date'].n_unique()} unique dates")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def load_assets_data(
    parquet_path: Union[str, Path], enforce_schema: bool = True
) -> pl.DataFrame:
    """
    Load assets dataset with proper schema enforcement

    Args:
        parquet_path: Path to parquet file
        enforce_schema: Whether to enforce schema (default: True)

    Returns:
        Polars DataFrame with proper schema

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    print(f"📖 Loading assets dataset from {parquet_path}")
    df = pl.read_parquet(parquet_path)

    if enforce_schema:
        schema = get_assets_schema()

        # Ensure all expected columns exist with proper types
        for col_name, expected_dtype in schema.items():
            if col_name not in df.columns:
                # Add missing column with null values
                if expected_dtype == pl.Categorical:
                    df = df.with_columns(
                        pl.lit(None).cast(pl.Utf8).cast(pl.Categorical).alias(col_name)
                    )
                else:
                    df = df.with_columns(
                        pl.lit(None).cast(expected_dtype).alias(col_name)
                    )
            else:
                # Ensure correct data type
                current_dtype = df.schema[col_name]
                if current_dtype != expected_dtype:
                    df = df.with_columns(pl.col(col_name).cast(expected_dtype))

        # Ensure column order matches schema
        df = df.select(list(schema.keys()))

    print(f"✅ Loaded {len(df):,} records from {df['date'].n_unique()} unique dates")

    return df


def save_billionaires_data(
    df: pl.DataFrame,
    parquet_path: Union[str, Path],
    sort_data: bool = True,
    enforce_schema: bool = True,
) -> None:
    """
    Save billionaires dataset with proper sorting and compression

    Args:
        df: Polars DataFrame to save
        parquet_path: Path to save parquet file
        sort_data: Whether to sort before saving (default: True)
        enforce_schema: Whether to enforce schema before saving (default: True)
    """
    parquet_path = Path(parquet_path)

    # Ensure directory exists
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Enforce schema if requested
    if enforce_schema:
        schema = get_billionaires_schema()

        # Ensure all expected columns exist with proper types
        for col_name, expected_dtype in schema.items():
            if col_name not in df.columns:
                # Add missing column with null values
                if expected_dtype == pl.Categorical:
                    df = df.with_columns(
                        pl.lit(None).cast(pl.Utf8).cast(pl.Categorical).alias(col_name)
                    )
                else:
                    df = df.with_columns(
                        pl.lit(None).cast(expected_dtype).alias(col_name)
                    )
            else:
                # Ensure correct data type
                current_dtype = df.schema[col_name]
                if current_dtype != expected_dtype:
                    df = df.with_columns(pl.col(col_name).cast(expected_dtype))

        # Ensure column order matches schema
        df = df.select(list(schema.keys()))

    # Sort data if requested
    if sort_data:
        sort_columns = get_sort_columns("billionaires")
        print(f"🔀 Sorting billionaires by {', '.join(sort_columns)}...")
        df = df.sort(sort_columns)

    # Save with compression
    print(f"💾 Saving billionaires to {parquet_path} (brotli compression)...")
    df.write_parquet(parquet_path, compression="brotli", compression_level=11)

    print(f"✅ Saved {len(df):,} records to {parquet_path}")


def save_assets_data(
    df: pl.DataFrame,
    parquet_path: Union[str, Path],
    sort_data: bool = True,
    enforce_schema: bool = True,
) -> None:
    """
    Save assets dataset with proper sorting and compression

    Args:
        df: Polars DataFrame to save
        parquet_path: Path to save parquet file
        sort_data: Whether to sort before saving (default: True)
        enforce_schema: Whether to enforce schema before saving (default: True)
    """
    parquet_path = Path(parquet_path)

    # Ensure directory exists
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Enforce schema if requested
    if enforce_schema:
        schema = get_assets_schema()

        # Ensure all expected columns exist with proper types
        for col_name, expected_dtype in schema.items():
            if col_name not in df.columns:
                # Add missing column with null values
                if expected_dtype == pl.Categorical:
                    df = df.with_columns(
                        pl.lit(None).cast(pl.Utf8).cast(pl.Categorical).alias(col_name)
                    )
                else:
                    df = df.with_columns(
                        pl.lit(None).cast(expected_dtype).alias(col_name)
                    )
            else:
                # Ensure correct data type
                current_dtype = df.schema[col_name]
                if current_dtype != expected_dtype:
                    df = df.with_columns(pl.col(col_name).cast(expected_dtype))

        # Ensure column order matches schema
        df = df.select(list(schema.keys()))

    # Sort data if requested
    if sort_data:
        sort_columns = get_sort_columns("assets")
        print(f"🔀 Sorting assets by {', '.join(sort_columns)}...")
        df = df.sort(sort_columns)

    # Save with compression
    print(f"💾 Saving assets to {parquet_path} (brotli compression)...")
    df.write_parquet(parquet_path, compression="brotli", compression_level=11)

    print(f"✅ Saved {len(df):,} records to {parquet_path}")


def load_dataset(
    parquet_path: Union[str, Path],
    dataset_type: str = "billionaires",
    enforce_schema: bool = True,
) -> pl.DataFrame:
    """
    Generic function to load any dataset type

    Args:
        parquet_path: Path to parquet file
        dataset_type: Type of dataset ("billionaires" or "assets")
        enforce_schema: Whether to enforce schema (default: True)

    Returns:
        Polars DataFrame with proper schema
    """
    if dataset_type == "billionaires":
        return load_billionaires_data(parquet_path, enforce_schema)
    elif dataset_type == "assets":
        return load_assets_data(parquet_path, enforce_schema)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def save_dataset(
    df: pl.DataFrame,
    parquet_path: Union[str, Path],
    dataset_type: str = "billionaires",
    sort_data: bool = True,
    enforce_schema: bool = True,
) -> None:
    """
    Generic function to save any dataset type

    Args:
        df: Polars DataFrame to save
        parquet_path: Path to save parquet file
        dataset_type: Type of dataset ("billionaires" or "assets")
        sort_data: Whether to sort before saving (default: True)
        enforce_schema: Whether to enforce schema before saving (default: True)
    """
    if dataset_type == "billionaires":
        save_billionaires_data(df, parquet_path, sort_data, enforce_schema)
    elif dataset_type == "assets":
        save_assets_data(df, parquet_path, sort_data, enforce_schema)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_empty_dataset(dataset_type: str = "billionaires") -> pl.DataFrame:
    """
    Create an empty dataset with proper schema

    Args:
        dataset_type: Type of dataset ("billionaires" or "assets")

    Returns:
        Empty Polars DataFrame with proper schema
    """
    if dataset_type == "billionaires":
        schema = get_billionaires_schema()
    elif dataset_type == "assets":
        schema = get_assets_schema()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return pl.DataFrame(schema=schema)


def validate_dataset_schema(
    df: pl.DataFrame, dataset_type: str = "billionaires"
) -> tuple[bool, list]:
    """
    Validate that a dataset matches the expected schema

    Args:
        df: Polars DataFrame to validate
        dataset_type: Type of dataset ("billionaires" or "assets")

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    if dataset_type == "billionaires":
        expected_schema = get_billionaires_schema()
    elif dataset_type == "assets":
        expected_schema = get_assets_schema()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    issues = []

    # Check for missing columns
    for col_name in expected_schema.keys():
        if col_name not in df.columns:
            issues.append(f"Missing column: {col_name}")

    # Check for extra columns
    for col_name in df.columns:
        if col_name not in expected_schema:
            issues.append(f"Unexpected column: {col_name}")

    # Check data types for existing columns
    for col_name, expected_dtype in expected_schema.items():
        if col_name in df.columns:
            actual_dtype = df.schema[col_name]
            if actual_dtype != expected_dtype:
                issues.append(
                    f"Column {col_name}: expected {expected_dtype}, got {actual_dtype}"
                )

    return len(issues) == 0, issues


# Backwards compatibility aliases
def load_or_create_parquet(file_path, schema_func):
    """Legacy function for backwards compatibility"""
    if file_path.exists():
        return pl.read_parquet(file_path)
    else:
        schema = schema_func()
        return pl.DataFrame(schema=schema).select(list(schema.keys()))

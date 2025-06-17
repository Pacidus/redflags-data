#!/usr/bin/env python3
"""Repair functions library for Forbes billionaires dataset"""

import polars as pl
import re
from typing import List, Dict, Optional, Tuple

pl.enable_string_cache()


# ============================================================================
# 0TH ORDER REPAIRS (Whitespace and Unknown Values)
# ============================================================================


def clean_whitespace_and_unknowns(
    df: pl.DataFrame, dataset_type: str = None
) -> pl.DataFrame:
    """
    Clean whitespace and unknown values from string columns.

    Args:
        df: Input dataframe
        dataset_type: Optional dataset type for logging

    Returns:
        Cleaned dataframe
    """
    if dataset_type:
        print(f"🧹 0th order: Cleaning {dataset_type}")

    # Get string columns
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in [pl.Utf8, pl.Categorical]
    ]

    if not string_cols:
        return df

    # Unknown patterns - only match "unknown" and "unknown_123" style
    patterns = [r"(?i)^unknown$", r"(?i)^unknown_-?\d+$"]

    def clean_value(val):
        if val is None:
            return None
        cleaned = val.strip()
        if cleaned == "":
            return None
        for pat in patterns:
            if re.search(pat, cleaned):
                return None
        return cleaned

    # Clean each column
    exprs = []
    for col in string_cols:
        dtype = df.schema[col]
        expr = (
            pl.col(col)
            .cast(pl.Utf8)
            .map_elements(clean_value, return_dtype=pl.Utf8)
            .cast(dtype)
            .alias(col)
        )
        exprs.append(expr)

    # Add non-string columns
    non_string = [col for col in df.columns if col not in string_cols]
    all_exprs = [pl.col(col) for col in non_string] + exprs

    cleaned = df.select(all_exprs)

    if dataset_type:
        print(f"   ✓ Cleaned {len(string_cols)} string columns")

    return cleaned


def count_0th_order_issues(df: pl.DataFrame) -> Dict[str, int]:
    """Count 0th order issues in dataframe"""
    string_cols = [
        col for col, dtype in df.schema.items() if dtype in (pl.Utf8, pl.Categorical)
    ]

    if not string_cols:
        return {"whitespace": 0, "unknown": 0}

    unk_pattern = r"(?i)^(unknown|unknown_-?\d+)$"
    whitespace = unknown = 0

    for col in string_cols:
        col_str = pl.col(col).cast(pl.Utf8)

        # Whitespace issues
        ws = df.filter(
            col_str.is_not_null() & (col_str != col_str.str.strip_chars())
        ).height

        # Unknown variations
        unk = df.filter(
            col_str.is_not_null() & col_str.str.contains(unk_pattern)
        ).height

        whitespace += ws
        unknown += unk

    return {"whitespace": whitespace, "unknown": unknown}


# ============================================================================
# 1ST ORDER REPAIRS (Identity Consistency)
# ============================================================================


def find_canonical_identity_values(
    df: pl.DataFrame, id_keys: List[str] = None, fix_fields: List[str] = None
) -> pl.DataFrame:
    """
    Find canonical identity values for each person.

    Args:
        df: Input dataframe
        id_keys: Keys to identify unique persons (default: ["personName"])
        fix_fields: Fields to fix (default: ["lastName", "birthDate", "gender"])

    Returns:
        Dataframe with canonical values
    """
    if id_keys is None:
        id_keys = ["personName"]
    if fix_fields is None:
        fix_fields = ["lastName", "birthDate", "gender"]

    print(f"🔍 1st order: Finding canonical values for {', '.join(fix_fields)}")

    # Clean empty strings
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

    # Get unique identities
    unique_ids = df_clean.select(id_keys).unique()
    canonical = []

    for id_row in unique_ids.iter_rows(named=True):
        # Build filter for this person
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
            if field in df.columns:
                non_null = person_data.filter(pl.col(field).is_not_null())
                rec[field] = non_null[field][0] if len(non_null) > 0 else None

        canonical.append(rec)

    # Convert to dataframe with matching types
    canonical_df = pl.DataFrame(canonical)
    for col in id_keys + fix_fields:
        if col in df.columns and col in canonical_df.columns:
            canonical_df = canonical_df.with_columns(pl.col(col).cast(df.schema[col]))

    print(f"   ✓ Found canonical values for {len(canonical_df):,} identities")
    return canonical_df


def apply_identity_fixes(
    df: pl.DataFrame,
    canonical_df: pl.DataFrame,
    id_keys: List[str] = None,
    fix_fields: List[str] = None,
) -> pl.DataFrame:
    """
    Apply canonical identity values to dataframe.

    Args:
        df: Input dataframe
        canonical_df: Canonical values dataframe
        id_keys: Keys to join on
        fix_fields: Fields to fix

    Returns:
        Fixed dataframe
    """
    if id_keys is None:
        id_keys = ["personName"]
    if fix_fields is None:
        fix_fields = ["lastName", "birthDate", "gender"]

    print(f"🔧 1st order: Applying identity fixes")

    # Clean empty strings in original
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

    # Prepare canonical for join
    rename_dict = {
        field: f"new_{field}" for field in fix_fields if field in canonical_df.columns
    }
    canonical_join = canonical_df.select(id_keys + list(rename_dict.keys())).rename(
        rename_dict
    )

    # Join and replace
    fixed = df_clean.join(canonical_join, on=id_keys, how="left")

    # Replace with canonical values
    for field in fix_fields:
        if f"new_{field}" in fixed.columns:
            fixed = fixed.with_columns(pl.col(f"new_{field}").alias(field))
            fixed = fixed.drop(f"new_{field}")

    print(f"   ✓ Applied identity fixes")
    return fixed


def repair_identity_consistency(
    df: pl.DataFrame,
    id_keys: List[str] = None,
    fix_fields: List[str] = None,
    people_filter: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Complete identity consistency repair pipeline.

    Args:
        df: Input dataframe
        id_keys: Keys to identify unique persons
        fix_fields: Fields to fix
        people_filter: Optional list of people to focus repair on (for optimization)

    Returns:
        Repaired dataframe
    """
    if people_filter:
        print(f"🎯 1st order: Focusing on {len(people_filter)} people")
        # Get data for these people plus some historical context
        relevant_data = df.filter(pl.col("personName").is_in(people_filter))
        other_data = df.filter(~pl.col("personName").is_in(people_filter))
    else:
        relevant_data = df
        other_data = None

    # Find canonical values (using relevant data)
    canonical = find_canonical_identity_values(relevant_data, id_keys, fix_fields)

    # Apply fixes
    if other_data is not None:
        # Apply fixes only to relevant data, then recombine
        fixed_relevant = apply_identity_fixes(
            relevant_data, canonical, id_keys, fix_fields
        )
        result = pl.concat([other_data, fixed_relevant], how="vertical_relaxed")
    else:
        result = apply_identity_fixes(df, canonical, id_keys, fix_fields)

    return result


# ============================================================================
# 2ND ORDER REPAIRS (Forward/Backward Fill)
# ============================================================================


def clean_second_order_empty_strings(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[str]]:
    """Convert empty strings to nulls for second order fields"""
    fields = ["countryOfCitizenship", "city", "state", "source", "industries"]
    existing = [f for f in fields if f in df.columns]

    if not existing:
        return df, []

    print(f"🧹 2nd order: Converting empty strings to nulls")

    df_clean = df.with_columns(
        [
            pl.when(pl.col(f) == "").then(None).otherwise(pl.col(f)).alias(f)
            for f in existing
        ]
    )

    return df_clean, existing


def apply_forward_backward_fill(df: pl.DataFrame, field: str) -> pl.DataFrame:
    """Apply forward/backward fill to a specific field"""
    # Forward fill
    df_fwd = df.with_columns(
        pl.col(field)
        .fill_null(strategy="forward")
        .over("personName")
        .alias(f"{field}_tmp")
    )

    # Backward fill remaining
    df_filled = df_fwd.with_columns(
        pl.col(f"{field}_tmp")
        .fill_null(strategy="backward")
        .over("personName")
        .alias(field)
    ).drop(f"{field}_tmp")

    return df_filled


def repair_second_order_fields(
    df: pl.DataFrame,
    fields: List[str] = None,
    people_filter: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Repair second order fields using forward/backward fill.

    Args:
        df: Input dataframe
        fields: Fields to repair (auto-detected if None)
        people_filter: Optional list of people to focus repair on

    Returns:
        Repaired dataframe
    """
    print(f"🔧 2nd order: Forward/backward fill repair")

    # Clean empty strings and get fields
    if people_filter:
        print(f"🎯 2nd order: Focusing on {len(people_filter)} people")
        relevant_data = df.filter(pl.col("personName").is_in(people_filter))
        other_data = df.filter(~pl.col("personName").is_in(people_filter))
        df_clean, repair_fields = clean_second_order_empty_strings(relevant_data)
    else:
        df_clean, repair_fields = clean_second_order_empty_strings(df)
        other_data = None

    if fields:
        repair_fields = [f for f in fields if f in repair_fields]

    if not repair_fields:
        print("   ⚠️ No fields to repair")
        return df

    # Sort by person and date
    df_sorted = df_clean.sort(["personName", "date"])

    # Apply fill to each field
    repaired = df_sorted
    for field in repair_fields:
        repaired = apply_forward_backward_fill(repaired, field)

    # Combine with other data if we filtered
    if other_data is not None:
        result = pl.concat([other_data, repaired], how="vertical_relaxed")
    else:
        result = repaired

    print(f"   ✓ Applied fill to {len(repair_fields)} fields")
    return result


# ============================================================================
# INTEGRATED REPAIR PIPELINE
# ============================================================================


def repair_all_orders(
    df: pl.DataFrame,
    dataset_type: str = "billionaires",
    people_filter: Optional[List[str]] = None,
    apply_0th: bool = True,
    apply_1st: bool = True,
    apply_2nd: bool = True,
) -> pl.DataFrame:
    """
    Apply all repair orders to dataframe.

    Args:
        df: Input dataframe
        dataset_type: Type of dataset for logging
        people_filter: Optional list of people to focus 1st/2nd order repairs on
        apply_0th: Whether to apply 0th order repairs
        apply_1st: Whether to apply 1st order repairs
        apply_2nd: Whether to apply 2nd order repairs

    Returns:
        Fully repaired dataframe
    """
    print(f"\n🔧 INTEGRATED REPAIR PIPELINE - {dataset_type}")
    print("=" * 60)

    result = df

    # 0th Order: Clean whitespace and unknowns (always apply to all data)
    if apply_0th:
        result = clean_whitespace_and_unknowns(result, dataset_type)

    # 1st Order: Identity consistency (can be optimized with people_filter)
    if apply_1st and dataset_type == "billionaires":
        result = repair_identity_consistency(result, people_filter=people_filter)

    # 2nd Order: Forward/backward fill (can be optimized with people_filter)
    if apply_2nd and dataset_type == "billionaires":
        result = repair_second_order_fields(result, people_filter=people_filter)

    print(f"✅ Repair pipeline completed for {len(result):,} records")
    return result


def get_people_in_new_data(new_df: pl.DataFrame) -> List[str]:
    """Get list of people who appear in new data for optimization"""
    if "personName" not in new_df.columns:
        return []

    people = new_df.select("personName").unique().to_series().to_list()
    people = [p for p in people if p is not None]
    return people


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def analyze_repair_impact(
    original: pl.DataFrame, repaired: pl.DataFrame, repair_type: str = "unknown"
) -> Dict:
    """Analyze the impact of repairs"""
    print(f"\n📊 {repair_type.upper()} REPAIR IMPACT")
    print("=" * 40)

    # Basic stats
    stats = {
        "original_rows": len(original),
        "repaired_rows": len(repaired),
        "repair_type": repair_type,
    }

    # Field-specific analysis depends on repair type
    if repair_type == "0th_order":
        original_issues = count_0th_order_issues(original)
        repaired_issues = count_0th_order_issues(repaired)

        stats.update(
            {
                "whitespace_fixed": original_issues["whitespace"]
                - repaired_issues["whitespace"],
                "unknown_fixed": original_issues["unknown"]
                - repaired_issues["unknown"],
            }
        )

        print(
            f"Whitespace issues: {original_issues['whitespace']} → {repaired_issues['whitespace']}"
        )
        print(
            f"Unknown variations: {original_issues['unknown']} → {repaired_issues['unknown']}"
        )

    return stats

"""
Automated Data Cleaning & Preparation Pipeline.

Handles missing values, data harmonization, type casting,
and deduplication for uploaded CSV datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# Common alias maps for retail data harmonization
KNOWN_ALIASES = {
    "lf": "Low Fat",
    "low fat": "Low Fat",
    "lowfat": "Low Fat",
    "reg": "Regular",
    "regular": "Regular",
}


def _harmonize_categorical(series: pd.Series) -> pd.Series:
    """Normalize inconsistent categorical values using alias mapping."""
    if series.dtype != "object":
        return series

    def normalize(val):
        if pd.isna(val):
            return val
        stripped = str(val).strip()
        lookup = stripped.lower()
        return KNOWN_ALIASES.get(lookup, stripped)

    return series.apply(normalize)


def _try_parse_dates(series: pd.Series) -> pd.Series:
    """Attempt to parse a string column as datetime."""
    if series.dtype != "object":
        return series

    sample = series.dropna().head(50)
    if len(sample) == 0:
        return series

    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True)
        # If >80% parsed successfully, convert the whole column
        if parsed.notna().sum() / len(sample) > 0.8:
            return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    except (ValueError, TypeError):
        pass

    return series


def _try_parse_numeric(series: pd.Series) -> pd.Series:
    """Attempt to parse a string column as numeric."""
    if series.dtype != "object":
        return series

    sample = series.dropna().head(50)
    if len(sample) == 0:
        return series

    try:
        parsed = pd.to_numeric(sample, errors="coerce")
        if parsed.notna().sum() / len(sample) > 0.8:
            return pd.to_numeric(series, errors="coerce")
    except (ValueError, TypeError):
        pass

    return series


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Run the full automated cleaning pipeline on a dataframe.

    Returns:
        (cleaned_df, list of cleaning actions performed)
    """
    df = df.copy()
    actions = []

    # --- Step 1: Drop exact duplicate rows ---
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        actions.append(f"🗑️ Removed {dup_count} duplicate rows")

    # --- Step 2: Type casting (dates & numerics) ---
    for col in df.columns:
        original_dtype = df[col].dtype

        # Try dates first
        converted = _try_parse_dates(df[col])
        if converted.dtype != original_dtype and pd.api.types.is_datetime64_any_dtype(converted):
            df[col] = converted
            actions.append(f"📅 Converted '{col}' to datetime")
            continue

        # Try numeric
        converted = _try_parse_numeric(df[col])
        if converted.dtype != original_dtype and pd.api.types.is_numeric_dtype(converted):
            df[col] = converted
            actions.append(f"🔢 Converted '{col}' to numeric")

    # --- Step 3: Data harmonization for categorical columns ---
    for col in df.select_dtypes(include=["object"]).columns:
        original_unique = df[col].nunique()
        df[col] = _harmonize_categorical(df[col])
        new_unique = df[col].nunique()
        if new_unique < original_unique:
            actions.append(
                f"🔄 Harmonized '{col}': {original_unique} → {new_unique} unique values"
            )

    # --- Step 4: Missing value imputation ---
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            actions.append(
                f"📊 Filled {null_count} missing values in '{col}' with median ({median_val:.2f})"
            )
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Leave datetime NaTs as-is — filling with a fake date is misleading
            actions.append(f"⏳ '{col}' has {null_count} missing dates (left as NaT)")
        else:
            df[col] = df[col].fillna("Unknown")
            actions.append(
                f"📝 Filled {null_count} missing values in '{col}' with 'Unknown'"
            )

    if not actions:
        actions.append("✅ Dataset is already clean — no changes needed")

    return df, actions


def get_schema_summary(df: pd.DataFrame) -> str:
    """
    Generate a structured schema summary string for the LLM agent.

    Includes column names, dtypes, non-null counts, unique values,
    and sample values — everything the LLM needs to write correct code.
    """
    lines = []
    lines.append(f"Dataset: {len(df)} rows × {len(df.columns)} columns\n")
    lines.append("| Column | Type | Non-Null | Unique | Sample Values |")
    lines.append("|--------|------|----------|--------|---------------|")

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        unique = df[col].nunique()

        # Get representative sample values
        sample_vals = df[col].dropna().unique()[:5]
        if pd.api.types.is_numeric_dtype(df[col]):
            sample_str = ", ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in sample_vals)
        else:
            sample_str = ", ".join(str(v) for v in sample_vals)

        if len(sample_str) > 60:
            sample_str = sample_str[:57] + "..."

        lines.append(f"| {col} | {dtype} | {non_null} | {unique} | {sample_str} |")

    # Add basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        lines.append(f"\nNumeric columns: {', '.join(numeric_cols)}")
        stats = df[numeric_cols].describe().round(2)
        lines.append(f"\nBasic statistics:\n{stats.to_string()}")

    # Add date range info for datetime columns
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if date_cols:
        lines.append(f"\nDate columns: {', '.join(date_cols)}")
        for col in date_cols:
            min_date = df[col].min()
            max_date = df[col].max()
            lines.append(f"  {col}: {min_date} to {max_date}")

    return "\n".join(lines)

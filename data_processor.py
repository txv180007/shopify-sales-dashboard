import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from config import NUMERIC_COLUMNS, STRING_COLUMNS, DATE_COLUMN_CANDIDATES


def coerce_num(value: Any, is_percentage: bool = False) -> float:
    """Parse currency, percentage, and numeric values with proper handling of brackets and symbols."""
    if pd.isna(value) or value == "":
        return pd.NA

    if isinstance(value, (int, float)):
        return float(value)

    string_val = str(value).strip()
    if not string_val:
        return pd.NA

    # Handle parentheses (negative values)
    is_negative = string_val.startswith("(") and string_val.endswith(")")
    if is_negative:
        string_val = string_val[1:-1].strip()

    # Check for percentage symbol
    has_percent = string_val.endswith("%")

    # Remove all non-numeric characters except decimal point and minus sign
    cleaned = re.sub(r"[^0-9.\-]", "", string_val)

    if cleaned == "":
        return pd.NA

    try:
        numeric_value = float(cleaned)
    except ValueError:
        return pd.NA

    if is_negative:
        numeric_value = -numeric_value

    if is_percentage and has_percent:
        numeric_value = numeric_value / 100.0

    return numeric_value


def process_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Process string columns to ensure consistent formatting."""
    df_processed = df.copy()

    for col in STRING_COLUMNS:
        if col in df_processed.columns:
            df_processed[col] = (df_processed[col]
                               .astype(str)
                               .replace({"nan": "", "None": ""})
                               .str.strip())

    return df_processed


def process_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Process numeric columns with appropriate percentage handling."""
    df_processed = df.copy()

    for col, is_percentage in NUMERIC_COLUMNS.items():
        if col in df_processed.columns and not pd.api.types.is_numeric_dtype(df_processed[col]):
            df_processed[col] = pd.to_numeric(
                df_processed[col].apply(lambda v: coerce_num(v, is_percentage)),
                errors="coerce"
            )

    return df_processed


def process_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Process date columns and create standardized date fields."""
    df_processed = df.copy()

    # Find the date column
    date_source = None
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in df_processed.columns:
            date_source = candidate
            break

    if date_source:
        df_processed["dateYMD"] = pd.to_datetime(df_processed[date_source], errors="coerce")
        df_processed["date_display"] = df_processed["dateYMD"].dt.strftime("%b %d, %Y")
    else:
        df_processed["dateYMD"] = pd.NaT
        df_processed["date_display"] = ""

    return df_processed


def process_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Process boolean columns."""
    df_processed = df.copy()

    if "hasImage" in df_processed.columns:
        df_processed["hasImage"] = (df_processed["hasImage"]
                                   .astype(str)
                                   .str.lower()
                                   .isin(["true", "1", "yes", "y"]))

    return df_processed


def coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to process all column types."""
    if df.empty:
        return df

    # Process each column type
    df = process_string_columns(df)
    df = process_numeric_columns(df)
    df = process_date_column(df)
    df = process_boolean_columns(df)

    return df


def contains_any(series: pd.Series, needles: List[str]) -> pd.Series:
    """Check if series contains any of the needle strings (case-insensitive)."""
    series_lower = series.astype(str).str.lower().fillna("")

    if not needles:
        return pd.Series(False, index=series.index)

    condition = pd.Series(False, index=series.index)
    for needle in needles:
        needle_clean = str(needle).strip().lower()
        if needle_clean:
            condition |= series_lower.str.contains(re.escape(needle_clean), na=False)

    return condition


def parse_search_terms(text: str) -> List[str]:
    """Parse comma or whitespace-separated search terms."""
    if not text:
        return []

    # Split by comma first, then by whitespace
    terms = []
    for chunk in str(text).split(","):
        terms.extend(chunk.strip().split())

    return [term.strip() for term in terms if term.strip()]


def aggregate_by_title(df: pd.DataFrame, agg_columns: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """Generic aggregation function by title."""
    if df.empty:
        return pd.DataFrame()

    return df.groupby("title", as_index=False).agg(**agg_columns)


def get_latest_non_null_by_title(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Get the latest non-null value per title within the filtered window."""
    if value_column not in df.columns or df.empty:
        return pd.DataFrame(columns=["title", value_column])

    def get_last_non_null(group_data):
        sorted_data = group_data.sort_values("dateYMD")[value_column]
        non_null_data = sorted_data.dropna()
        return non_null_data.iloc[-1] if len(non_null_data) > 0 else pd.NA

    result = (df.groupby("title", group_keys=False)
              .apply(get_last_non_null)
              .rename(value_column)
              .reset_index())

    return result


def get_most_frequent_non_empty(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Get the most frequent non-empty string per title."""
    if column not in df.columns or df.empty:
        return pd.DataFrame(columns=["title", column])

    def get_mode_value(group_data):
        series = (group_data[column]
                 .astype(str)
                 .replace({"nan": "", "None": ""})
                 .str.strip())

        non_empty = series[series != ""]
        if non_empty.empty:
            return ""

        return non_empty.value_counts().idxmax()

    result = (df.groupby("title", group_keys=False)
              .apply(get_mode_value)
              .rename(column)
              .reset_index())

    return result
"""
Utility functions for the dashboard
"""
import pandas as pd
import json
import streamlit as st
from datetime import datetime, timedelta
from typing import Tuple, Optional
from config import DATE_RANGE_OPTIONS, DEFAULT_DATE_RANGE


def get_date_range_from_option(option: str, data_min: pd.Timestamp, data_max: pd.Timestamp) -> Tuple[datetime, datetime]:
    """Calculate date range based on quick filter option."""
    today = datetime.now().date()

    if option == "Custom range":
        # Return data bounds for custom range
        return data_min.date() if pd.notna(data_min) else today, data_max.date() if pd.notna(data_max) else today

    days_back = DATE_RANGE_OPTIONS.get(option, 7)

    if days_back == 0:  # Today
        return today, today
    elif days_back == 1:  # Yesterday
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    else:  # Last N days
        start_date = today - timedelta(days=days_back - 1)  # -1 to include today
        return start_date, today


def create_download_buttons(df: pd.DataFrame, filename_prefix: str, key_prefix: str):
    """Create download buttons for CSV and JSON export."""
    if df.empty:
        st.info("No data to export")
        return

    col1, col2, col3 = st.columns(3)

    # CSV download
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv"
        )

    # JSON download
    with col2:
        json_data = df.to_json(orient='records', indent=2).encode('utf-8')
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"{key_prefix}_json"
        )

    # Copy to clipboard button
    with col3:
        if st.button("📋 Copy to Clipboard", key=f"{key_prefix}_copy"):
            # Create a text version that can be copied
            csv_text = df.to_csv(index=False)
            st.code(csv_text, language=None)
            st.success("Table copied as CSV format above. Select and copy the text.")


def format_dataframe_for_display(df: pd.DataFrame, numeric_columns: dict = None) -> pd.DataFrame:
    """Format dataframe for better display with proper number formatting."""
    if df.empty:
        return df

    display_df = df.copy()

    # Default formatting rules if not provided
    if numeric_columns is None:
        numeric_columns = {}

    # Apply formatting to numeric columns
    for col in display_df.columns:
        if col in numeric_columns:
            format_type = numeric_columns[col]
            if format_type == "currency":
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            elif format_type == "percentage":
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            elif format_type == "integer":
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            elif format_type == "decimal":
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

    return display_df


def create_exportable_table_section(title: str, df: pd.DataFrame, filename_prefix: str,
                                  key_prefix: str, show_download: bool = True):
    """Create a table section with export capabilities."""
    st.markdown(f"### {title}")

    if df.empty:
        st.info("No data to display")
        return

    # Display the table
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Add export buttons if requested
    if show_download:
        st.markdown("**Export Options:**")
        create_download_buttons(df, filename_prefix, key_prefix)
        st.divider()


def get_table_height(num_rows: int, min_height: int = 300, max_height: int = 600,
                    row_height: int = 35) -> int:
    """Calculate optimal table height based on number of rows."""
    calculated_height = min_height + (num_rows * row_height)
    return min(max(calculated_height, min_height), max_height)
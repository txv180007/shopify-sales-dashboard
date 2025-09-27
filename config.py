# Configuration constants
import streamlit as st

# App Configuration
APP_TITLE = "Daily Products Dashboard"
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "collapsed"

# Google Sheets Configuration
def get_spreadsheet_config():
    return {
        'spreadsheet_id': st.secrets["SPREADSHEET_ID"],
        'worksheet_name': st.secrets.get("WORKSHEET_NAME", "Daily Products"),
        'presets_sheet_name': st.secrets.get("PRESETS_SHEET_NAME", "Presets"),
        'scopes': [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
    }

# Data Processing Configuration
NUMERIC_COLUMNS = {
    "ordersCount": False,
    "qtySold": False,
    "grossSales": False,
    "netSales": False,
    "avgUnitNet": False,
    "avgUnitGross": False,
    "discountRate": True,   # percentage
    "inventoryNow": False,
    "unitCost": False,
    "totalCost": False,
    "grossProfit": False,
    "grossMarginRate": True,  # percentage
}

STRING_COLUMNS = ["sku", "barcode", "barcodesDistinct", "title", "productHandle", "variantId"]

DATE_COLUMN_CANDIDATES = ["dateYMD", "Date", "date", "runDateYMD"]

# UI Configuration
CHART_STYLES = ["Line", "Bar"]
HAS_IMAGE_OPTIONS = ["Both", "Has image", "No image"]
PRESET_COMBINE_MODES = ["OR", "AND"]

# Date Range Quick Filters
DATE_RANGE_OPTIONS = {
    "Today": 0,
    "Yesterday": 1,
    "Last 7 days": 7,
    "Last 14 days": 14,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Custom range": None
}

DEFAULT_DATE_RANGE = "Last 7 days"

# Stock Analysis Thresholds
STOCK_THRESHOLDS = {
    "high_price": {"min_price": 30, "threshold": 2},
    "medium_price": {"min_price": 20, "max_price": 30, "threshold": 3},
    "low_price": {"max_price": 20, "threshold": 5}
}

# Stale Inventory Configuration
STALE_INVENTORY_DEFAULTS = {
    "window_days": 30,
    "no_sale_days": 30,
    "days_of_cover_min": 60
}

# Cache Configuration
CACHE_TTL = {
    "data": 300,  # 5 minutes
    "presets": 60  # 1 minute
}

# Timezone
TIMEZONE = "America/Chicago"
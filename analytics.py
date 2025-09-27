import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from data_processor import (
    get_latest_non_null_by_title,
    get_most_frequent_non_empty,
    aggregate_by_title
)
from config import TIMEZONE, STOCK_THRESHOLDS


class StockAnalyzer:
    """Analyzes stock levels and generates stock-related insights."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_shopping_list(self, exclude_low_velocity: bool = False,
                         price_threshold: float = 10.0,
                         sold_percent_threshold: float = 10.0) -> pd.DataFrame:
        """Generate a shopping list based on quantity sold and inventory."""
        if self.df.empty:
            return pd.DataFrame()

        # Get aggregated data
        qty_by_title = aggregate_by_title(self.df, {"qtySold": ("qtySold", "sum")})
        inv_latest = get_latest_non_null_by_title(self.df, "inventoryNow").rename(
            columns={"inventoryNow": "Inventory Left"}
        )
        price_latest = get_latest_non_null_by_title(self.df, "avgUnitNet").rename(
            columns={"avgUnitNet": "Item Price"}
        )

        # Fallback to gross price if net price is missing
        if price_latest["Item Price"].isna().any():
            gross_latest = get_latest_non_null_by_title(self.df, "avgUnitGross").rename(
                columns={"avgUnitGross": "Item Price (Gross)"}
            )
            price_latest = price_latest.merge(gross_latest, on="title", how="left")
            price_latest["Item Price"] = price_latest["Item Price"].fillna(
                price_latest["Item Price (Gross)"]
            )
            price_latest = price_latest[["title", "Item Price"]]

        # Get additional info
        barcode = get_most_frequent_non_empty(self.df, "barcode").rename(
            columns={"barcode": "Barcode"}
        )
        sku = get_most_frequent_non_empty(self.df, "sku").rename(columns={"sku": "SKU"})

        # Merge all data
        result = (qty_by_title
                  .merge(inv_latest, on="title", how="left")
                  .merge(price_latest, on="title", how="left")
                  .merge(barcode, on="title", how="left")
                  .merge(sku, on="title", how="left")
                  .rename(columns={"title": "Product", "qtySold": "Qty Sold"}))

        # Calculate sell-through percentage
        denominator = result["Qty Sold"].fillna(0) + result["Inventory Left"].fillna(0)
        result["Sold %"] = (result["Qty Sold"].fillna(0) / denominator.where(denominator > 0, 1)) * 100

        # Apply low-velocity filter if requested
        if exclude_low_velocity:
            keep_mask = ~((result["Item Price"].fillna(0) < price_threshold) &
                         (result["Sold %"] < sold_percent_threshold))
            result = result[keep_mask]

        return result.sort_values(["Qty Sold", "Item Price"], ascending=[False, True])

    def get_out_of_stock_items(self) -> pd.DataFrame:
        """Get items that are out of stock."""
        shopping_list = self.get_shopping_list()
        return shopping_list[shopping_list["Inventory Left"].fillna(0) <= 0].copy()

    def get_low_stock_items(self) -> pd.DataFrame:
        """Get items with low stock using tiered thresholds based on price."""
        shopping_list = self.get_shopping_list()

        def get_threshold_for_price(price: float) -> int:
            """Get stock threshold based on item price."""
            if pd.isna(price):
                return STOCK_THRESHOLDS["medium_price"]["threshold"]

            if price > STOCK_THRESHOLDS["high_price"]["min_price"]:
                return STOCK_THRESHOLDS["high_price"]["threshold"]
            elif (STOCK_THRESHOLDS["medium_price"]["min_price"] <= price <=
                  STOCK_THRESHOLDS["medium_price"]["max_price"]):
                return STOCK_THRESHOLDS["medium_price"]["threshold"]
            else:
                return STOCK_THRESHOLDS["low_price"]["threshold"]

        thresholds = shopping_list["Item Price"].apply(get_threshold_for_price)
        low_stock_mask = ((shopping_list["Inventory Left"].fillna(0) > 0) &
                         (shopping_list["Inventory Left"] <= thresholds))

        low_stock = shopping_list[low_stock_mask].copy()
        low_stock = low_stock.assign(Threshold=thresholds[low_stock.index].values)

        return low_stock.sort_values(["Inventory Left", "Item Price"], ascending=[True, False])

    def get_missing_cogs_items(self) -> pd.DataFrame:
        """Get items with missing or zero COGS."""
        if self.df.empty:
            return pd.DataFrame()

        cogs_agg = aggregate_by_title(
            self.df,
            {
                "Total_COGS": ("totalCost", "sum"),
                "Revenue": ("netSales", "sum")
            }
        )

        missing_cogs = cogs_agg[cogs_agg["Total_COGS"].fillna(0) == 0].copy()

        if missing_cogs.empty:
            return pd.DataFrame()

        # Add additional info
        shopping_list = self.get_shopping_list()
        base_data = shopping_list.rename(columns={"Product": "title"})

        result = (missing_cogs.merge(base_data, on="title", how="left")
                              .rename(columns={"title": "Product"}))

        columns_order = [
            "Product", "Qty Sold", "Inventory Left", "Item Price",
            "Barcode", "SKU", "Revenue", "Total_COGS"
        ]

        return result[columns_order].sort_values(["Revenue", "Product"], ascending=[False, True])


class StockOutAnalyzer:
    """Analyzes stock-out projections."""

    def __init__(self, df: pd.DataFrame, timezone: str = TIMEZONE):
        self.df = df
        self.timezone = timezone

    def build_stockout_table(self) -> pd.DataFrame:
        """Build projected stock-out table."""
        if self.df.empty:
            return pd.DataFrame(columns=[
                "title", "Inventory Left", "Avg Daily Sold",
                "Days to Stock-Out", "Projected Stock-Out Date"
            ])

        # Calculate window span in days
        date_min = pd.to_datetime(self.df["dateYMD"].min())
        date_max = pd.to_datetime(self.df["dateYMD"].max())

        if pd.isna(date_min) or pd.isna(date_max):
            days = 1
        else:
            days = max((date_max.normalize() - date_min.normalize()).days + 1, 1)

        # Get latest inventory per title
        inv_latest = get_latest_non_null_by_title(self.df, "inventoryNow").rename(
            columns={"inventoryNow": "Inventory Left"}
        )

        # Calculate average daily sold
        sold_total = aggregate_by_title(self.df, {"_sold_total": ("qtySold", "sum")})
        sold_total["Avg Daily Sold"] = sold_total["_sold_total"] / float(days)

        # Merge data
        result = inv_latest.merge(
            sold_total[["title", "Avg Daily Sold"]],
            on="title",
            how="outer"
        )

        result["Inventory Left"] = pd.to_numeric(result["Inventory Left"], errors="coerce")
        result["Avg Daily Sold"] = pd.to_numeric(result["Avg Daily Sold"], errors="coerce").fillna(0.0)

        # Calculate days to stock-out
        def calculate_days_to_stockout(inventory: float, avg_daily: float) -> float:
            if pd.isna(inventory):
                return np.inf
            if avg_daily and avg_daily > 0:
                return float(inventory) / float(avg_daily)
            return np.inf

        result["Days to Stock-Out"] = result.apply(
            lambda row: calculate_days_to_stockout(
                row["Inventory Left"], row["Avg Daily Sold"]
            ),
            axis=1
        )

        # Calculate projected date
        now_tz = pd.Timestamp.now(tz=self.timezone).normalize()

        def calculate_projected_date(days_value: float) -> str:
            if pd.isna(days_value) or not np.isfinite(days_value):
                return ""
            return (now_tz + pd.Timedelta(days=int(np.ceil(days_value)))).date().isoformat()

        result["Projected Stock-Out Date"] = result["Days to Stock-Out"].apply(
            calculate_projected_date
        )

        return result


class StaleInventoryAnalyzer:
    """Analyzes stale inventory metrics."""

    def __init__(self, df: pd.DataFrame, timezone: str = TIMEZONE):
        self.df = df
        self.timezone = timezone

    def compute_stale_metrics(self, key: str = "title", window_days: int = 30,
                            stale_days_since_last: int = 30,
                            stale_doc_min: float = 60) -> pd.DataFrame:
        """Compute comprehensive stale inventory metrics."""
        if self.df.empty:
            return pd.DataFrame(columns=[
                key, "Inventory Left", f"Sold_{window_days}",
                f"Velocity_{window_days}", f"DaysToStockOut_{window_days}",
                "LastSaleDate", "DaysSinceLastSale", "FirstSeenDate",
                "AgeDays", "StaleFlag"
            ])

        data = self.df.copy()
        data["dateYMD"] = pd.to_datetime(data["dateYMD"], errors="coerce")
        data["qtySold"] = pd.to_numeric(data.get("qtySold", 0), errors="coerce").fillna(0)
        data["inventoryNow"] = pd.to_numeric(data.get("inventoryNow", pd.NA), errors="coerce")

        today = pd.Timestamp.now(tz=self.timezone).normalize().tz_localize(None)
        start_window = today - pd.Timedelta(days=window_days - 1)

        # Latest inventory
        inventory_data = (
            data.sort_values([key, "dateYMD"])
            .groupby(key, as_index=False)
            .agg(**{"Inventory Left": ("inventoryNow", lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else pd.NA)})
        )

        # Sales in window
        window_data = data[data["dateYMD"] >= start_window]
        sold_data = (
            window_data.groupby(key, as_index=False)["qtySold"]
            .sum()
            .rename(columns={"qtySold": f"Sold_{window_days}"})
        )

        # Ensure all keys represented
        sold_data = inventory_data[[key]].merge(sold_data, on=key, how="left").fillna({f"Sold_{window_days}": 0})
        sold_data[f"Velocity_{window_days}"] = sold_data[f"Sold_{window_days}"] / float(window_days)

        # Merge and calculate days to stock-out
        merged = sold_data.merge(inventory_data, on=key, how="left")

        def calculate_days_to_stockout(inventory: float, velocity: float) -> float:
            if pd.isna(inventory) or velocity is None or velocity <= 0:
                return np.inf
            return float(inventory) / float(max(velocity, 1e-9))

        merged[f"DaysToStockOut_{window_days}"] = merged.apply(
            lambda row: calculate_days_to_stockout(
                row.get("Inventory Left"), row.get(f"Velocity_{window_days}")
            ),
            axis=1
        )

        # Last sale date
        last_sale = (
            data[data["qtySold"] > 0]
            .groupby(key, as_index=False)["dateYMD"]
            .max()
            .rename(columns={"dateYMD": "LastSaleDate"})
        )
        last_sale["DaysSinceLastSale"] = (today - pd.to_datetime(last_sale["LastSaleDate"])).dt.days

        # First seen date
        first_seen = (
            data.groupby(key, as_index=False)["dateYMD"]
            .min()
            .rename(columns={"dateYMD": "FirstSeenDate"})
        )
        first_seen["AgeDays"] = (today - pd.to_datetime(first_seen["FirstSeenDate"])).dt.days

        # Final merge
        result = (
            merged.merge(last_sale[[key, "LastSaleDate", "DaysSinceLastSale"]], on=key, how="left")
            .merge(first_seen[[key, "FirstSeenDate", "AgeDays"]], on=key, how="left")
        )

        # Stale flag
        doc_col = f"DaysToStockOut_{window_days}"
        result["StaleFlag"] = (
            (result["DaysSinceLastSale"].fillna(10**9) >= stale_days_since_last) |
            (pd.to_numeric(result[doc_col], errors="coerce").fillna(np.inf) >= stale_doc_min)
        )

        columns = [
            key, "Inventory Left",
            f"Sold_{window_days}", f"Velocity_{window_days}", doc_col,
            "LastSaleDate", "DaysSinceLastSale", "FirstSeenDate", "AgeDays", "StaleFlag"
        ]

        return result[columns]
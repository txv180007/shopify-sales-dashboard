"""
Shopify Sales Dashboard - Refactored Main Application
"""

# Standard library
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Third-party
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit import column_config as cc

# Local modules
from config import APP_TITLE, PAGE_LAYOUT, INITIAL_SIDEBAR_STATE, CHART_STYLES
from utils import create_download_buttons, create_exportable_table_section, get_table_height
from auth import AuthManager
from data_loader import load_main_data, PresetManager
from data_processor import (
    parse_search_terms, aggregate_by_title,
    get_latest_non_null_by_title, get_most_frequent_non_empty
)
from analytics import StockAnalyzer, StockOutAnalyzer, StaleInventoryAnalyzer
from ui_components import FilterManager, ChartRenderer


def setup_page():
    """Configure Streamlit page settings and CSS."""
    st.set_page_config(
        page_title=APP_TITLE,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE,
    )

    # Sticky tabs styling
    st.markdown("""
    <style>
      :root {
        --tabs-stick-top: 0px;
        --tabs-height: 46px;
      }

      section.main { position: relative; }

      section.main div[data-testid="stTabs"] div[role="tablist"],
      section.main .stTabs div[role="tablist"] {
        position: sticky;
        top: var(--tabs-stick-top);
        z-index: 1000;
        background-color: inherit;
        padding: .25rem 0 .35rem 0;
        border-bottom: 1px solid rgba(128,128,128,.2);
      }

      section.main div[data-testid="stTabs"] [role="tab"],
      section.main .stTabs [role="tab"] { overflow: visible; }

      .sticky-chipbar {
        position: sticky;
        top: calc(var(--tabs-stick-top) + var(--tabs-height));
        z-index: 999;
        background-color: inherit;
        padding: .25rem 0 .4rem 0;
        margin-bottom: .25rem;
        border-bottom: 1px solid rgba(128,128,128,.15);
        overflow-x: auto;
        white-space: nowrap;
      }
      .sticky-chipbar::-webkit-scrollbar { height: 6px; }
      .sticky-chipbar .stButton > button {
        display: inline-block;
        margin-right: .35rem;
        margin-bottom: .25rem;
        border-radius: 999px !important;
        padding: .25rem .65rem !important;
        font-size: 0.85rem !important;
        border: 1px solid rgba(0,0,0,.15) !important;
        background: rgba(0,0,0,.02) !important;
      }

      header { position: static !important; }
    </style>
    """, unsafe_allow_html=True)


def render_preset_chips():
    """Render preset chips for quick filtering."""
    preset_manager = PresetManager()
    presets_all = preset_manager.list_presets()

    # Order: favorites first (name begins with '*'), then alphabetically
    favorites = [p for p in presets_all if str(p["name"]).strip().startswith("*")]
    regular = [p for p in presets_all if p not in favorites]
    chips = (favorites + regular)[:12]

    # Initialize session state
    if "active_chips" not in st.session_state:
        st.session_state["active_chips"] = []
    if "chip_terms_map" not in st.session_state:
        st.session_state["chip_terms_map"] = {}

    def get_terms_from_preset(filter_def: dict) -> list:
        """Extract terms from preset filters."""
        title_terms = [str(x).strip() for x in (filter_def.get("title_contains", []) or []) if str(x).strip()]
        barcode_terms = [str(x).strip() for x in (filter_def.get("barcode_contains", []) or []) if str(x).strip()]
        return sorted(set(title_terms) | set(barcode_terms))

    if chips:
        st.markdown("<div class='sticky-chipbar'>", unsafe_allow_html=True)
        cols = st.columns(len(chips) + 1)

        for i, preset in enumerate(chips):
            name = preset["name"]
            filter_def = preset["filters"]
            terms_for_chip = get_terms_from_preset(filter_def)
            is_active = name in st.session_state["active_chips"]

            label = f"✓ {name}" if is_active else name
            help_text = "Click to disable" if is_active else "Click to enable"

            with cols[i]:
                if st.button(label, key=f"chip_{i}", help=help_text, use_container_width=False):
                    if is_active:
                        # Turn OFF
                        try:
                            st.session_state["active_chips"].remove(name)
                        except ValueError:
                            pass
                        st.session_state["chip_terms_map"].pop(name, None)
                    else:
                        # Turn ON
                        st.session_state["active_chips"].append(name)
                        st.session_state["chip_terms_map"][name] = terms_for_chip
                    st.rerun()

        # Clear filters button
        with cols[-1]:
            if st.button("Clear filters", key="chip_clear", help="Reset all filters", use_container_width=False):
                keys_to_clear = [
                    "contains_terms", "neg_contains_terms", "has_img_choice",
                    "pick_titles", "__preset_dates__", "__new_preset_name__",
                    "compact", "active_chips", "chip_terms_map"
                ]
                for key in keys_to_clear:
                    st.session_state.pop(key, None)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def format_date(date_val):
    """Format date for display."""
    if date_val is None or pd.isna(date_val):
        return "—"
    try:
        return pd.to_datetime(date_val).strftime("%b %d, %Y")
    except Exception:
        return "—"


def render_report_period_banner(report_period, df):
    """Render the report period banner."""
    start_date, end_date = report_period if isinstance(report_period, tuple) else (None, None)

    if (start_date is not None and end_date is not None and
        pd.notna(start_date) and pd.notna(end_date)):
        if pd.to_datetime(start_date).date() == pd.to_datetime(end_date).date():
            st.markdown(f"**Report Date:** {format_date(start_date)}")
        else:
            st.markdown(f"**Report Period:** {format_date(start_date)} → {format_date(end_date)}")
    else:
        # Fallback to dataset span
        data_min = pd.to_datetime(df['dateYMD'].min())
        data_max = pd.to_datetime(df['dateYMD'].max())
        st.markdown(f"**Report Period:** {format_date(data_min)} → {format_date(data_max)}")


def build_daily_timeseries(filtered_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Build comprehensive daily time series data."""
    if filtered_df.empty:
        start_date = pd.to_datetime(full_df["dateYMD"].min())
        end_date = pd.to_datetime(full_df["dateYMD"].max())
    else:
        start_date = pd.to_datetime(filtered_df["dateYMD"].min())
        end_date = pd.to_datetime(filtered_df["dateYMD"].max())

    if pd.isna(start_date) or pd.isna(end_date):
        base_day = pd.Timestamp.today().normalize()
        date_index = pd.date_range(start=base_day, end=base_day, freq="D")
    else:
        date_index = pd.date_range(start=start_date.normalize(), end=end_date.normalize(), freq="D")

    # All data aggregation for context
    daily_all = (
        full_df[full_df["dateYMD"].between(date_index.min(), date_index.max())]
        .groupby("dateYMD", as_index=False)
        .agg(net_total=("netSales", "sum"))
        .set_index("dateYMD")
        .reindex(date_index)
        .fillna(0.0)
        .rename_axis("dateYMD")
        .reset_index()
    )

    # Selected data aggregation
    if not filtered_df.empty:
        agg_dict = {
            "net": ("netSales", "sum"),
            "qty": ("qtySold", "sum"),
            "inv": ("inventoryNow", "sum"),
        }

        if "grossProfit" in filtered_df.columns:
            agg_dict["gp"] = ("grossProfit", "sum")
        elif "netSales" in filtered_df.columns and "totalCost" in filtered_df.columns:
            filtered_df = filtered_df.assign(
                __gp__=pd.to_numeric(filtered_df["netSales"], errors="coerce") -
                       pd.to_numeric(filtered_df["totalCost"], errors="coerce")
            )
            agg_dict["gp"] = ("__gp__", "sum")

        daily_selected = filtered_df.groupby("dateYMD", as_index=False).agg(**agg_dict)
    else:
        # Empty selection
        columns = {
            "dateYMD": date_index,
            "net": [0.0] * len(date_index),
            "qty": [0.0] * len(date_index),
            "inv": [0.0] * len(date_index),
            "gp": [0.0] * len(date_index)
        }
        daily_selected = pd.DataFrame(columns)

    # Merge data
    daily_data = pd.merge(daily_all, daily_selected, on="dateYMD", how="left")

    for col in ["net", "qty", "inv", "net_total", "gp"]:
        if col not in daily_data:
            daily_data[col] = 0.0

    daily_data[["net", "qty", "inv", "net_total", "gp"]] = daily_data[
        ["net", "qty", "inv", "net_total", "gp"]
    ].fillna(0.0)

    # Calculate percentages and margins
    daily_data["pct_of_day_revenue"] = daily_data.apply(
        lambda row: (row["net"] / row["net_total"] * 100.0) if row["net_total"] > 0 else 0.0,
        axis=1
    )

    daily_data["avg_margin_pct"] = daily_data.apply(
        lambda row: (row["gp"] / row["net"] * 100.0) if row["net"] > 0 else 0.0,
        axis=1
    )

    daily_data = daily_data.sort_values("dateYMD")
    daily_data["date_label"] = daily_data["dateYMD"].dt.strftime("%b %d, %Y")

    return daily_data


def render_dashboard_tab(filtered_df: pd.DataFrame, full_df: pd.DataFrame, compact: bool):
    """Render the main dashboard tab."""
    # KPIs
    total_net = float(filtered_df.get("netSales", pd.Series(dtype=float)).sum())
    total_cogs = float(filtered_df.get("totalCost", pd.Series(dtype=float)).sum()) if "totalCost" in filtered_df.columns else 0.0

    if "grossProfit" in filtered_df.columns:
        total_gp = float(filtered_df["grossProfit"].sum())
    else:
        total_gp = total_net - total_cogs

    overall_margin = (total_gp / total_net) if total_net > 0 else 0.0

    if not compact:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Revenue (Net)", f"${total_net:,.2f}")
        with col2: st.metric("COGS", f"${total_cogs:,.2f}")
        with col3: st.metric("Gross Profit", f"${total_gp:,.2f}")
        with col4: st.metric("Gross Margin", f"{overall_margin:.2%}")
    else:
        # Mobile layout
        r1c1, r1c2 = st.columns(2)
        with r1c1: st.metric("Revenue (Net)", f"${total_net:,.2f}")
        with r1c2: st.metric("COGS", f"${total_cogs:,.2f}")
        r2c1, r2c2 = st.columns(2)
        with r2c1: st.metric("Gross Profit", f"${total_gp:,.2f}")
        with r2c2: st.metric("Gross Margin", f"{overall_margin:.2%}")

    st.divider()

    # Top Products Table
    st.subheader("Top Products (click rows → filter)")

    by_title = aggregate_by_title(
        filtered_df,
        {"qtySold": ("qtySold", "sum"), "netSales": ("netSales", "sum")}
    )

    inv_latest = get_latest_non_null_by_title(filtered_df, "inventoryNow").rename(
        columns={"inventoryNow": "Inventory Left"}
    )

    top_data = by_title.merge(inv_latest, on="title", how="left")
    top_data = top_data.sort_values(["netSales", "qtySold"], ascending=[False, False])

    top_display = top_data.rename(columns={
        "title": "Product",
        "qtySold": "Qty Sold",
        "netSales": "Revenue",
    })

    # Configure grid
    grid_builder = GridOptionsBuilder.from_dataframe(top_display)
    grid_builder.configure_selection(selection_mode="multiple", use_checkbox=True)
    grid_builder.configure_column("Revenue", valueFormatter="value != null ? Intl.NumberFormat().format(value) : ''")
    grid_builder.configure_column("Qty Sold", valueFormatter="value != null ? Math.round(value) : ''")
    grid_builder.configure_column("Inventory Left", valueFormatter="value != null ? Math.round(value) : ''")
    grid_builder.configure_grid_options(domLayout="normal")

    grid_options = grid_builder.build()
    grid_options["sortModel"] = [
        {"colId": "Revenue", "sort": "desc"},
        {"colId": "Qty Sold", "sort": "desc"},
    ]

    # Calculate height based on data size (make it larger as requested)
    table_height = get_table_height(
        len(top_display),
        min_height=400 if not compact else 450,
        max_height=700 if not compact else 600
    )

    grid_result = AgGrid(
        top_display,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=table_height,
        fit_columns_on_grid_load=True,
        theme="alpine",
    )

    # Filter buttons and export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Filter to selected Products (Top Table)", key="btn_filter_selected_titles"):
            selected_titles = sorted({row.get("Product") for row in grid_result["selected_rows"] if row.get("Product")})
            st.session_state["pick_titles"] = selected_titles
            st.rerun()

    with col2:
        if st.button("Clear product/title filter", key="btn_clear_selected_titles"):
            st.session_state["pick_titles"] = []
            st.rerun()

    with col3:
        with st.expander("📥 Export Top Products"):
            create_download_buttons(top_display, "top_products", "top_products")

    st.divider()

    # Time Series Charts
    with st.sidebar:
        chart_style = st.selectbox("Time-series style", CHART_STYLES, index=0, key="ts_style")

    daily_data = build_daily_timeseries(filtered_df, full_df)

    if compact:
        # Mobile: vertical layout
        st.plotly_chart(
            ChartRenderer.create_time_series(
                chart_style, daily_data, "dateYMD", "net",
                "Daily Revenue (Net)", hover_format=":.2f"
            ),
            use_container_width=True
        )
        st.plotly_chart(
            ChartRenderer.create_time_series(
                chart_style, daily_data, "dateYMD", "qty",
                "Daily Quantity Sold", hover_format=":,.0f", is_count=True
            ),
            use_container_width=True
        )
        st.plotly_chart(
            ChartRenderer.create_time_series(
                chart_style, daily_data, "dateYMD", "inv",
                "Inventory Left (sum of selected)", hover_format=":,.0f", is_count=True
            ),
            use_container_width=True
        )
        st.plotly_chart(
            ChartRenderer.create_time_series(
                chart_style, daily_data, "dateYMD", "pct_of_day_revenue",
                "% of Daily Revenue (Selected vs Total)", hover_format=":.2f", is_percentage=True
            ),
            use_container_width=True
        )
        st.plotly_chart(
            ChartRenderer.create_time_series(
                chart_style, daily_data, "dateYMD", "avg_margin_pct",
                "Avg Gross Margin (Daily)", hover_format=":.2f", is_percentage=True
            ),
            use_container_width=True
        )
    else:
        # Desktop: grid layout
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                ChartRenderer.create_time_series(
                    chart_style, daily_data, "dateYMD", "net",
                    "Daily Revenue (Net)", hover_format=":.2f"
                ),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                ChartRenderer.create_time_series(
                    chart_style, daily_data, "dateYMD", "qty",
                    "Daily Quantity Sold", hover_format=":,.0f", is_count=True
                ),
                use_container_width=True
            )

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                ChartRenderer.create_time_series(
                    chart_style, daily_data, "dateYMD", "inv",
                    "Inventory Left (sum of selected)", hover_format=":,.0f", is_count=True
                ),
                use_container_width=True
            )
        with c4:
            st.plotly_chart(
                ChartRenderer.create_time_series(
                    chart_style, daily_data, "dateYMD", "pct_of_day_revenue",
                    "% of Daily Revenue (Selected vs Total)", hover_format=":.2f", is_percentage=True
                ),
                use_container_width=True
            )

        st.plotly_chart(
            ChartRenderer.create_time_series(
                chart_style, daily_data, "dateYMD", "avg_margin_pct",
                "Avg Gross Margin (Daily)", hover_format=":.2f", is_percentage=True
            ),
            use_container_width=True
        )

    st.divider()

    # Top 10 Rankings
    st.subheader("Top 10 Rankings")

    rank_agg = aggregate_by_title(
        filtered_df,
        {
            "qtySold": ("qtySold", "sum"),
            "netSales": ("netSales", "sum"),
            "grossSales": ("grossSales", "sum"),
            "grossProfit": ("grossProfit", "sum") if "grossProfit" in filtered_df.columns else ("netSales", "sum")
        }
    )

    total_net_rank = float(rank_agg["netSales"].sum()) if not rank_agg.empty else 0.0
    if total_net_rank == 0:
        rank_agg["pctRevenue"] = 0.0
    else:
        rank_agg["pctRevenue"] = (rank_agg["netSales"] / total_net_rank) * 100.0

    # Add margin percentage
    if "grossProfit" in rank_agg.columns:
        rank_agg["Margin %"] = np.where(
            rank_agg["netSales"] > 0,
            (rank_agg["grossProfit"] / rank_agg["netSales"]) * 100.0,
            0.0
        )
    else:
        rank_agg["Margin %"] = 0.0

    # Create ranking tables
    top_qty = (
        rank_agg.sort_values(["qtySold", "netSales"], ascending=[False, False])
        .head(10)[["title", "qtySold"]]
        .rename(columns={"title": "Product", "qtySold": "Qty Sold"})
    )

    top_rev = (
        rank_agg.sort_values(["netSales", "grossSales"], ascending=[False, False])
        .head(10)[["title", "grossSales", "netSales", "grossProfit", "Margin %"]]
        .rename(columns={
            "title": "Product",
            "grossSales": "Gross",
            "netSales": "Net",
            "grossProfit": "Gross Profit"
        })
    )

    top_pct = (
        rank_agg.sort_values(["pctRevenue", "netSales"], ascending=[False, False])
        .head(10)[["title", "pctRevenue"]]
        .rename(columns={"title": "Product", "pctRevenue": "% of Revenue"})
    )

    # Display rankings
    col_q, col_r, col_p = st.columns(3)

    with col_q:
        st.markdown("**Top by Qty Sold**")
        st.dataframe(
            top_qty,
            use_container_width=True,
            hide_index=True,
            column_config={"Qty Sold": cc.NumberColumn(format="%d")}
        )

    with col_r:
        st.markdown("**Top by Revenue**")
        st.dataframe(
            top_rev,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Gross": cc.NumberColumn(format="$%.2f"),
                "Net": cc.NumberColumn(format="$%.2f"),
                "Gross Profit": cc.NumberColumn(format="$%.2f"),
                "Margin %": cc.NumberColumn(format="%.2f%%"),
            }
        )

    with col_p:
        st.markdown("**Top by % of Total Revenue**")
        st.dataframe(
            top_pct,
            use_container_width=True,
            hide_index=True,
            column_config={"% of Revenue": cc.NumberColumn(format="%.2f%%")}
        )

    st.divider()

    # Detail Table
    st.subheader("Detail")

    sorted_df = filtered_df.sort_values(["netSales", "qtySold"], ascending=[False, False])

    columns_order = [
        "date_display", "title", "barcode", "barcodesDistinct", "sku",
        "ordersCount", "qtySold", "grossSales", "netSales", "avgUnitNet", "avgUnitGross",
        "discountRate", "unitCost", "totalCost", "grossProfit", "grossMarginRate",
        "inventoryNow", "hasImage"
    ]

    available_columns = [col for col in columns_order if col in sorted_df.columns]
    display_df = sorted_df[available_columns].rename(columns={
        "date_display": "Date",
        "title": "Product",
        "barcode": "Barcode",
        "barcodesDistinct": "Barcodes (Distinct)",
        "sku": "SKU",
        "ordersCount": "Orders",
        "qtySold": "Qty",
        "grossSales": "Gross",
        "netSales": "Net",
        "avgUnitNet": "Avg Unit Net",
        "avgUnitGross": "Avg Unit Gross",
        "discountRate": "Discount",
        "unitCost": "Unit Cost",
        "totalCost": "Total Cost",
        "grossProfit": "Gross Profit",
        "grossMarginRate": "Gross Margin",
        "inventoryNow": "Inventory",
        "hasImage": "Has Image"
    })

    # Ensure string columns are properly formatted
    for col in ["SKU", "Barcode", "Barcodes (Distinct)", "Product"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype(str)

    # Column configuration
    column_config = {}
    config_mapping = {
        "Gross": ("$%.2f", cc.NumberColumn),
        "Net": ("$%.2f", cc.NumberColumn),
        "Avg Unit Net": ("$%.4f", cc.NumberColumn),
        "Avg Unit Gross": ("$%.4f", cc.NumberColumn),
        "Discount": ("%.2f%%", cc.NumberColumn),
        "Qty": ("%d", cc.NumberColumn),
        "Orders": ("%d", cc.NumberColumn),
        "Inventory": ("%d", cc.NumberColumn),
        "Unit Cost": ("$%.2f", cc.NumberColumn),
        "Total Cost": ("$%.2f", cc.NumberColumn),
        "Gross Profit": ("$%.2f", cc.NumberColumn),
        "Gross Margin": ("%.2f%%", cc.NumberColumn),
    }

    for col_name, (format_str, col_type) in config_mapping.items():
        if col_name in display_df.columns:
            column_config[col_name] = col_type(format=format_str)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=520 if not compact else 680,
        column_config=column_config,
        hide_index=True,
    )

    # Export options for detail table
    with st.expander("📥 Export Detail Table"):
        create_download_buttons(display_df, "detail_data", "detail")


def render_stock_views_tab(filtered_df: pd.DataFrame):
    """Render the stock views tab."""
    st.subheader("Stock Views")

    stock_analyzer = StockAnalyzer(filtered_df)

    # Shopping List
    st.markdown("### Shopping List")

    # Controls for low-velocity filter
    st.markdown("""
    <style>
    .shop-controls .stMarkdown { margin-bottom: 0.25rem; }
    .shop-controls [data-testid="stHorizontalBlock"] > div { padding-right: .5rem; }
    .shop-controls label { font-size: 0.875rem; }
    .shop-controls .stSlider > div[data-baseweb="slider"] { margin-top: .35rem; }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='shop-controls'>", unsafe_allow_html=True)

        hdr1, hdr2, hdr3 = st.columns([1.1, 0.9, 1.6])
        with hdr1:
            st.markdown("**Low-velocity toggle**")
        with hdr2:
            st.markdown("**Price <**")
        with hdr3:
            st.markdown("**Sold % <**")

        col1, col2, col3 = st.columns([1.1, 0.9, 1.6])
        with col1:
            exclude_low_vel = st.toggle(
                "Exclude low-velocity cheap items",
                value=st.session_state.get("shop_exclude_low_vel", False),
                key="shop_exclude_low_vel"
            )
        with col2:
            price_threshold = st.number_input(
                label="Price <",
                min_value=0.0,
                value=float(st.session_state.get("shop_price_thr", 10.0)),
                step=0.5,
                key="shop_price_thr",
                label_visibility="collapsed",
                help="Exclude only if price is below this value."
            )
        with col3:
            sold_pct_threshold = st.slider(
                label="Sold % <",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get("shop_pct_thr", 10.0)),
                step=1.0,
                key="shop_pct_thr",
                label_visibility="collapsed",
                help="Exclude only if sold-through % is below this value."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    shopping_list = stock_analyzer.get_shopping_list(
        exclude_low_velocity=exclude_low_vel,
        price_threshold=price_threshold,
        sold_percent_threshold=sold_pct_threshold
    )

    st.dataframe(
        shopping_list,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qty Sold": cc.NumberColumn(format="%d"),
            "Inventory Left": cc.NumberColumn(format="%d"),
            "Item Price": cc.NumberColumn(format="$%.2f"),
            "Sold %": cc.NumberColumn(format="%.1f%%"),
        }
    )

    # Export options for shopping list
    col_export1, col_export2 = st.columns(2)

    with col_export1:
        with st.expander("📥 Export Shopping List"):
            create_download_buttons(shopping_list, "shopping_list", "shopping_list")

    with col_export2:
        from data_loader import PinnedShoppingListManager

        if st.button("📌 Add to My Shopping List", key="export_to_shopping_list", help="Add all items to your mobile shopping list"):
            pinned_mgr = PinnedShoppingListManager()

            # Convert shopping list to format for bulk_pin_items
            items_to_pin = []
            for _, row in shopping_list.iterrows():
                items_to_pin.append({
                    "product": row.get("Product", ""),
                    "barcode": row.get("Barcode", ""),
                    "sku": row.get("SKU", "")
                })

            # Bulk pin items
            count = pinned_mgr.bulk_pin_items(items_to_pin)

            if count > 0:
                st.success(f"✅ Added {count} new items to My Shopping List! (Duplicates skipped)")
            else:
                st.info("ℹ️ All items are already in your shopping list!")

            st.rerun()

    st.divider()

    # Out of Stock
    st.markdown("### Out of Stock")
    oos_items = stock_analyzer.get_out_of_stock_items()

    st.dataframe(
        oos_items,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qty Sold": cc.NumberColumn(format="%d"),
            "Inventory Left": cc.NumberColumn(format="%d"),
            "Item Price": cc.NumberColumn(format="$%.2f"),
        }
    )

    # Export options for out of stock
    col_oos1, col_oos2 = st.columns(2)

    with col_oos1:
        with st.expander("📥 Export Out of Stock"):
            create_download_buttons(oos_items, "out_of_stock", "oos")

    with col_oos2:
        if st.button("📌 Add to My Shopping List", key="export_oos_to_shopping_list", help="Add all out of stock items to your mobile shopping list"):
            pinned_mgr = PinnedShoppingListManager()

            # Convert out of stock to format for bulk_pin_items
            items_to_pin = []
            for _, row in oos_items.iterrows():
                items_to_pin.append({
                    "product": row.get("Product", ""),
                    "barcode": row.get("Barcode", ""),
                    "sku": row.get("SKU", "")
                })

            # Bulk pin items
            count = pinned_mgr.bulk_pin_items(items_to_pin)

            if count > 0:
                st.success(f"✅ Added {count} new items to My Shopping List! (Duplicates skipped)")
            else:
                st.info("ℹ️ All items are already in your shopping list!")

            st.rerun()

    st.divider()

    # Low on Stock
    st.markdown("### Low on Stock (tiered)")
    low_stock = stock_analyzer.get_low_stock_items()
    low_stock_display = low_stock[["Product", "Qty Sold", "Inventory Left", "Threshold", "Item Price", "Barcode", "SKU"]]

    st.dataframe(
        low_stock_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qty Sold": cc.NumberColumn(format="%d"),
            "Inventory Left": cc.NumberColumn(format="%d"),
            "Threshold": cc.NumberColumn(format="%d"),
            "Item Price": cc.NumberColumn(format="$%.2f"),
        }
    )

    # Export options for low stock
    col_low1, col_low2 = st.columns(2)

    with col_low1:
        with st.expander("📥 Export Low Stock"):
            create_download_buttons(low_stock_display, "low_stock", "low_stock")

    with col_low2:
        if st.button("📌 Add to My Shopping List", key="export_low_to_shopping_list", help="Add all low stock items to your mobile shopping list"):
            pinned_mgr = PinnedShoppingListManager()

            # Convert low stock to format for bulk_pin_items
            items_to_pin = []
            for _, row in low_stock_display.iterrows():
                items_to_pin.append({
                    "product": row.get("Product", ""),
                    "barcode": row.get("Barcode", ""),
                    "sku": row.get("SKU", "")
                })

            # Bulk pin items
            count = pinned_mgr.bulk_pin_items(items_to_pin)

            if count > 0:
                st.success(f"✅ Added {count} new items to My Shopping List! (Duplicates skipped)")
            else:
                st.info("ℹ️ All items are already in your shopping list!")

            st.rerun()

    st.divider()

    # Missing COGS
    st.markdown("### Missing COGS")
    missing_cogs = stock_analyzer.get_missing_cogs_items()

    if missing_cogs.empty:
        st.info("✅ No items with missing COGS found in the current filter window.")
    else:
        st.dataframe(
            missing_cogs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Qty Sold": cc.NumberColumn(format="%d"),
                "Inventory Left": cc.NumberColumn(format="%d"),
                "Item Price": cc.NumberColumn(format="$%.2f"),
                "Revenue": cc.NumberColumn(format="$%.2f"),
                "Total_COGS": cc.NumberColumn(format="$%.2f"),
            }
        )

        # Export options for missing COGS
        with st.expander("📥 Export Missing COGS"):
            create_download_buttons(missing_cogs, "missing_cogs", "missing_cogs")


def render_pareto_stockout_tab(filtered_df: pd.DataFrame):
    """Render Pareto and Stock-out analysis tab."""
    st.subheader("Pareto (Revenue) & Projected Stock-Out Dates")

    # Pareto Chart
    pareto_data = aggregate_by_title(filtered_df, {"netSales": ("netSales", "sum")})
    pareto_data = pareto_data.sort_values("netSales", ascending=False)

    total_net = float(pareto_data["netSales"].sum()) if not pareto_data.empty else 0.0
    if total_net > 0:
        pareto_data["Share %"] = (pareto_data["netSales"] / total_net) * 100.0
        pareto_data["Cum %"] = pareto_data["Share %"].cumsum()
    else:
        pareto_data["Share %"] = 0.0
        pareto_data["Cum %"] = 0.0

    fig_pareto = ChartRenderer.create_pareto_chart(pareto_data)
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.divider()

    # Stock-Out Analysis
    st.markdown("### Projected Stock-Out Dates")

    so_days_threshold = st.slider(
        "Show items projecting stock-out within (days)",
        min_value=0,
        max_value=120,
        value=14,
        step=1,
        key="so_days_thr",
        help="Only show products whose projected stock-out date is within X days from today."
    )

    stockout_analyzer = StockOutAnalyzer(filtered_df)
    stockout_table = stockout_analyzer.build_stockout_table()

    # Add identifiers
    barcode_data = get_most_frequent_non_empty(filtered_df, "barcode").rename(
        columns={"barcode": "Barcode"}
    )
    sku_data = get_most_frequent_non_empty(filtered_df, "sku").rename(columns={"sku": "SKU"})

    stockout_table = (
        stockout_table.merge(barcode_data, on="title", how="left")
        .merge(sku_data, on="title", how="left")
        .rename(columns={"title": "Product"})
    )

    # Calculate days until stock-out
    today_cst = pd.Timestamp.now(tz="America/Chicago").normalize().tz_localize(None)
    stockout_table["_proj_ts"] = pd.to_datetime(stockout_table["Projected Stock-Out Date"], errors="coerce")
    stockout_table["Days Until"] = (stockout_table["_proj_ts"] - today_cst).dt.days

    # Filter by threshold
    if so_days_threshold is not None:
        stockout_table = stockout_table[
            (stockout_table["Days Until"].notna()) &
            (stockout_table["Days Until"] >= 0) &
            (stockout_table["Days Until"] <= so_days_threshold)
        ]

    # Sort by urgency
    if "Days Until" in stockout_table.columns and stockout_table["Days Until"].notna().any():
        stockout_table = stockout_table.sort_values(["Days Until"], ascending=[True])
    else:
        days_numeric = pd.to_numeric(stockout_table["Days to Stock-Out"], errors="coerce")
        stockout_table["__is_inf__"] = ~np.isfinite(days_numeric)
        stockout_table["__days__"] = days_numeric
        stockout_table = stockout_table.sort_values(["__is_inf__", "__days__"], ascending=[True, True])
        stockout_table = stockout_table.drop(columns=["__is_inf__", "__days__"])

    stockout_display = stockout_table[["Product", "Inventory Left", "Avg Daily Sold", "Days to Stock-Out",
                                      "Days Until", "Projected Stock-Out Date", "Barcode", "SKU"]]

    st.dataframe(
        stockout_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Inventory Left": cc.NumberColumn(format="%d"),
            "Avg Daily Sold": cc.NumberColumn(format="%.2f"),
            "Days to Stock-Out": cc.NumberColumn(format="%.1f"),
            "Days Until": cc.NumberColumn(format="%d"),
        }
    )

    # Export options for stock-out projections
    with st.expander("📥 Export Stock-Out Projections"):
        create_download_buttons(stockout_display, "stock_out_projections", "stockout")


def render_stale_inventory_tab(filtered_df: pd.DataFrame):
    """Render stale inventory analysis tab."""
    st.subheader("Stale Inventory (Python)")

    # Controls
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        window_days = st.slider("Window (days)", 7, 90, 30, step=1, key="stale_win")
    with col_b:
        no_sale_days = st.number_input("Flag if no sale for ≥ (days)", 7, 180, 30, step=1, key="stale_nosale")
    with col_c:
        doc_min = st.number_input("Flag if Days of Cover ≥", 15, 365, 60, step=5, key="stale_doc")

    stale_analyzer = StaleInventoryAnalyzer(filtered_df)
    stale_data = stale_analyzer.compute_stale_metrics(
        key="title",
        window_days=int(window_days),
        stale_days_since_last=int(no_sale_days),
        stale_doc_min=float(doc_min),
    )

    # Add identifiers
    barcode_data = get_most_frequent_non_empty(filtered_df, "barcode").rename(columns={"barcode": "Barcode"})
    sku_data = get_most_frequent_non_empty(filtered_df, "sku").rename(columns={"sku": "SKU"})

    stale_data = (
        stale_data.merge(barcode_data, left_on="title", right_on="title", how="left")
        .merge(sku_data, left_on="title", right_on="title", how="left")
    )

    # Sort by stale flag and metrics
    doc_col = f"DaysToStockOut_{window_days}"
    stale_data = stale_data.sort_values(["StaleFlag", "DaysSinceLastSale", doc_col], ascending=[False, False, False])

    # Rename columns
    stale_data = stale_data.rename(columns={
        "title": "Product",
        f"Sold_{window_days}": f"Sold_{window_days}",
        f"Velocity_{window_days}": f"Velocity_{window_days}",
        doc_col: doc_col,
    })

    stale_display = stale_data[["Product", "Inventory Left", f"Sold_{window_days}", f"Velocity_{window_days}",
                               doc_col, "LastSaleDate", "DaysSinceLastSale", "FirstSeenDate", "AgeDays",
                               "StaleFlag", "Barcode", "SKU"]]

    st.dataframe(
        stale_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Inventory Left": cc.NumberColumn(format="%d"),
            f"Sold_{window_days}": cc.NumberColumn(format="%d"),
            f"Velocity_{window_days}": cc.NumberColumn(format="%.2f"),
            doc_col: cc.NumberColumn(format="%.1f"),
            "DaysSinceLastSale": cc.NumberColumn(format="%d"),
            "AgeDays": cc.NumberColumn(format="%d"),
        }
    )

    # Export options for stale inventory
    with st.expander("📥 Export Stale Inventory"):
        create_download_buttons(stale_display, "stale_inventory", "stale")


def render_shopping_list_tab(df: pd.DataFrame):
    """Render the mobile-optimized shopping list tab."""
    from data_loader import PinnedShoppingListManager

    # Mobile-optimized CSS
    st.markdown("""
        <style>
        /* Shopping List Mobile Styles */
        .shopping-list-item {
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .shopping-list-item.checked {
            opacity: 0.6;
            background: #f5f5f5;
        }
        .item-name {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .item-name.checked {
            text-decoration: line-through;
            color: #888;
        }
        .item-details {
            font-size: 0.85rem;
            color: #666;
            margin-top: 0.25rem;
        }
        .search-box {
            position: sticky;
            top: 0;
            z-index: 100;
            background: white;
            padding: 1rem 0;
            margin-bottom: 1rem;
        }
        .action-buttons {
            display: flex;
            gap: 0.5rem;
            margin: 1rem 0;
        }
        /* Make buttons full width on mobile */
        @media (max-width: 768px) {
            .action-buttons {
                flex-direction: column;
            }
            div[data-testid="stButton"] > button {
                width: 100%;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛒 My Shopping List")

    pinned_mgr = PinnedShoppingListManager()

    # Search and Add Section
    st.markdown("<div class='search-box'>", unsafe_allow_html=True)

    search_col1, search_col2 = st.columns([3, 1])

    with search_col1:
        search_query = st.text_input(
            "Search or scan barcode",
            placeholder="Type product name or barcode...",
            key="shopping_search",
            label_visibility="collapsed"
        )

    with search_col2:
        scan_button = st.button("📷 Scan", key="scan_barcode_btn", help="Scan barcode with camera", use_container_width=True)

    # Initialize camera state
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    # Toggle camera state when button is clicked
    if scan_button:
        st.session_state.camera_active = not st.session_state.camera_active

    # Camera input for barcode scanning
    if st.session_state.camera_active:
        st.info("📸 Point camera at barcode and capture")
        camera_input = st.camera_input("Scan barcode", key="barcode_camera")

        if camera_input:
            try:
                from PIL import Image
                from pyzbar.pyzbar import decode
                import cv2
                import numpy as np

                # Read the image
                image = Image.open(camera_input)

                # Convert PIL image to OpenCV format for better processing
                img_array = np.array(image)

                # Convert to grayscale for better barcode detection
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                # Apply sharpening to improve focus (simulates autofocus)
                kernel = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
                sharpened = cv2.filter2D(gray, -1, kernel)

                # Try multiple preprocessing techniques for better detection
                barcodes = []

                # Try 1: Sharpened grayscale (best for slightly out-of-focus images)
                barcodes = decode(sharpened)

                # Try 2: Original grayscale
                if not barcodes:
                    barcodes = decode(gray)

                # Try 3: Adaptive thresholding with sharpening
                if not barcodes:
                    adaptive = cv2.adaptiveThreshold(
                        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                    )
                    barcodes = decode(adaptive)

                # Try 4: Increase contrast
                if not barcodes:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    barcodes = decode(enhanced)

                # Try 5: Simple thresholding
                if not barcodes:
                    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    barcodes = decode(thresh)

                # Try 6: Inverted thresholding (for dark barcodes on light background)
                if not barcodes:
                    _, inv_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    barcodes = decode(inv_thresh)

                # Try 7: Original color image as fallback
                if not barcodes:
                    barcodes = decode(img_array)

                if barcodes:
                    # Get the first barcode found
                    barcode_data = barcodes[0].data.decode('utf-8')
                    barcode_type = barcodes[0].type

                    st.success(f"✅ Barcode detected: {barcode_data} ({barcode_type})")

                    # Auto-fill search with the detected barcode
                    st.session_state["shopping_search"] = barcode_data

                    # Show the detected area on the image
                    points = barcodes[0].polygon
                    if len(points) == 4:
                        pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
                        cv2.polylines(img_array, [pts], True, (0, 255, 0), 3)

                    # Display the image with detection box
                    st.image(img_array, caption="Detected barcode area (green outline)", use_container_width=True)

                    # Trigger search
                    search_query = barcode_data

                else:
                    st.error("❌ No barcode detected. Please ensure:")
                    st.markdown("""
                    - The barcode is well-lit
                    - The barcode is in focus
                    - The entire barcode is visible in the frame
                    - Try holding the camera steady
                    """)
                    st.image(image, caption="Captured image", use_container_width=True)

            except ImportError as e:
                st.error("⚠️ Barcode scanning libraries not installed. Please install:")
                st.code("pip install pyzbar Pillow opencv-python-headless")
                st.info("Or type the barcode manually in the search box above.")
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.info("Please try again or type the barcode manually.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Search functionality
    if search_query:
        search_lower = search_query.lower()

        # Search in current data
        matches = df[
            (df["title"].astype(str).str.lower().str.contains(search_lower, na=False)) |
            (df["barcode"].astype(str).str.lower().str.contains(search_lower, na=False))
        ]

        if not matches.empty:
            st.success(f"Found {len(matches)} product(s) in inventory")

            # Show matches
            for idx, row in matches.head(10).iterrows():
                product_name = row.get("title", "Unknown")
                barcode = row.get("barcode", "")
                sku = row.get("sku", "")

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{product_name}**")
                    if barcode:
                        st.caption(f"Barcode: {barcode}")
                with col2:
                    if st.button("➕ Add", key=f"add_{idx}"):
                        pinned_mgr.pin_item(product_name, barcode, sku)
                        st.success(f"Added to list!")
                        st.rerun()
        else:
            # Check stale inventory
            st.warning("Not found in current inventory. Checking stale inventory...")
            stale_matches = df[df["title"].astype(str).str.lower().str.contains(search_lower, na=False)]

            if not stale_matches.empty:
                st.info("Found in historical data (may be out of stock)")
                for idx, row in stale_matches.head(5).iterrows():
                    product_name = row.get("title", "Unknown")
                    barcode = row.get("barcode", "")
                    sku = row.get("sku", "")

                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{product_name}**")
                        st.caption("⚠️ May be out of stock")
                    with col2:
                        if st.button("➕", key=f"add_stale_{idx}"):
                            pinned_mgr.pin_item(product_name, barcode, sku)
                            st.success("Added to list!")
                            st.rerun()
            else:
                st.error("Product not found in inventory")

        st.divider()

    # Load shopping list
    shopping_list = pinned_mgr.list_pinned_items()

    if not shopping_list:
        st.info("Your shopping list is empty. Search for products above to add them!")
    else:
        # Action buttons
        st.markdown("<div class='action-buttons'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear Checked", key="clear_checked_btn", use_container_width=True):
                count = pinned_mgr.clear_checked_items()
                if count > 0:
                    st.success(f"Removed {count} checked item(s)")
                    st.rerun()
                else:
                    st.info("No checked items to remove")

        with col2:
            # Count items
            total_items = len(shopping_list)
            checked_items = sum(1 for item in shopping_list if item.get("checked"))
            st.metric("Progress", f"{checked_items}/{total_items}")

        with col3:
            if st.button("📋 Export List", key="export_list_btn", use_container_width=True):
                # Create text export
                export_text = "Shopping List:\n\n"
                for item in shopping_list:
                    status = "☑" if item.get("checked") else "☐"
                    export_text += f"{status} {item['product']}\n"
                    if item.get("barcode"):
                        export_text += f"   Barcode: {item['barcode']}\n"
                st.download_button(
                    "Download as TXT",
                    export_text,
                    "shopping_list.txt",
                    "text/plain",
                    use_container_width=True
                )

        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()

        # Display shopping list items
        st.subheader(f"📝 Your List ({len(shopping_list)} items)")

        # Separate checked and unchecked
        unchecked_items = [item for item in shopping_list if not item.get("checked")]
        checked_items_list = [item for item in shopping_list if item.get("checked")]

        # Display unchecked items first
        for item in unchecked_items:
            product = item["product"]
            barcode = item.get("barcode", "")
            sku = item.get("sku", "")

            st.markdown(f"<div class='shopping-list-item'>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 6, 1])

            with col1:
                checked = st.checkbox(
                    "✓",
                    value=False,
                    key=f"check_{product}",
                    label_visibility="collapsed"
                )
                if checked:
                    pinned_mgr.toggle_checked(product, True)
                    st.rerun()

            with col2:
                st.markdown(f"<div class='item-name'>{product}</div>", unsafe_allow_html=True)
                details = []
                if barcode:
                    details.append(f"📊 {barcode}")
                if sku:
                    details.append(f"🔖 {sku}")
                if details:
                    st.markdown(f"<div class='item-details'>{' | '.join(details)}</div>", unsafe_allow_html=True)

            with col3:
                if st.button("🗑️", key=f"del_{product}", help="Remove from list"):
                    pinned_mgr.unpin_item(product)
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # Display checked items
        if checked_items_list:
            st.markdown("---")
            st.caption(f"✅ Checked Items ({len(checked_items_list)})")

            for item in checked_items_list:
                product = item["product"]
                barcode = item.get("barcode", "")
                sku = item.get("sku", "")

                st.markdown(f"<div class='shopping-list-item checked'>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns([1, 6, 1])

                with col1:
                    checked = st.checkbox(
                        "✓",
                        value=True,
                        key=f"check_{product}",
                        label_visibility="collapsed"
                    )
                    if not checked:
                        pinned_mgr.toggle_checked(product, False)
                        st.rerun()

                with col2:
                    st.markdown(f"<div class='item-name checked'>{product}</div>", unsafe_allow_html=True)
                    details = []
                    if barcode:
                        details.append(f"📊 {barcode}")
                    if sku:
                        details.append(f"🔖 {sku}")
                    if details:
                        st.markdown(f"<div class='item-details'>{' | '.join(details)}</div>", unsafe_allow_html=True)

                with col3:
                    if st.button("🗑️", key=f"del_{product}", help="Remove from list"):
                        pinned_mgr.unpin_item(product)
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Setup
    setup_page()

    # Authentication
    auth_manager = AuthManager()
    auth_manager.require_login()

    # Load data
    try:
        df = load_main_data()
    except Exception as e:
        st.error("Couldn't load Google Sheet. Check sharing, ID, and tab name.")
        st.exception(e)
        st.stop()

    if df.empty:
        st.info("No data found in the sheet.")
        st.stop()

    # Apply filters
    filter_manager = FilterManager(df)
    filtered_df, pick_titles, compact, report_period = filter_manager.render_sidebar_filters()

    # Report period banner
    render_report_period_banner(report_period, df)

    if filtered_df.empty:
        st.warning("No rows match your filters.")
        st.stop()

    # Preset chips
    render_preset_chips()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "📦 Stock Views", "📈 Pareto & Stock-Out", "🧊 Stale Inventory", "🛒 My Shopping List"
    ])

    with tab1:
        render_dashboard_tab(filtered_df, df, compact)

    with tab2:
        render_stock_views_tab(filtered_df)

    with tab3:
        render_pareto_stockout_tab(filtered_df)

    with tab4:
        render_stale_inventory_tab(filtered_df)

    with tab5:
        render_shopping_list_tab(df)


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit import column_config as cc
from typing import Dict, List, Tuple, Any, Optional
from data_processor import parse_search_terms, contains_any
from data_loader import PresetManager
from config import HAS_IMAGE_OPTIONS, PRESET_COMBINE_MODES, CHART_STYLES, DATE_RANGE_OPTIONS, DEFAULT_DATE_RANGE
from utils import get_date_range_from_option


class FilterManager:
    """Manages dashboard filters and state."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.preset_manager = PresetManager()

    def apply_preset_to_session(self, filter_definition: Dict[str, Any],
                              negate: bool = False, combine_mode: str = "OR") -> None:
        """Apply preset filters to session state."""
        def sanitize_list(value):
            return [s for s in (str(v).strip() for v in (value or [])) if s]

        titles = sanitize_list(filter_definition.get("title_contains", []))
        barcodes = sanitize_list(filter_definition.get("barcode_contains", []))
        merged_terms = sorted(set(titles) | set(barcodes))

        if negate:
            st.session_state["neg_contains_terms"] = merged_terms
            st.session_state["contains_terms"] = ""
        else:
            current_terms = parse_search_terms(st.session_state.get("contains_terms", ""))
            if combine_mode == "AND" and current_terms:
                merged = sorted(set(current_terms) & set(merged_terms))
            else:
                merged = sorted(set(current_terms) | set(merged_terms))
            st.session_state["contains_terms"] = ", ".join(merged)
            st.session_state.pop("neg_contains_terms", None)

        # Clear legacy keys
        for key in [
            "q", "t_contains", "b_contains",
            "neg_title_contains", "neg_barcode_contains", "neg_pick_titles"
        ]:
            st.session_state.pop(key, None)

        # Handle free-text query
        query = (filter_definition.get("query", "") or "").strip()
        if query:
            current_terms = parse_search_terms(st.session_state.get("contains_terms", ""))
            query_terms = parse_search_terms(query)
            st.session_state["contains_terms"] = ", ".join(
                sorted(set(current_terms + query_terms))
            )

    def render_sidebar_filters(self) -> Tuple[pd.DataFrame, List[str], bool, Tuple]:
        """Render sidebar filters and return filtered data."""
        date_min, date_max = self.df["dateYMD"].min(), self.df["dateYMD"].max()

        with st.sidebar:
            st.header("Filters")

            # Date range quick filter dropdown
            date_option = st.selectbox(
                "Date Range",
                options=list(DATE_RANGE_OPTIONS.keys()),
                index=list(DATE_RANGE_OPTIONS.keys()).index(
                    st.session_state.get("date_option", DEFAULT_DATE_RANGE)
                ),
                key="date_option_select"
            )
            st.session_state["date_option"] = date_option

            # Custom date range input (only show if "Custom range" is selected)
            if date_option == "Custom range":
                default_dates = st.session_state.get("__preset_dates__")
                if default_dates and isinstance(default_dates, tuple) and len(default_dates) == 2:
                    date_range_default = default_dates
                else:
                    date_range_default = (
                        date_min.date() if pd.notna(date_min) else pd.Timestamp.today().date(),
                        date_max.date() if pd.notna(date_max) else pd.Timestamp.today().date()
                    )

                date_range = st.date_input(
                    "Custom Date Range",
                    value=date_range_default,
                    key="flt_date_range"
                )

                if isinstance(date_range, tuple) and len(date_range) == 2:
                    st.session_state["__preset_dates__"] = date_range
                    effective_date_range = date_range
                else:
                    effective_date_range = date_range_default
            else:
                # Calculate date range from quick filter option
                start_date, end_date = get_date_range_from_option(date_option, date_min, date_max)
                effective_date_range = (start_date, end_date)
                st.session_state["__preset_dates__"] = effective_date_range

                # Show the calculated range for user reference
                st.caption(f"📅 {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}")

            date_range = effective_date_range

            # Search terms filter
            contains_terms = st.text_input(
                "Title or Barcode contains (comma-separated)",
                value=st.session_state.get("contains_terms", ""),
                key="flt_contains_terms",
                placeholder="e.g. sonny, hipper, 0123456789"
            )
            st.session_state["contains_terms"] = contains_terms
            st.caption("Matches any of the terms in **Title** or **Barcode**. Separate with commas or spaces.")

            # Has image filter
            has_img_choice = st.selectbox(
                "Has image",
                options=HAS_IMAGE_OPTIONS,
                index=self._get_has_image_index(),
                key="flt_has_image"
            )
            st.session_state["has_img_choice"] = has_img_choice

            # Compact layout toggle
            compact = st.checkbox(
                "Compact layout (mobile)",
                value=st.session_state.get("compact", True),
                key="flt_compact"
            )
            st.session_state["compact"] = compact

            self._render_presets_section()

            # Clear filters button
            if st.button("Clear filters", key="flt_clear", use_container_width=True):
                self._clear_all_filters()
                st.rerun()

        # Apply filters and return results
        return self._apply_filters_to_data(date_range)

    def _get_has_image_index(self) -> int:
        """Get the index for has_image selectbox."""
        choice = st.session_state.get("has_img_choice", "Both")
        try:
            return {"Both": 0, "Has image": 1, "No image": 2}[choice]
        except KeyError:
            return 0

    def _render_presets_section(self) -> None:
        """Render the presets management section."""
        st.markdown("### Presets")

        presets = self.preset_manager.list_presets()
        preset_names = [p["name"] for p in presets]

        # Preset selection
        chosen_presets = st.multiselect(
            "Select preset(s) to apply",
            options=preset_names,
            default=[],
            key="presets_multi"
        )

        # Combine mode
        combine_mode = st.radio(
            "Combine mode",
            options=PRESET_COMBINE_MODES,
            horizontal=True,
            index=0,
            key="presets_combine"
        )

        # Negate option
        negate_apply = st.checkbox(
            "Apply as EXCLUDE (negate match)",
            value=False,
            key="presets_negate"
        )

        # Apply presets button
        if st.button("Apply selected preset(s)", key="presets_apply_btn", use_container_width=True):
            self._apply_selected_presets(presets, chosen_presets, combine_mode, negate_apply)

        # Delete presets button
        if st.button("Delete selected preset(s)", key="presets_delete_btn", use_container_width=True):
            self._delete_selected_presets(chosen_presets)

        st.markdown("---")

        # Save new preset
        self._render_save_preset_section()

    def _apply_selected_presets(self, presets: List[Dict], chosen_names: List[str],
                              combine_mode: str, negate_apply: bool) -> None:
        """Apply selected presets with specified mode."""
        selected_presets = [p for p in presets if p["name"] in chosen_names]

        if not selected_presets:
            st.warning("Pick at least one preset.")
            return

        # Collect all terms from selected presets
        all_terms = []
        for preset in selected_presets:
            filter_def = preset["filters"]
            title_terms = [str(x).strip() for x in (filter_def.get("title_contains", []) or []) if str(x).strip()]
            barcode_terms = [str(x).strip() for x in (filter_def.get("barcode_contains", []) or []) if str(x).strip()]
            all_terms.extend(title_terms + barcode_terms)

        # Apply combine mode
        if combine_mode == "AND" and len(selected_presets) > 1:
            # For AND mode, find intersection of terms
            term_sets = []
            for preset in selected_presets:
                filter_def = preset["filters"]
                preset_terms = set()
                preset_terms.update(str(x).strip() for x in (filter_def.get("title_contains", []) or []) if str(x).strip())
                preset_terms.update(str(x).strip() for x in (filter_def.get("barcode_contains", []) or []) if str(x).strip())
                term_sets.append(preset_terms)

            final_terms = sorted(list(set.intersection(*term_sets))) if term_sets else []
        else:
            # OR mode - union of all terms
            final_terms = sorted(list(set(all_terms)))

        # Apply to session state
        if negate_apply:
            st.session_state["neg_contains_terms"] = final_terms
            st.session_state["contains_terms"] = ""
        else:
            current_terms = parse_search_terms(st.session_state.get("contains_terms", ""))
            st.session_state["contains_terms"] = ", ".join(sorted(set(current_terms) | set(final_terms)))
            st.session_state.pop("neg_contains_terms", None)

        st.success(f"Applied {len(chosen_names)} preset(s) with {combine_mode}{' (negated)' if negate_apply else ''}.")
        st.rerun()

    def _delete_selected_presets(self, chosen_names: List[str]) -> None:
        """Delete selected presets."""
        if not chosen_names:
            st.warning("Pick at least one preset to delete.")
            return

        for name in chosen_names:
            self.preset_manager.delete_preset(name)

        st.success(f"Deleted {len(chosen_names)} preset(s).")
        st.rerun()

    def _render_save_preset_section(self) -> None:
        """Render preset saving section."""
        preset_name = st.text_input(
            "New preset name",
            value=st.session_state.get("__new_preset_name__", ""),
            key="presets_name",
            placeholder="e.g. Sonny / Hipper / Weekend push",
        )
        st.session_state["__new_preset_name__"] = preset_name

        if st.button("Save current filters as preset", key="presets_save_btn", use_container_width=True):
            if preset_name.strip():
                terms = parse_search_terms(st.session_state.get("contains_terms", ""))
                filter_def = {
                    "query": "",
                    "title_contains": terms,
                    "barcode_contains": terms,
                    "has_image": st.session_state.get("has_img_choice", "Both"),
                    "pick_titles": st.session_state.get("pick_titles", []),
                }
                self.preset_manager.save_preset(preset_name.strip(), filter_def)
                st.success(f"Saved preset: {preset_name.strip()}")
            else:
                st.warning("Enter a preset name.")

    def _clear_all_filters(self) -> None:
        """Clear all filter states."""
        keys_to_clear = [
            "contains_terms", "neg_contains_terms", "has_img_choice",
            "pick_titles", "__preset_dates__", "__new_preset_name__", "compact"
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)

    def _apply_filters_to_data(self, date_range) -> Tuple[pd.DataFrame, List[str], bool, Tuple]:
        """Apply all filters to the data and return filtered result."""
        mask = pd.Series(True, index=self.df.index)
        effective_start, effective_end = pd.to_datetime(self.df["dateYMD"].min()), pd.to_datetime(self.df["dateYMD"].max())

        # Apply date filter
        if isinstance(date_range, tuple) and len(date_range) == 2:
            try:
                start, end = [pd.to_datetime(d, errors="coerce") for d in date_range]
            except Exception:
                start, end = None, None

            data_min = pd.to_datetime(self.df["dateYMD"].min())
            data_max = pd.to_datetime(self.df["dateYMD"].max())

            if pd.isna(start) or pd.isna(end):
                st.caption("Date range invalid — showing all available data.")
                effective_start, effective_end = data_min, data_max
            else:
                start, end = start.normalize(), end.normalize()

                if pd.notna(data_min) and pd.notna(data_max):
                    overlap_start = max(start, data_min)
                    overlap_end = min(end, data_max)

                    if overlap_start > overlap_end:
                        st.info(
                            f"No data exists in your selected range {start.date()} → {end.date()}. "
                            f"Available data spans {data_min.date()} → {data_max.date()}."
                        )
                        mask &= pd.Series(False, index=self.df.index)
                        effective_start, effective_end = overlap_start, overlap_end
                    else:
                        if overlap_start != start or overlap_end != end:
                            st.caption(f"Clamped to available data: {overlap_start.date()} → {overlap_end.date()}.")
                        mask &= self.df["dateYMD"].between(overlap_start, overlap_end)
                        effective_start, effective_end = overlap_start, overlap_end
                else:
                    st.caption("No date column available; skipping date filter.")
                    effective_start, effective_end = None, None

        # Apply contains filters
        manual_terms = parse_search_terms(st.session_state.get("contains_terms", ""))
        chip_maps = st.session_state.get("chip_terms_map", {})
        chip_terms_all = sorted(set().union(*chip_maps.values())) if chip_maps else []
        terms = sorted(set(manual_terms) | set(chip_terms_all))

        if terms:
            title_condition = contains_any(self.df.get("title", ""), terms)
            barcode_condition = contains_any(self.df.get("barcode", ""), terms)
            mask &= (title_condition | barcode_condition)

        # Apply negative contains filters
        neg_terms = st.session_state.get("neg_contains_terms", [])
        if neg_terms:
            neg_title = contains_any(self.df.get("title", ""), neg_terms)
            neg_barcode = contains_any(self.df.get("barcode", ""), neg_terms)
            mask &= ~(neg_title | neg_barcode)

        # Apply has image filter
        pick_titles = st.session_state.get("pick_titles", [])
        choice = st.session_state.get("has_img_choice", "Both")

        if "hasImage" in self.df.columns:
            if choice == "Has image":
                mask &= self.df["hasImage"] == True
            elif choice == "No image":
                mask &= (self.df["hasImage"] == False) | self.df["hasImage"].isna()

        # Apply title filter from top table clicks
        if pick_titles:
            mask &= self.df["title"].astype(str).isin(pick_titles)

        filtered_df = self.df[mask].copy()
        return filtered_df, pick_titles, st.session_state.get("compact", True), (effective_start, effective_end)


class ChartRenderer:
    """Handles chart rendering for the dashboard."""

    @staticmethod
    def format_date_axis(fig):
        """Apply consistent date axis formatting."""
        fig.update_xaxes(tickformat="%b %d", dtick="D7", showgrid=False)
        fig.update_yaxes(showgrid=True)
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        return fig

    @staticmethod
    def create_time_series(chart_type: str, data: pd.DataFrame, x: str, y: str,
                          title: str, hover_format: str = ":.2f",
                          is_percentage: bool = False, is_count: bool = False):
        """Create a time series chart (line or bar)."""
        if chart_type == "Bar":
            fig = px.bar(
                data, x=x, y=y,
                hover_data={"date_label": True, y: hover_format, x: False},
                title=title,
                text=y
            )

            if is_percentage:
                fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", cliponaxis=False)
            elif is_count:
                fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
            else:
                fig.update_traces(texttemplate="%{y:,.2f}", textposition="outside", cliponaxis=False)
        else:
            fig = px.line(
                data, x=x, y=y,
                hover_data={"date_label": True, y: hover_format, x: False},
                title=title
            )

            if is_percentage:
                fig.update_traces(mode="lines+markers+text", texttemplate="%{y:.2f}%", textposition="top center")
            elif is_count:
                fig.update_traces(mode="lines+markers+text", texttemplate="%{y:,.0f}", textposition="top center")
            else:
                fig.update_traces(mode="lines+markers+text", texttemplate="%{y:,.2f}", textposition="top center")

        fig = ChartRenderer.format_date_axis(fig)
        if is_percentage:
            fig.update_yaxes(ticksuffix="%", rangemode="tozero")

        return fig

    @staticmethod
    def create_pareto_chart(data: pd.DataFrame) -> go.Figure:
        """Create a Pareto chart for revenue analysis."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add revenue bars
        fig.add_trace(
            go.Bar(x=data["title"], y=data["netSales"], name="Revenue"),
            secondary_y=False
        )

        # Add cumulative percentage line
        fig.add_trace(
            go.Scatter(x=data["title"], y=data["Cum %"], name="Cumulative %", mode="lines+markers"),
            secondary_y=True
        )

        # Add 80% reference line
        fig.add_hline(y=80, line_dash="dash", line_color="red", secondary_y=True)

        fig.update_layout(
            title="Pareto of Net Revenue by Product",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Product (sorted by revenue)",
            yaxis_title="Revenue",
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True, ticksuffix="%")

        return fig
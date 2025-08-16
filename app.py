import os, json, re
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2 import service_account
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit import column_config as cc
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Daily Products Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",  # nicer on mobile
)

# --- Sticky tabs + chips styling ---
st.markdown(
    """
    <style>
      /* ===== Sticky Tabs + Chipbar (scoped) ===== */
      :root {
        /* distance from top of the main content where tabs should stick */
        --tabs-stick-top: 0px;
        /* visual tab height used to place chipbar right under */
        --tabs-height: 46px;
      }

      /* Ensure the main scroll container can host sticky children */
      section.main { position: relative; }

      /* Target Streamlit's tab list robustly */
      section.main div[data-testid="stTabs"] div[role="tablist"],
      section.main .stTabs div[role="tablist"] {
        position: sticky;
        top: var(--tabs-stick-top);
        z-index: 1000;
        background-color: inherit;   /* match theme background */
        padding: .25rem 0 .35rem 0;
        border-bottom: 1px solid rgba(128,128,128,.2);
      }

      /* Avoid clipping labels when sticky */
      section.main div[data-testid="stTabs"] [role="tab"],
      section.main .stTabs [role="tab"] { overflow: visible; }

      /* Chipbar sits right below the tabs and scrolls horizontally if needed */
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

      /* Do not alter the Streamlit global header */
      header { position: static !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Optional: minimal password gate (uses secrets)
# =========================
def require_login():
    user = st.secrets.get("DASHBOARD_USERNAME", "").strip()
    pwd  = st.secrets.get("DASHBOARD_PASSWORD", "").strip()
    if not user or not pwd:
        return True  # auth disabled if either is blank
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        st.title("Daily Products Dashboard")
        u = st.text_input("Username", key="auth_user")
        p = st.text_input("Password", type="password", key="auth_pass")
        if st.button("Login", key="auth_login_btn"):
            if u == user and p == pwd:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.stop()
    return True
require_login()  # <- uncomment to enable login

# =========================
# Google Sheets client (WRITE scope for presets)
# =========================
@st.cache_resource
def get_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",        # write (for presets)
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    sa_json = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    sa_file = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON_FILE", "")
    if sa_json:
        creds = service_account.Credentials.from_service_account_info(json.loads(sa_json), scopes=scopes)
    elif sa_file:
        creds = service_account.Credentials.from_service_account_file(sa_file, scopes=scopes)
    else:
        raise ValueError("No service account credentials provided in secrets.toml")
    return gspread.authorize(creds)

# =========================
# Parsing helpers
# =========================
def _coerce_num(x, pct=False):
    """Parse $, commas, %, (), blanks → float. If pct=True and input has '%', outputs fraction."""
    if pd.isna(x) or x == "": return pd.NA
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s: return pd.NA
    paren = s.startswith("(") and s.endswith(")")
    if paren: s = s[1:-1].strip()
    saw_pct = s.endswith("%")
    s = re.sub(r"[^0-9.\-]", "", s)  # strip $, commas, spaces, %
    if s == "": return pd.NA
    try:
        val = float(s)
    except Exception:
        return pd.NA
    if paren: val = -val
    if pct and saw_pct:
        val = val / 100.0
    return val

def coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Force id-like/text fields to strings
    for c in ["sku","barcode","barcodesDistinct","title","productHandle","variantId"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan":"", "None":""}).str.strip()

    # Numeric columns (incl. discountRate which may already be numeric percent)
    spec = {
        "ordersCount": False, "qtySold": False,
        "grossSales": False, "netSales": False,
        "avgUnitNet": False, "avgUnitGross": False,
        "discountRate": True,   # "12.3%" -> 0.123 if string; 12.3 stays 12.3 if numeric
        "inventoryNow": False,
        "unitCost": False,
        "totalCost": False,
        "grossProfit": False,
        "grossMarginRate": True,
    }
    for col, is_pct in spec.items():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].apply(lambda v: _coerce_num(v, pct=is_pct)), errors="coerce")

    # Date column: be flexible about header name
    date_source = None
    for cand in ["dateYMD", "Date", "date", "runDateYMD"]:
        if cand in df.columns:
            date_source = cand
            break
    if date_source:
        df["dateYMD"] = pd.to_datetime(df[date_source], errors="coerce")
        df["date_display"] = df["dateYMD"].dt.strftime("%b %d, %Y")
    else:
        df["dateYMD"] = pd.NaT
        df["date_display"] = ""

    # Booleans
    if "hasImage" in df.columns:
        df["hasImage"] = df["hasImage"].astype(str).str.lower().isin(["true","1","yes","y"])

    return df

def contains_any(series: pd.Series, needles: list[str]) -> pd.Series:
    """Case-insensitive OR across needles. Returns all-False for empty needles."""
    s = series.astype(str).str.lower().fillna("")
    if not needles:
        return pd.Series(False, index=s.index)
    cond = pd.Series(False, index=s.index)
    for n in needles:
        n = str(n).strip().lower()
        if not n:
            continue
        cond |= s.str.contains(re.escape(n), na=False)
    return cond

# =========================
# Load data
# =========================
@st.cache_data(ttl=300)
def load_data():
    gc = get_client()
    sh = gc.open_by_key(st.secrets["SPREADSHEET_ID"])
    ws = sh.worksheet(st.secrets.get("WORKSHEET_NAME", "Daily Products"))
    rows = ws.get_all_records()  # header-based
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return coerce_numeric_cols(df)

try:
    df = load_data()
except Exception as e:
    st.error("Couldn't load Google Sheet. Check sharing, ID, and tab name.")
    st.exception(e)
    st.stop()

if df.empty:
    st.info("No data found in the 'Daily Products' sheet.")
    st.stop()

# =========================
# Presets (stored in "Presets" worksheet)
# =========================
PRESETS_SHEET_NAME = st.secrets.get("PRESETS_SHEET_NAME", "Presets")

def _ensure_presets_sheet(sh):
    try:
        ws = sh.worksheet(PRESETS_SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(PRESETS_SHEET_NAME, rows=1000, cols=3)
        ws.update("A1:C1", [["name","updatedAt","filters_json"]])
    return ws

@st.cache_data(ttl=60)
def list_presets():
    """Return list[dict]: {"name":..., "filters":{...}}"""
    gc = get_client()
    sh = gc.open_by_key(st.secrets["SPREADSHEET_ID"])
    ws = _ensure_presets_sheet(sh)
    rows = ws.get_all_records()
    out = []
    for r in rows:
        try:
            filters = json.loads(r.get("filters_json","{}"))
        except Exception:
            filters = {}
        if r.get("name"):
            out.append({"name": r["name"], "filters": filters})
    out.sort(key=lambda x: x["name"].lower())
    return out

def save_preset(name: str, filters: dict):
    gc = get_client()
    sh = gc.open_by_key(st.secrets["SPREADSHEET_ID"])
    ws = _ensure_presets_sheet(sh)
    rows = ws.get_all_records()
    payload = [name, datetime.utcnow().isoformat(timespec="seconds")+"Z", json.dumps(filters, ensure_ascii=False)]
    # update if exists
    for i, r in enumerate(rows, start=2):  # header at row 1
        if r.get("name") == name:
            ws.update(f"A{i}:C{i}", [payload])
            list_presets.clear()
            return
    # append new
    ws.append_row(payload, value_input_option="RAW")
    list_presets.clear()

def delete_preset(name: str):
    gc = get_client()
    sh = gc.open_by_key(st.secrets["SPREADSHEET_ID"])
    ws = _ensure_presets_sheet(sh)
    rows = ws.get_all_records()
    for i, r in enumerate(rows, start=2):
        if r.get("name") == name:
            ws.delete_rows(i)
            break
    list_presets.clear()

# --- Helper: Apply a preset to session state (non-destructive and reusable) ---
def apply_preset_to_session(fdef: dict, negate: bool = False, combine_mode: str = "OR"):
    """
    Apply a preset's filters into session state.
    - When negate=True, stores filters into the negative keys and clears overlapping positives.
    - combine_mode controls how lists merge with existing values when not negated ("OR" union / "AND" intersection).
    """
    def _sanitize_list(x):
        return [s for s in (str(v).strip() for v in (x or [])) if s]

    titles   = _sanitize_list(fdef.get("title_contains",  []))
    barcodes = _sanitize_list(fdef.get("barcode_contains",[]))
    picks    = _sanitize_list(fdef.get("pick_titles",     []))
    q        = (fdef.get("query","") or "").strip()


    if negate:
        st.session_state["neg_title_contains"]   = titles
        st.session_state["neg_barcode_contains"] = barcodes
        st.session_state["neg_pick_titles"]      = picks
        # Clear positives for intuitive exclusion behavior
        st.session_state["t_contains"] = ""
        st.session_state["b_contains"] = ""
        st.session_state["pick_titles"] = []
    else:
        # Merge with existing positives
        def _merge(curr, add):
            curr_set = set([s.strip() for s in (curr or "").split(",") if s.strip()]) if isinstance(curr, str) else set(curr or [])
            add_set  = set(add or [])
            if combine_mode == "AND" and curr_set:
                return list(curr_set & add_set)
            # default OR
            return sorted(curr_set | add_set)

        st.session_state["t_contains"] = ",".join(_merge(st.session_state.get("t_contains",""), titles)).strip(",")
        st.session_state["b_contains"] = ",".join(_merge(st.session_state.get("b_contains",""), barcodes)).strip(",")
        st.session_state["pick_titles"] = sorted(_merge(st.session_state.get("pick_titles", []), picks))

        # Clear negatives when applying a positive preset
        st.session_state.pop("neg_title_contains", None)
        st.session_state.pop("neg_barcode_contains", None)
        st.session_state.pop("neg_pick_titles", None)

    # Merge query
    if q:
        q0 = st.session_state.get("q","")
        st.session_state["q"] = (" ".join([q0, q])).strip()

# =========================
# Global filters (sidebar) + presets + custom groups
# =========================
def apply_global_filters(df: pd.DataFrame):
    min_d, max_d = df["dateYMD"].min(), df["dateYMD"].max()
    with st.sidebar:
        st.header("Filters")

        # If a preset was applied, carry its dates into the widget default
        default_dates = st.session_state.get("__preset_dates__")
        if default_dates and isinstance(default_dates, tuple) and len(default_dates)==2:
            dr_default = default_dates
        else:
            dr_default = (
                min_d.date() if pd.notna(min_d) else datetime.today().date(),
                max_d.date() if pd.notna(max_d) else datetime.today().date()
            )

        date_range = st.date_input("Date range", value=dr_default, key="flt_date_range")
        if isinstance(date_range, tuple) and len(date_range)==2:
            st.session_state["__preset_dates__"] = date_range

        # Search box limited to Title/Barcode
        q = st.text_input("Search (Title / Barcode)", value=st.session_state.get("q",""), key="flt_search").strip().lower()
        st.session_state["q"] = q

        # Custom groups (contains…)
        st.markdown("### Custom Groups")
        t_contains = st.text_input("Title contains (comma-sep)", value=st.session_state.get("t_contains",""), key="flt_t_contains")
        b_contains = st.text_input("Barcode contains (comma-sep)", value=st.session_state.get("b_contains",""), key="flt_b_contains")
        st.session_state["t_contains"] = t_contains
        st.session_state["b_contains"] = b_contains

        # Has image tri-state
        has_img_choice = st.selectbox(
            "Has image",
            options=["Both", "Has image", "No image"],
            index={"Both":0,"Has image":1,"No image":2}[st.session_state.get("has_img_choice","Both")] if st.session_state.get("has_img_choice") in {"Both","Has image","No image"} else 0,
            key="flt_has_image"
        )
        st.session_state["has_img_choice"] = has_img_choice

        # Mobile layout toggle
        compact = st.checkbox("Compact layout (mobile)", value=st.session_state.get("compact", True), key="flt_compact")
        st.session_state["compact"] = compact

        # --- Presets (clean layout, multi-apply, OR/AND, negate) ---
        st.markdown("### Presets")

        st.markdown("""
        <style>
        section[data-testid="stSidebar"] .stButton > button { width: 100%; white-space: nowrap; }
        div.chipbar .stButton > button { width: auto; }
        </style>
        """, unsafe_allow_html=True)

        presets = list_presets()
        preset_names = [p["name"] for p in presets]

        chosen_many = st.multiselect(
            "Select preset(s) to apply",
            options=preset_names,
            default=[],
            key="presets_multi"
        )

        combine_mode = st.radio(
            "Combine mode",
            options=["OR", "AND"],
            horizontal=True,
            index=0,
            key="presets_combine"
        )

        negate_apply = st.checkbox(
            "Apply as EXCLUDE (negate match)",
            value=False,
            key="presets_negate"
        )

        if st.button("Apply selected preset(s)", key="presets_apply_btn", use_container_width=True):
            selected = [p for p in presets if p["name"] in chosen_many]
            if not selected:
                st.warning("Pick at least one preset.")
            else:
                title_lists, barcode_lists, picks_titles = [], [], []
                queries = []

                for p in selected:
                    fdef = p["filters"]
                    queries.append(fdef.get("query",""))
                    title_lists.append(fdef.get("title_contains", []))
                    barcode_lists.append(fdef.get("barcode_contains", []))
                    picks_titles.append(fdef.get("pick_titles", []))

                def _combine(list_of_lists, mode="OR"):
                    sets = [set(x) for x in list_of_lists if x]
                    if not sets:
                        return []
                    if mode == "AND":
                        s = sets[0].copy()
                        for z in sets[1:]:
                            s &= z
                        return sorted(s)
                    else:
                        s = set()
                        for z in sets:
                            s |= z
                        return sorted(s)

                def _sanitize(lst):
                    return [s for s in (str(x).strip() for x in (lst or [])) if s]

                merged_titles   = _sanitize(_combine(title_lists,   combine_mode))
                merged_barcodes = _sanitize(_combine(barcode_lists, combine_mode))
                merged_picks    = _sanitize(_combine(picks_titles,  combine_mode))

                if negate_apply:
                    # Store negatives and clear positives to avoid intersecting with prior positive filters
                    st.session_state["neg_title_contains"]   = merged_titles
                    st.session_state["neg_barcode_contains"] = merged_barcodes
                    st.session_state["neg_pick_titles"]      = merged_picks

                    # Clear conflicting positive filters so exclusion behaves intuitively
                    st.session_state["t_contains"] = ""
                    st.session_state["b_contains"] = ""
                    st.session_state["pick_titles"] = []
                else:
                    st.session_state["t_contains"] = ",".join(sorted(set(st.session_state.get("t_contains","").split(",")) | set(merged_titles))).strip(",")
                    st.session_state["b_contains"] = ",".join(sorted(set(st.session_state.get("b_contains","").split(",")) | set(merged_barcodes))).strip(",")
                    st.session_state["pick_titles"] = sorted(set(st.session_state.get("pick_titles", [])) | set(merged_picks))
                    st.session_state.pop("neg_title_contains", None)
                    st.session_state.pop("neg_barcode_contains", None)
                    st.session_state.pop("neg_pick_titles", None)

                q0 = st.session_state.get("q","")
                q_new = " ".join([q0] + [q for q in queries if q]).strip()
                st.session_state["q"] = q_new

                st.success(f"Applied {len(chosen_many)} preset(s) with {combine_mode}{' (negated)' if negate_apply else ''}.")
                st.rerun()

        if st.button("Delete selected preset(s)", key="presets_delete_btn", use_container_width=True):
            if not chosen_many:
                st.warning("Pick at least one preset to delete.")
            else:
                for name in chosen_many:
                    delete_preset(name)
                st.success(f"Deleted {len(chosen_many)} preset(s).")
                st.rerun()

        st.markdown("---")

        preset_name = st.text_input(
            "New preset name",
            value=st.session_state.get("__new_preset_name__",""),
            key="presets_name",
            placeholder="e.g. Sonny / Hipper / Weekend push",
        )
        st.session_state["__new_preset_name__"] = preset_name

        if st.button("Save current filters as preset", key="presets_save_btn", use_container_width=True):
            if preset_name.strip():
                fdef = {
                    "query": st.session_state.get("q",""),
                    "title_contains":  [s.strip() for s in st.session_state.get("t_contains","").split(",") if s.strip()],
                    "barcode_contains":[s.strip() for s in st.session_state.get("b_contains","").split(",") if s.strip()],
                    "has_image": st.session_state.get("has_img_choice","Both"),
                    "pick_titles": st.session_state.get("pick_titles", []),
                }
                save_preset(preset_name.strip(), fdef)
                st.success(f"Saved preset: {preset_name.strip()}")
            else:
                st.warning("Enter a preset name.")

        if st.button("Clear filters", key="flt_clear", use_container_width=True):
            for k in [
                "q","t_contains","b_contains","has_img_choice","pick_titles",
                "__preset_dates__","__new_preset_name__","compact",
                "neg_title_contains","neg_barcode_contains","neg_pick_titles"
            ]:
                st.session_state.pop(k, None)
            st.rerun()

    # Build mask once
    mask = pd.Series(True, index=df.index)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = [pd.to_datetime(d) for d in date_range]
        mask &= df["dateYMD"].between(start, end)

    if q:
        mask &= (
            df.get("title","").astype(str).str.lower().str.contains(q, na=False) |
            df.get("barcode","").astype(str).str.lower().str.contains(q, na=False)
        )

    # Positives: Custom contains (OR within field; AND across fields)
    t_needles = [s.strip() for s in st.session_state.get("t_contains","").split(",") if s.strip()]
    b_needles = [s.strip() for s in st.session_state.get("b_contains","").split(",") if s.strip()]

    if t_needles:
        mask &= contains_any(df.get("title",""), t_needles)
    if b_needles:
        mask &= contains_any(df.get("barcode",""), b_needles)

    # Has image tri-state
    if "hasImage" in df.columns:
        choice = st.session_state.get("has_img_choice","Both")
        if choice == "Has image":
            mask &= df["hasImage"] == True
        elif choice == "No image":
            mask &= (df["hasImage"] == False) | df["hasImage"].isna()

    # Click-to-filter (from top table)
    pick_titles = st.session_state.get("pick_titles", [])
    if pick_titles:
        mask &= df["title"].astype(str).isin(pick_titles)

    # Negative (exclude) presets: union-of-negatives then subtract
    neg_titles   = [s for s in (st.session_state.get("neg_title_contains", []) or []) if str(s).strip()]
    neg_barcodes = [s for s in (st.session_state.get("neg_barcode_contains", []) or []) if str(s).strip()]
    neg_picks    = [s for s in (st.session_state.get("neg_pick_titles", []) or []) if str(s).strip()]

    if neg_titles or neg_barcodes or neg_picks:
        exclude = pd.Series(False, index=df.index)
        if neg_titles:
            exclude |= contains_any(df.get("title",""), neg_titles)
        if neg_barcodes:
            exclude |= contains_any(df.get("barcode",""), neg_barcodes)
        if neg_picks:
            exclude |= df["title"].astype(str).isin([str(x) for x in neg_picks])
        mask &= ~exclude

    f = df[mask].copy()
    return f, pick_titles, st.session_state.get("compact", True)

f, pick_titles, compact = apply_global_filters(df)
if f.empty:
    st.warning("No rows match your filters.")
    st.stop()

# =========================
# Helpers for stock tab
# =========================
def latest_nonnull_by_title(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Get the latest non-null value per title within the filtered window.
    Returns DF with columns ['title', value_col].
    """
    if value_col not in frame.columns:
        return pd.DataFrame(columns=["title", value_col])
    def _last_nonnull(d):
        s = d.sort_values("dateYMD")[value_col]
        s = s.dropna()
        return s.iloc[-1] if len(s) else pd.NA
    out = (
        frame.groupby("title", group_keys=False).apply(_last_nonnull).rename(value_col).reset_index()
    )
    return out

def most_frequent_nonempty(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Mode-ish: most frequent non-empty string per title.
    Returns DF ['title', col]
    """
    if col not in frame.columns:
        return pd.DataFrame(columns=["title", col])
    def _mode(d):
        s = d[col].astype(str).replace({"nan":"", "None":""}).str.strip()
        s = s[s != ""]
        if s.empty: return ""
        return s.value_counts().idxmax()
    out = frame.groupby("title", group_keys=False).apply(_mode).rename(col).reset_index()
    return out

def build_stockout_table(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the current filtered window `frame` (f) to compute:
      - latest inventory per title (Inventory Left)
      - average daily sold per title over the window (Avg Daily Sold)
      - Days to Stock-Out (Inventory / AvgDaily; inf if AvgDaily==0)
      - Projected Stock-Out Date (America/Chicago), ceil(Days) from 'now'
    Returns DataFrame with columns:
      ['title','Inventory Left','Avg Daily Sold','Days to Stock-Out','Projected Stock-Out Date']
    """
    if frame.empty:
        return pd.DataFrame(columns=["title","Inventory Left","Avg Daily Sold","Days to Stock-Out","Projected Stock-Out Date"])

    # Window span in days (inclusive; at least 1)
    dmin = pd.to_datetime(frame["dateYMD"].min())
    dmax = pd.to_datetime(frame["dateYMD"].max())
    if pd.isna(dmin) or pd.isna(dmax):
        days = 1
    else:
        days = max((dmax.normalize() - dmin.normalize()).days + 1, 1)

    # Latest inventory per title (you already have this helper)
    inv_latest = latest_nonnull_by_title(frame, "inventoryNow").rename(columns={"inventoryNow":"Inventory Left"})

    # Average daily sold per title across the window
    sold_total = frame.groupby("title", as_index=False)["qtySold"].sum().rename(columns={"qtySold":"_sold_total"})
    sold_total["Avg Daily Sold"] = sold_total["_sold_total"] / float(days)

    # Merge and compute projections
    out = inv_latest.merge(sold_total[["title","Avg Daily Sold"]], on="title", how="outer")
    out["Inventory Left"] = pd.to_numeric(out["Inventory Left"], errors="coerce")
    out["Avg Daily Sold"] = pd.to_numeric(out["Avg Daily Sold"], errors="coerce").fillna(0.0)

    # Days to stock-out (float; inf when avg is 0 or inv missing)
    def _days_to_so(inv, avg):
        if pd.isna(inv):
            return np.inf
        if avg and avg > 0:
            return float(inv) / float(avg)
        return np.inf

    out["Days to Stock-Out"] = out.apply(lambda r: _days_to_so(r["Inventory Left"], r["Avg Daily Sold"]), axis=1)

    # Projected date (America/Chicago) — ceil to whole days
    now_cst = pd.Timestamp.now(tz="America/Chicago").normalize()
    def _proj_date(days_val):
        if pd.isna(days_val) or not np.isfinite(days_val):
            return ""
        return (now_cst + pd.Timedelta(days=int(np.ceil(days_val)))).date().isoformat()

    out["Projected Stock-Out Date"] = out["Days to Stock-Out"].apply(_proj_date)

    return out


# -------------------------
# Stale inventory metrics (Python-only)
# -------------------------

def _latest_nonnull(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if len(s) else pd.NA


def compute_stale_metrics(
    df: pd.DataFrame,
    key: str = "title",
    window_days: int = 30,
    tz: str = "America/Chicago",
    stale_days_since_last: int = 30,   # flag if no sale for >= this many days
    stale_doc_min: float = 60,         # flag if Days of Cover >= this (using window velocity)
) -> pd.DataFrame:
    """
    Returns one row per `key` with inventory, recent sales/velocity, days of cover,
    days since last sale, age, and a simple StaleFlag.
    Uses only the rows in `df` passed (so share your app filters by passing `f`).
    """
    if df.empty:
        return pd.DataFrame(columns=[
            key,"Inventory Left",f"Sold_{window_days}",f"Velocity_{window_days}",
            f"DaysToStockOut_{window_days}","LastSaleDate","DaysSinceLastSale",
            "FirstSeenDate","AgeDays","StaleFlag"
        ])

    data = df.copy()
    data["dateYMD"] = pd.to_datetime(data["dateYMD"], errors="coerce")
    data["qtySold"] = pd.to_numeric(data.get("qtySold", 0), errors="coerce").fillna(0)
    data["inventoryNow"] = pd.to_numeric(data.get("inventoryNow", pd.NA), errors="coerce")

    today = pd.Timestamp.now(tz=tz).normalize().tz_localize(None)
    start_window = today - pd.Timedelta(days=window_days-1)

    # Latest non-null inventory per key (within the filtered window range provided)
    inv = (
        data.sort_values([key, "dateYMD"])  # ensure temporal order per key
            .groupby(key, as_index=False)
            .agg(**{"Inventory Left": ("inventoryNow", _latest_nonnull)})
    )

    # Sales in window & velocity
    in_window = data[data["dateYMD"] >= start_window]
    sold = (
        in_window.groupby(key, as_index=False)["qtySold"].sum()
                 .rename(columns={"qtySold": f"Sold_{window_days}"})
    )
    # Ensure all keys represented
    sold = inv[[key]].merge(sold, on=key, how="left").fillna({f"Sold_{window_days}": 0})
    sold[f"Velocity_{window_days}"] = sold[f"Sold_{window_days}"] / float(window_days)

    # Days to stock-out (∞ if velocity is 0 or inv missing)
    def _days_to_so(inv_left, vel):
        if pd.isna(inv_left) or vel is None or vel <= 0:
            return np.inf
        return float(inv_left) / float(max(vel, 1e-9))

    # merge inv into sold to compute days to SO
    tmp = sold.merge(inv, on=key, how="left")
    tmp[f"DaysToStockOut_{window_days}"] = tmp.apply(
        lambda r: _days_to_so(r.get("Inventory Left"), r.get(f"Velocity_{window_days}")), axis=1
    )

    # Last sale date & days since
    last_sale = (data[data["qtySold"] > 0]
                 .groupby(key, as_index=False)["dateYMD"].max()
                 .rename(columns={"dateYMD": "LastSaleDate"}))
    last_sale["DaysSinceLastSale"] = (today - pd.to_datetime(last_sale["LastSaleDate"])) .dt.days

    # First seen date (any presence), and age
    first_seen = (data.groupby(key, as_index=False)["dateYMD"].min()
                  .rename(columns={"dateYMD": "FirstSeenDate"}))
    first_seen["AgeDays"] = (today - pd.to_datetime(first_seen["FirstSeenDate"])) .dt.days

    # Assemble
    out = (tmp.merge(last_sale[[key, "LastSaleDate", "DaysSinceLastSale"]], on=key, how="left")
              .merge(first_seen[[key, "FirstSeenDate", "AgeDays"]], on=key, how="left"))

    # Stale flag (tweak thresholds)
    doc_col = f"DaysToStockOut_{window_days}"
    out["StaleFlag"] = (
        (out["DaysSinceLastSale"].fillna(10**9) >= stale_days_since_last) |
        (pd.to_numeric(out[doc_col], errors="coerce").fillna(np.inf) >= stale_doc_min)
    )

    cols = [
        key, "Inventory Left",
        f"Sold_{window_days}", f"Velocity_{window_days}", doc_col,
        "LastSaleDate","DaysSinceLastSale","FirstSeenDate","AgeDays","StaleFlag"
    ]
    return out[cols]


# =========================
# Quick Preset Chips (fast apply)
# =========================
presets_all = list_presets()
# Order: favorites first (name begins with '*'), then alphabetically
fav = [p for p in presets_all if str(p["name"]).strip().startswith("*")]
regular = [p for p in presets_all if p not in fav]
chips = (fav + regular)[:12]

if chips:
    st.markdown("<div class='sticky-chipbar'>", unsafe_allow_html=True)
    cols = st.columns(len(chips) + 1)
    for i, p in enumerate(chips):
        label = p["name"]
        with cols[i]:
            st.button(label, key=f"chip_{i}", help="Apply preset", use_container_width=False)
            if st.session_state.get(f"chip_{i}"):
                apply_preset_to_session(p["filters"], negate=False, combine_mode="OR")
                st.rerun()
    # Clear Filters pill
    with cols[-1]:
        if st.button("Clear filters", key="chip_clear", help="Reset all filters", use_container_width=False):
            for k in [
                "q","t_contains","b_contains","has_img_choice","pick_titles",
                "__preset_dates__","__new_preset_name__","compact",
                "neg_title_contains","neg_barcode_contains","neg_pick_titles"
            ]:
                st.session_state.pop(k, None)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📦 Stock Views", "📈 Pareto & Stock-Out", "🧊 Stale Inventory"])

with tab1:
    # ---------------------------
    # KPIs — Revenue → COGS → Gross Profit → Gross Margin
    # ---------------------------
    total_net  = float(f.get("netSales", pd.Series(dtype=float)).sum())
    total_cogs = float(f.get("totalCost", pd.Series(dtype=float)).sum()) if "totalCost" in f.columns else 0.0
    if "grossProfit" in f.columns:
        total_gp = float(f["grossProfit"].sum())
    else:
        total_gp = total_net - total_cogs
    overall_margin = (total_gp / total_net) if total_net > 0 else 0.0

    if not compact:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Revenue (Net)", f"${total_net:,.2f}")
        with c2: st.metric("COGS", f"${total_cogs:,.2f}")
        with c3: st.metric("Gross Profit", f"${total_gp:,.2f}")
        with c4: st.metric("Gross Margin", f"{overall_margin:.2%}")
    else:
        # Mobile: 2 columns per row
        r1c1, r1c2 = st.columns(2)
        with r1c1: st.metric("Revenue (Net)", f"${total_net:,.2f}")
        with r1c2: st.metric("COGS", f"${total_cogs:,.2f}")
        r2c1, r2c2 = st.columns(2)
        with r2c1: st.metric("Gross Profit", f"${total_gp:,.2f}")
        with r2c2: st.metric("Gross Margin", f"{overall_margin:.2%}")

    st.divider()

    # ---------------------------
    # TOP SUMMARY TABLE (click-to-filter) — with Inventory Left
    # ---------------------------
    st.subheader("Top Products (click rows → filter)")

    by_title = (
        f.groupby("title", as_index=False)
         .agg(qtySold=("qtySold","sum"),
              netSales=("netSales","sum"))
    )
    inv_latest_series = latest_nonnull_by_title(f, "inventoryNow").rename(columns={"inventoryNow":"Inventory Left"})
    top_join = by_title.merge(inv_latest_series, on="title", how="left")
    top_join = top_join.sort_values(["netSales","qtySold"], ascending=[False, False])

    top_df = top_join.rename(columns={
        "title":"Product",
        "qtySold":"Qty Sold",
        "netSales":"Revenue",
    })

    gb_top = GridOptionsBuilder.from_dataframe(top_df)
    gb_top.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb_top.configure_column("Revenue", valueFormatter="value != null ? Intl.NumberFormat().format(value) : ''")
    gb_top.configure_column("Qty Sold", valueFormatter="value != null ? Math.round(value) : ''")
    gb_top.configure_column("Inventory Left", valueFormatter="value != null ? Math.round(value) : ''")
    gb_top.configure_grid_options(domLayout="normal")
    top_opts = gb_top.build()
    top_opts["sortModel"] = [
        {"colId":"Revenue","sort":"desc"},
        {"colId":"Qty Sold","sort":"desc"},
    ]

    grid_top = AgGrid(
        top_df,
        gridOptions=top_opts,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=300 if not compact else 380,
        fit_columns_on_grid_load=True,
        theme="alpine",
    )
    sel_top = grid_top["selected_rows"]

    cft1, cft2 = st.columns(2)
    with cft1:
        if st.button("Filter to selected Products (Top Table)", key="btn_filter_selected_titles"):
            picked_titles = sorted({r.get("Product") for r in sel_top if r.get("Product")})
            st.session_state["pick_titles"] = picked_titles
            st.rerun()
    with cft2:
        if st.button("Clear product/title filter", key="btn_clear_selected_titles"):
            st.session_state["pick_titles"] = []
            st.rerun()

    st.divider()

    # ---------------------------
    # Robust daily time series (never blank) + data labels
    # ---------------------------
    with st.sidebar:
        chart_style = st.selectbox("Time-series style", ["Line", "Bar"], index=0, key="ts_style")

    def build_daily_timeseries(f_sel: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
        if f_sel.empty:
            start = pd.to_datetime(df_all["dateYMD"].min())
            end   = pd.to_datetime(df_all["dateYMD"].max())
        else:
            start = pd.to_datetime(f_sel["dateYMD"].min())
            end   = pd.to_datetime(f_sel["dateYMD"].max())

        if pd.isna(start) or pd.isna(end):
            base_day = pd.Timestamp.today().normalize()
            idx = pd.date_range(start=base_day, end=base_day, freq="D")
        else:
            idx = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")

        daily_all = (
            df_all[df_all["dateYMD"].between(idx.min(), idx.max())]
            .groupby("dateYMD", as_index=False)
            .agg(net_total=("netSales", "sum"))
        )
        daily_all = (
            daily_all.set_index("dateYMD")
                     .reindex(idx)
                     .fillna(0.0)
                     .rename_axis("dateYMD")
                     .reset_index()
        )

        # Selected series: include gross profit if available so we can compute daily margin
        if not f_sel.empty:
            agg_dict = {
                "net": ("netSales", "sum"),
                "qty": ("qtySold", "sum"),
                "inv": ("inventoryNow", "sum"),
            }
            if "grossProfit" in f_sel.columns:
                agg_dict["gp"] = ("grossProfit", "sum")
            else:
                # fallback: try compute gp from net & totalCost when available
                if ("netSales" in f_sel.columns) and ("totalCost" in f_sel.columns):
                    f_sel = f_sel.assign(__gp__=pd.to_numeric(f_sel["netSales"], errors="coerce") - pd.to_numeric(f_sel["totalCost"], errors="coerce"))
                    agg_dict["gp"] = ("__gp__", "sum")
            daily_sel = f_sel.groupby("dateYMD", as_index=False).agg(**agg_dict)
        else:
            cols = {"dateYMD": idx, "net": [0.0]*len(idx), "qty": [0.0]*len(idx), "inv": [0.0]*len(idx)}
            cols["gp"] = [0.0]*len(idx)
            daily_sel = pd.DataFrame(cols)

        daily = pd.merge(daily_all, daily_sel, on="dateYMD", how="left")
        for c in ["net", "qty", "inv", "net_total", "gp"]:
            if c not in daily:
                daily[c] = 0.0
        daily[["net","qty","inv","net_total","gp"]] = daily[["net","qty","inv","net_total","gp"]].fillna(0.0)

        daily["pct_of_day_revenue"] = daily.apply(
            lambda r: (r["net"] / r["net_total"] * 100.0) if r["net_total"] > 0 else 0.0,
            axis=1
        )

        # Weighted average daily gross margin (%): grossProfit_sum / netSales_sum * 100
        daily["avg_margin_pct"] = daily.apply(
            lambda r: (r["gp"] / r["net"] * 100.0) if r["net"] > 0 else 0.0,
            axis=1
        )

        daily = daily.sort_values("dateYMD")
        daily["date_label"] = daily["dateYMD"].dt.strftime("%b %d, %Y")
        return daily

    daily = build_daily_timeseries(f, df)

    def apply_time_axis(fig):
        fig.update_xaxes(tickformat="%b %d", dtick="D7", showgrid=False)
        fig.update_yaxes(showgrid=True)
        fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
        return fig

    def make_series(fig_kind: str, data, x, y, title, hover_fmt=":.2f", is_percent=False, is_count=False):
        if fig_kind == "Bar":
            fig = px.bar(
                data, x=x, y=y,
                hover_data={"date_label": True, y: hover_fmt, x: False},
                title=title,
                text=y
            )
            if is_percent:
                fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside", cliponaxis=False)
            elif is_count:
                fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
            else:
                fig.update_traces(texttemplate="%{y:,.2f}", textposition="outside", cliponaxis=False)
        else:
            fig = px.line(
                data, x=x, y=y,
                hover_data={"date_label": True, y: hover_fmt, x: False},
                title=title
            )
            if is_percent:
                fig.update_traces(mode="lines+markers+text", texttemplate="%{y:.2f}%", textposition="top center")
            elif is_count:
                fig.update_traces(mode="lines+markers+text", texttemplate="%{y:,.0f}", textposition="top center")
            else:
                fig.update_traces(mode="lines+markers+text", texttemplate="%{y:,.2f}", textposition="top center")

        fig = apply_time_axis(fig)
        if is_percent:
            fig.update_yaxes(ticksuffix="%", rangemode="tozero")
        return fig

    if compact:
        st.plotly_chart(make_series(chart_style, daily, "dateYMD", "net",
                                    "Daily Revenue (Net)", hover_fmt=":.2f"), use_container_width=True)
        st.plotly_chart(make_series(chart_style, daily, "dateYMD", "qty",
                                    "Daily Quantity Sold", hover_fmt=":,.0f", is_count=True), use_container_width=True)
        st.plotly_chart(make_series(chart_style, daily, "dateYMD", "inv",
                                    "Inventory Left (sum of selected)", hover_fmt=":,.0f", is_count=True), use_container_width=True)
        st.plotly_chart(make_series(chart_style, daily, "dateYMD", "pct_of_day_revenue",
                                    "% of Daily Revenue (Selected vs Total)", hover_fmt=":.2f", is_percent=True), use_container_width=True)
        st.plotly_chart(
            make_series(
                chart_style, daily, "dateYMD", "avg_margin_pct",
                "Avg Gross Margin (Daily)", hover_fmt=":.2f", is_percent=True
            ),
            use_container_width=True
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_series(chart_style, daily, "dateYMD", "net",
                                        "Daily Revenue (Net)", hover_fmt=":.2f"), use_container_width=True)
        with c2:
            st.plotly_chart(make_series(chart_style, daily, "dateYMD", "qty",
                                        "Daily Quantity Sold", hover_fmt=":,.0f", is_count=True), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(make_series(chart_style, daily, "dateYMD", "inv",
                                        "Inventory Left (sum of selected)", hover_fmt=":,.0f", is_count=True), use_container_width=True)
        with c4:
            st.plotly_chart(make_series(chart_style, daily, "dateYMD", "pct_of_day_revenue",
                                        "% of Daily Revenue (Selected vs Total)", hover_fmt=":.2f", is_percent=True), use_container_width=True)
        st.plotly_chart(
            make_series(
                chart_style, daily, "dateYMD", "avg_margin_pct",
                "Avg Gross Margin (Daily)", hover_fmt=":.2f", is_percent=True
            ),
            use_container_width=True
        )

    st.divider()

    # ---------------------------
    # Rankings (Top 10) — Qty, Revenue, % of Revenue
    # ---------------------------
    st.subheader("Top 10 Rankings")

    # Aggregate per title on the filtered set
    rank_agg = (
        f.groupby("title", as_index=False)
         .agg(qtySold=("qtySold","sum"),
              netSales=("netSales","sum"),
              grossSales=("grossSales","sum"),
              grossProfit=("grossProfit","sum") if "grossProfit" in f.columns else ("netSales","sum"))
    )
    # Guard against empty totals
    total_net_rank = float(rank_agg["netSales"].sum()) if not rank_agg.empty else 0.0
    if total_net_rank == 0:
        rank_agg["pctRevenue"] = 0.0
    else:
        rank_agg["pctRevenue"] = (rank_agg["netSales"] / total_net_rank) * 100.0

    # Add Margin % per title
    if "grossProfit" in rank_agg.columns:
        rank_agg["Margin %"] = np.where(rank_agg["netSales"] > 0, (rank_agg["grossProfit"] / rank_agg["netSales"]) * 100.0, 0.0)
    else:
        rank_agg["Margin %"] = 0.0

    # Build three top-10 tables
    top_qty = (
        rank_agg.sort_values(["qtySold","netSales"], ascending=[False, False])
                .head(10)[["title","qtySold"]]
                .rename(columns={"title":"Product","qtySold":"Qty Sold"})
    )

    top_rev = (
        rank_agg.sort_values(["netSales","grossSales"], ascending=[False, False])
                .head(10)[["title","grossSales","netSales","grossProfit","Margin %"]]
                .rename(columns={"title":"Product","grossSales":"Gross","netSales":"Net","grossProfit":"Gross Profit"})
    )

    top_pct = (
        rank_agg.sort_values(["pctRevenue","netSales"], ascending=[False, False])
                .head(10)[["title","pctRevenue"]]
                .rename(columns={"title":"Product","pctRevenue":"% of Revenue"})
    )

    # Display in 3 columns
    colq, colr, colp = st.columns(3)
    with colq:
        st.markdown("**Top by Qty Sold**")
        st.dataframe(
            top_qty,
            use_container_width=True,
            hide_index=True,
            column_config={"Qty Sold": cc.NumberColumn(format="%d")}
        )
    with colr:
        st.markdown("**Top by Revenue**")
        st.dataframe(
            top_rev,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Gross": cc.NumberColumn(format="$%.2f"),
                "Net":   cc.NumberColumn(format="$%.2f"),
                "Gross Profit": cc.NumberColumn(format="$%.2f"),
                "Margin %": cc.NumberColumn(format="%.2f%%"),
            }
        )
    with colp:
        st.markdown("**Top by % of Total Revenue**")
        st.dataframe(
            top_pct,
            use_container_width=True,
            hide_index=True,
            column_config={"% of Revenue": cc.NumberColumn(format="%.2f%%")}
        )

    st.divider()

    # ---------------------------
    # BOTTOM DETAIL TABLE (view-only)
    # ---------------------------
    st.subheader("Detail")

    f_sorted = f.sort_values(["netSales","qtySold"], ascending=[False, False])

    cols_order = [
        "date_display", "title", "barcode", "barcodesDistinct", "sku",
        "ordersCount","qtySold","grossSales","netSales","avgUnitNet","avgUnitGross",
        "discountRate",
        "unitCost","totalCost","grossProfit","grossMarginRate",
        "inventoryNow","hasImage"
    ]
    cols_order = [c for c in cols_order if c in f_sorted.columns]
    display = (
        f_sorted[cols_order]
        .rename(columns={
            "date_display":"Date",
            "title":"Product",
            "barcode":"Barcode",
            "barcodesDistinct":"Barcodes (Distinct)",
            "sku":"SKU",
            "ordersCount":"Orders",
            "qtySold":"Qty",
            "grossSales":"Gross",
            "netSales":"Net",
            "avgUnitNet":"Avg Unit Net",
            "avgUnitGross":"Avg Unit Gross",
            "discountRate":"Discount",
            "unitCost":"Unit Cost",
            "totalCost":"Total Cost",
            "grossProfit":"Gross Profit",
            "grossMarginRate":"Gross Margin",
            "inventoryNow":"Inventory",
            "hasImage":"Has Image"
        })
    )

    for c in ["SKU","Barcode","Barcodes (Distinct)","Product"]:
        if c in display.columns:
            display[c] = display[c].astype(str)

    colcfg = {}
    if "Gross" in display.columns:         colcfg["Gross"] = cc.NumberColumn(format="$%.2f")
    if "Net" in display.columns:           colcfg["Net"] = cc.NumberColumn(format="$%.2f")
    if "Avg Unit Net" in display.columns:  colcfg["Avg Unit Net"] = cc.NumberColumn(format="$%.4f")
    if "Avg Unit Gross" in display.columns:colcfg["Avg Unit Gross"] = cc.NumberColumn(format="$%.4f")
    if "Discount" in display.columns:      colcfg["Discount"] = cc.NumberColumn(format="%.2f%%")
    if "Qty" in display.columns:           colcfg["Qty"] = cc.NumberColumn(format="%d")
    if "Orders" in display.columns:        colcfg["Orders"] = cc.NumberColumn(format="%d")
    if "Inventory" in display.columns:     colcfg["Inventory"] = cc.NumberColumn(format="%d")
    if "Unit Cost" in display.columns:     colcfg["Unit Cost"] = cc.NumberColumn(format="$%.2f")
    if "Total Cost" in display.columns:    colcfg["Total Cost"] = cc.NumberColumn(format="$%.2f")
    if "Gross Profit" in display.columns:  colcfg["Gross Profit"] = cc.NumberColumn(format="$%.2f")
    if "Gross Margin" in display.columns:  colcfg["Gross Margin"] = cc.NumberColumn(format="%.2f%%")

    st.dataframe(
        display,
        use_container_width=True,
        height=520 if not compact else 680,
        column_config=colcfg,
        hide_index=True,
    )

with tab2:
    st.subheader("Stock Views")

    # Build per-title aggregates used by all three tables
    qty_by_title = f.groupby("title", as_index=False)["qtySold"].sum()

    inv_latest = latest_nonnull_by_title(f, "inventoryNow").rename(columns={"inventoryNow":"Inventory Left"})
    price_latest = latest_nonnull_by_title(f, "avgUnitNet").rename(columns={"avgUnitNet":"Item Price"})
    # fallback to gross if net missing
    missing_price = price_latest["Item Price"].isna()
    if missing_price.any():
        gross_latest = latest_nonnull_by_title(f, "avgUnitGross").rename(columns={"avgUnitGross":"Item Price (Gross)"})
        price_latest = price_latest.merge(gross_latest, on="title", how="left")
        price_latest["Item Price"] = price_latest["Item Price"].fillna(price_latest["Item Price (Gross)"])
        price_latest = price_latest[["title","Item Price"]]

    primary_barcode = most_frequent_nonempty(f, "barcode").rename(columns={"barcode":"Barcode"})
    primary_sku     = most_frequent_nonempty(f, "sku").rename(columns={"sku":"SKU"})

    base = qty_by_title.merge(inv_latest, on="title", how="left") \
                       .merge(price_latest, on="title", how="left") \
                       .merge(primary_barcode, on="title", how="left") \
                       .merge(primary_sku, on="title", how="left")

    base = base.rename(columns={"title":"Product"})
    # ---- 1) Shopping list
    shop_cols = ["Product","qtySold","Inventory Left","Item Price","Barcode","SKU"]
    shop = base[shop_cols].copy().rename(columns={"qtySold":"Qty Sold"})

    st.markdown("### Shopping List")
    # Optional filter: exclude low-velocity cheap items (prettified/aligned)
    # --- compact CSS for nicer row alignment in the sidebar/content ---
    st.markdown(
        """
        <style>
        /* tighten spacing inside this section */
        .shop-controls .stMarkdown { margin-bottom: 0.25rem; }
        .shop-controls [data-testid="stHorizontalBlock"] > div { padding-right: .5rem; }
        .shop-controls label { font-size: 0.875rem; }
        .shop-controls .stSlider > div[data-baseweb="slider"] { margin-top: .35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("<div class='shop-controls'>", unsafe_allow_html=True)

        hdr1, hdr2, hdr3 = st.columns([1.1, 0.9, 1.6])
        with hdr1: st.markdown("**Low-velocity toggle**")
        with hdr2: st.markdown("**Price <**")
        with hdr3: st.markdown("**Sold % <**")

        c1, c2, c3 = st.columns([1.1, 0.9, 1.6])
        with c1:
            st.toggle(
                "Exclude low-velocity cheap items",
                value=st.session_state.get("shop_exclude_low_vel", False),
                key="shop_exclude_low_vel"
            )
        with c2:
            st.number_input(
                label="Price <",
                min_value=0.0,
                value=float(st.session_state.get("shop_price_thr", 10.0)),
                step=0.5,
                key="shop_price_thr",
                label_visibility="collapsed",
                help="Exclude only if price is below this value."
            )
        with c3:
            st.slider(
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

    # Compute sell-through % = Qty / (Qty + Inventory) * 100
    denom = (shop["Qty Sold"].fillna(0) + shop["Inventory Left"].fillna(0))
    shop["Sold %"] = (shop["Qty Sold"].fillna(0) / denom.where(denom > 0, 1)) * 100

    # Apply optional exclusion: price & sold% thresholds
    if st.session_state.get("shop_exclude_low_vel", False):
        pthr = float(st.session_state.get("shop_price_thr", 10.0))
        rthr = float(st.session_state.get("shop_pct_thr", 10.0))
        keep_mask = ~((shop["Item Price"].fillna(0) < pthr) & (shop["Sold %"] < rthr))
        shop = shop[keep_mask]

    shop = shop.sort_values(["Qty Sold","Item Price"], ascending=[False, True])

    st.dataframe(
        shop,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qty Sold": cc.NumberColumn(format="%d"),
            "Inventory Left": cc.NumberColumn(format="%d"),
            "Item Price": cc.NumberColumn(format="$%.2f"),
            "Sold %": cc.NumberColumn(format="%.1f%%"),
        }
    )

    st.divider()

    # ---- 2) Out of stock (Inventory Left <= 0 or NA treated as 0?)
    # Treat explicit 0 as OOS; leave NaN out of OOS (unknown)
    oos = shop[(shop["Inventory Left"].fillna(0) <= 0)].copy()
    oos = oos.sort_values(["Item Price","Product"], ascending=[False, True])

    st.markdown("### Out of Stock")
    st.dataframe(
        oos,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qty Sold": cc.NumberColumn(format="%d"),
            "Inventory Left": cc.NumberColumn(format="%d"),
            "Item Price": cc.NumberColumn(format="$%.2f"),
        }
    )

    st.divider()

    # ---- 3) Low on stock (tiered)
    def threshold_for_price(p):
        if pd.isna(p):  # unknown price, be conservative? pick mid-tier threshold
            return 3
        if p > 30:        return 2
        if 20 <= p <= 30: return 3
        return 5

    thr = shop["Item Price"].apply(threshold_for_price)
    low = shop[(shop["Inventory Left"].fillna(0) > 0) & (shop["Inventory Left"] <= thr)].copy()
    low = low.assign(Threshold=thr[low.index].values)
    low = low.sort_values(["Inventory Left","Item Price"], ascending=[True, False])

    st.markdown("### Low on Stock (tiered)")
    st.dataframe(
        low[["Product","Qty Sold","Inventory Left","Threshold","Item Price","Barcode","SKU"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qty Sold": cc.NumberColumn(format="%d"),
            "Inventory Left": cc.NumberColumn(format="%d"),
            "Threshold": cc.NumberColumn(format="%d"),
            "Item Price": cc.NumberColumn(format="$%.2f"),
        }
    )

    st.divider()

    # ---- Missing COGS (totalCost == 0 or NaN), placed at the bottom
    st.markdown("### Missing COGS")

    # Aggregate COGS and revenue per title over the filtered window
    cogs_agg = (
        f.groupby("title", as_index=False)
         .agg(Total_COGS=("totalCost", "sum"),
              Revenue=("netSales", "sum"))
    )

    # Keep rows where total COGS is zero or missing
    missing_cogs = cogs_agg[cogs_agg["Total_COGS"].fillna(0) == 0].copy()

    if missing_cogs.empty:
        st.info("✅ No items with missing COGS found in the current filter window.")
    else:
        # Merge in the same identifiers we use in the shopping list
        base_for_merge = base.rename(columns={"Product": "title"})
        mc = (missing_cogs.merge(base_for_merge, on="title", how="left")
                             .rename(columns={"title": "Product"}))

        # Ensure nice column order and names
        mc_display = mc[["Product",
                         "qtySold",
                         "Inventory Left",
                         "Item Price",
                         "Barcode",
                         "SKU",
                         "Revenue",
                         "Total_COGS"]].rename(columns={
                             "qtySold": "Qty Sold",
                             "Total_COGS": "COGS (Total)"
                         })

        # Sort by Revenue desc, then Product
        mc_display = mc_display.sort_values(["Revenue", "Product"], ascending=[False, True])

        st.dataframe(
            mc_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Qty Sold": cc.NumberColumn(format="%d"),
                "Inventory Left": cc.NumberColumn(format="%d"),
                "Item Price": cc.NumberColumn(format="$%.2f"),
                "Revenue": cc.NumberColumn(format="$%.2f"),
                "COGS (Total)": cc.NumberColumn(format="$%.2f"),
            }
        )
with tab3:
    st.subheader("Pareto (Revenue) & Projected Stock-Out Dates")

    # ---------- Pareto Chart (by Net Revenue) ----------
    pareto = (
        f.groupby("title", as_index=False)
         .agg(netSales=("netSales","sum"))
         .sort_values("netSales", ascending=False)
    )
    total_net = float(pareto["netSales"].sum()) if not pareto.empty else 0.0
    if total_net > 0:
        pareto["Share %"] = (pareto["netSales"] / total_net) * 100.0
        pareto["Cum %"] = pareto["Share %"].cumsum()
    else:
        pareto["Share %"] = 0.0
        pareto["Cum %"] = 0.0

    # Dual-axis Pareto: bars = revenue, line = cumulative %
    fig_p = make_subplots(specs=[[{"secondary_y": True}]])
    fig_p.add_trace(
        go.Bar(x=pareto["title"], y=pareto["netSales"], name="Revenue"),
        secondary_y=False
    )
    fig_p.add_trace(
        go.Scatter(x=pareto["title"], y=pareto["Cum %"], name="Cumulative %", mode="lines+markers"),
        secondary_y=True
    )
    # 80% reference line
    fig_p.add_hline(y=80, line_dash="dash", line_color="red", secondary_y=True)

    fig_p.update_layout(
        title="Pareto of Net Revenue by Product",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Product (sorted by revenue)",
        yaxis_title="Revenue",
        hovermode="x unified"
    )
    fig_p.update_yaxes(title_text="Cumulative %", secondary_y=True, ticksuffix="%")
    st.plotly_chart(fig_p, use_container_width=True)

    st.divider()

    # ---------- Projected Stock-Outs ----------
    st.markdown("### Projected Stock-Out Dates")

    # Filter: show items that will stock out within X days from today
    so_days_thr = st.slider(
        "Show items projecting stock-out within (days)",
        min_value=0, max_value=120, value=14, step=1, key="so_days_thr",
        help="Only show products whose projected stock-out date is within X days from today (America/Chicago)."
    )

    so = build_stockout_table(f)

    # Add common identifiers for context
    barcode_m = most_frequent_nonempty(f, "barcode").rename(columns={"barcode":"Barcode"})
    sku_m     = most_frequent_nonempty(f, "sku").rename(columns={"sku":"SKU"})
    so = (so.merge(barcode_m, on="title", how="left")
             .merge(sku_m, on="title", how="left"))

    so = so.rename(columns={"title":"Product"})

    # Compute days-until from today (America/Chicago) and filter by threshold
    _today_cst = pd.Timestamp.now(tz="America/Chicago").normalize().tz_localize(None)
    so["_proj_ts"] = pd.to_datetime(so["Projected Stock-Out Date"], errors="coerce")
    so["Days Until"] = (so["_proj_ts"] - _today_cst).dt.days

    # Keep only rows with a valid projected date within the chosen window (>=0 and <= threshold)
    if so_days_thr is not None:
        so = so[(so["Days Until"].notna()) & (so["Days Until"] >= 0) & (so["Days Until"] <= so_days_thr)]

    # Order: soonest stock-out first using Days Until (fallback to Days to Stock-Out)
    if "Days Until" in so.columns and so["Days Until"].notna().any():
        so = so.sort_values(["Days Until"], ascending=[True])
    else:
        days_num = pd.to_numeric(so["Days to Stock-Out"], errors="coerce")
        so["__is_inf__"] = ~np.isfinite(days_num)
        so["__days__"] = days_num
        so = so.sort_values(["__is_inf__", "__days__"], ascending=[True, True]).drop(columns=["__is_inf__","__days__"])

    st.dataframe(
        so[["Product","Inventory Left","Avg Daily Sold","Days to Stock-Out","Days Until","Projected Stock-Out Date","Barcode","SKU"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Inventory Left": cc.NumberColumn(format="%d"),
            "Avg Daily Sold": cc.NumberColumn(format="%.2f"),
            "Days to Stock-Out": cc.NumberColumn(format="%.1f"),
            "Days Until": cc.NumberColumn(format="%d"),
        }
    )


with tab4:
    st.subheader("Stale Inventory (Python)")

    # Controls
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        win = st.slider("Window (days)", 7, 90, 30, step=1, key="stale_win")
    with col_b:
        no_sale_days = st.number_input("Flag if no sale for ≥ (days)", 7, 180, 30, step=1, key="stale_nosale")
    with col_c:
        doc_min = st.number_input("Flag if Days of Cover ≥", 15, 365, 60, step=5, key="stale_doc")

    # Compute using the filtered data `f` so it shares all filters
    stale = compute_stale_metrics(
        df=f,
        key="title",
        window_days=int(win),
        tz="America/Chicago",
        stale_days_since_last=int(no_sale_days),
        stale_doc_min=float(doc_min),
    )

    # Add handy identifiers
    bc = most_frequent_nonempty(f, "barcode").rename(columns={"barcode":"Barcode"})
    sku = most_frequent_nonempty(f, "sku").rename(columns={"sku":"SKU"})
    stale = (stale.merge(bc, left_on="title", right_on="title", how="left")
                   .merge(sku, left_on="title", right_on="title", how="left"))

    # Sort: flagged first, then longest since sale, then largest days of cover
    doc_col = f"DaysToStockOut_{win}"
    stale = stale.sort_values(["StaleFlag","DaysSinceLastSale", doc_col], ascending=[False, False, False])

    # Pretty column names
    stale = stale.rename(columns={
        "title":"Product",
        f"Sold_{win}": f"Sold_{win}",
        f"Velocity_{win}": f"Velocity_{win}",
        doc_col: doc_col,
    })

    st.dataframe(
        stale[["Product","Inventory Left", f"Sold_{win}", f"Velocity_{win}", doc_col,
               "LastSaleDate","DaysSinceLastSale","FirstSeenDate","AgeDays","StaleFlag","Barcode","SKU"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Inventory Left": cc.NumberColumn(format="%d"),
            f"Sold_{win}": cc.NumberColumn(format="%d"),
            f"Velocity_{win}": cc.NumberColumn(format="%.2f"),
            doc_col: cc.NumberColumn(format="%.1f"),
            "DaysSinceLastSale": cc.NumberColumn(format="%d"),
            "AgeDays": cc.NumberColumn(format="%d"),
        }
    )
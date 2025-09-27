import json
import base64
import streamlit as st
import gspread
from google.oauth2 import service_account
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

from data_processor import coerce_numeric_cols
from config import get_spreadsheet_config, CACHE_TTL


@st.cache_resource
def get_google_client():
    """Initialize and return Google Sheets client."""
    config = get_spreadsheet_config()

    # Try split JSON variables format
    if all(key in st.secrets for key in ["GCP_TYPE", "GCP_PROJECT_ID", "GCP_PRIVATE_KEY", "GCP_CLIENT_EMAIL"]):
        service_account_info = {
            "type": st.secrets["GCP_TYPE"],
            "project_id": st.secrets["GCP_PROJECT_ID"],
            "private_key_id": st.secrets["GCP_PRIVATE_KEY_ID"],
            "private_key": st.secrets["GCP_PRIVATE_KEY"].replace("\\n", "\n"),
            "client_email": st.secrets["GCP_CLIENT_EMAIL"],
            "client_id": st.secrets["GCP_CLIENT_ID"],
            "auth_uri": st.secrets["GCP_AUTH_URI"],
            "token_uri": st.secrets["GCP_TOKEN_URI"],
            "auth_provider_x509_cert_url": st.secrets["GCP_AUTH_PROVIDER_X509_CERT_URL"],
            "client_x509_cert_url": st.secrets["GCP_CLIENT_X509_CERT_URL"],
            "universe_domain": st.secrets.get("GCP_UNIVERSE_DOMAIN", "googleapis.com")
        }
        creds = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=config['scopes']
        )
    # Try structured secrets format
    elif "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=config['scopes']
        )
    # Try base64 encoded JSON format
    elif "GOOGLE_SERVICE_ACCOUNT_B64" in st.secrets:
        sa_b64 = st.secrets["GOOGLE_SERVICE_ACCOUNT_B64"]
        sa_json = base64.b64decode(sa_b64).decode('utf-8')
        creds = service_account.Credentials.from_service_account_info(
            json.loads(sa_json),
            scopes=config['scopes']
        )
    # Fallback to old formats
    elif "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
        sa_json = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
        creds = service_account.Credentials.from_service_account_info(
            json.loads(sa_json),
            scopes=config['scopes']
        )
    elif "GOOGLE_SERVICE_ACCOUNT_JSON_FILE" in st.secrets:
        sa_file = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON_FILE"]
        creds = service_account.Credentials.from_service_account_file(
            sa_file,
            scopes=config['scopes']
        )
    else:
        raise ValueError("No service account credentials provided in secrets.toml")

    return gspread.authorize(creds)


@st.cache_data(ttl=CACHE_TTL["data"])
def load_main_data() -> pd.DataFrame:
    """Load and process main data from Google Sheets."""
    client = get_google_client()
    config = get_spreadsheet_config()

    try:
        spreadsheet = client.open_by_key(config['spreadsheet_id'])
        worksheet = spreadsheet.worksheet(config['worksheet_name'])
        rows = worksheet.get_all_records()

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        return coerce_numeric_cols(df)

    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        raise


class PresetManager:
    """Manages preset operations for the dashboard."""

    def __init__(self):
        self.config = get_spreadsheet_config()

    def _ensure_presets_sheet(self, spreadsheet):
        """Ensure the presets worksheet exists."""
        try:
            worksheet = spreadsheet.worksheet(self.config['presets_sheet_name'])
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                self.config['presets_sheet_name'],
                rows=1000,
                cols=3
            )
            worksheet.update("A1:C1", [["name", "updatedAt", "filters_json"]])

        return worksheet

    @st.cache_data(ttl=CACHE_TTL["presets"])
    def list_presets(_self) -> List[Dict[str, Any]]:
        """Return list of available presets."""
        client = get_google_client()
        spreadsheet = client.open_by_key(_self.config['spreadsheet_id'])
        worksheet = _self._ensure_presets_sheet(spreadsheet)

        rows = worksheet.get_all_records()
        presets = []

        for row in rows:
            try:
                filters = json.loads(row.get("filters_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                filters = {}

            if row.get("name"):
                presets.append({
                    "name": row["name"],
                    "filters": filters
                })

        return sorted(presets, key=lambda x: x["name"].lower())

    def save_preset(self, name: str, filters: Dict[str, Any]) -> None:
        """Save a preset to Google Sheets."""
        client = get_google_client()
        spreadsheet = client.open_by_key(self.config['spreadsheet_id'])
        worksheet = self._ensure_presets_sheet(spreadsheet)

        rows = worksheet.get_all_records()
        payload = [
            name,
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            json.dumps(filters, ensure_ascii=False)
        ]

        # Update if exists, otherwise append
        for i, row in enumerate(rows, start=2):  # Header is row 1
            if row.get("name") == name:
                worksheet.update(f"A{i}:C{i}", [payload])
                self.list_presets.clear()
                return

        # Append new preset
        worksheet.append_row(payload, value_input_option="RAW")
        self.list_presets.clear()

    def delete_preset(self, name: str) -> None:
        """Delete a preset from Google Sheets."""
        client = get_google_client()
        spreadsheet = client.open_by_key(self.config['spreadsheet_id'])
        worksheet = self._ensure_presets_sheet(spreadsheet)

        rows = worksheet.get_all_records()
        for i, row in enumerate(rows, start=2):  # Header is row 1
            if row.get("name") == name:
                worksheet.delete_rows(i)
                break

        self.list_presets.clear()
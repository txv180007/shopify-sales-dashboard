import streamlit as st
from typing import Optional


class AuthManager:
    """Handles authentication for the dashboard."""

    def __init__(self):
        self.username = st.secrets.get("DASHBOARD_USERNAME", "").strip()
        self.password = st.secrets.get("DASHBOARD_PASSWORD", "").strip()

    def is_auth_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return bool(self.username and self.password)

    def require_login(self) -> bool:
        """Require login if authentication is enabled."""
        if not self.is_auth_enabled():
            return True  # Authentication disabled

        if "auth_ok" not in st.session_state:
            st.session_state.auth_ok = False

        if not st.session_state.auth_ok:
            self._render_login_form()
            st.stop()

        return True

    def _render_login_form(self) -> None:
        """Render the login form."""
        st.title("Daily Products Dashboard")

        username_input = st.text_input("Username", key="auth_user")
        password_input = st.text_input("Password", type="password", key="auth_pass")

        if st.button("Login", key="auth_login_btn"):
            if self._validate_credentials(username_input, password_input):
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials."""
        return username == self.username and password == self.password
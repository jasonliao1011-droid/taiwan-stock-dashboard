from __future__ import annotations

import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="Taiwan Stock Platform",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    st.title("Taiwan Stock Data Platform")

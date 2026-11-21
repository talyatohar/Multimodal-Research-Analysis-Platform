"""Shared layout and styling for the research platform."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

PLATFORM_NAME = "Multimodal Research Analysis Platform"
PLATFORM_TAGLINE = "Behavioral • Physiological • EEG • Eye-Tracking Analysis"
PLATFORM_PAGE_SUFFIX = "Multimodal Research Platform"

DATAFRAME_HEADER_HEIGHT_PX = 38
DATAFRAME_ROW_HEIGHT_PX = 35
DEFAULT_MAX_VISIBLE_ROWS = 12
# Tables wider than their container show a horizontal scrollbar; an exact
# header+rows height leaves no room for it, which makes the grid oscillate
# between "fits" and "needs scroll" on every repaint (visible as flicker).
DATAFRAME_SCROLLBAR_BUFFER_PX = 17
EVENT_SUMMARY_TABLE_HEIGHT = DATAFRAME_HEADER_HEIGHT_PX + DATAFRAME_ROW_HEIGHT_PX + DATAFRAME_SCROLLBAR_BUFFER_PX
EVENT_EEG_DISTRIBUTION_TABLE_HEIGHT = (
    DATAFRAME_HEADER_HEIGHT_PX + DATAFRAME_ROW_HEIGHT_PX * 6 + DATAFRAME_SCROLLBAR_BUFFER_PX
)


def dataframe_height_for_rows(
    row_count: int,
    *,
    max_visible_rows: int | None = DEFAULT_MAX_VISIBLE_ROWS,
) -> int | None:
    """Compute a tight st.dataframe height for the given number of data rows."""
    if row_count <= 0:
        return None
    visible_rows = row_count
    if max_visible_rows is not None and row_count > max_visible_rows:
        visible_rows = max_visible_rows
    return DATAFRAME_HEADER_HEIGHT_PX + DATAFRAME_ROW_HEIGHT_PX * visible_rows + DATAFRAME_SCROLLBAR_BUFFER_PX


def configure_page(page_title: str, page_icon: str = "ℹ️") -> None:
    st.set_page_config(
        page_title=f"{page_title} · {PLATFORM_PAGE_SUFFIX}",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": None},
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1100px; }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f1419 0%, #121a22 100%);
                border-right: 1px solid #1e2835;
            }
            div[data-testid="stDataFrame"] > div {
                overflow-x: auto !important;
            }
            div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
                overflow-x: auto !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.caption(PLATFORM_TAGLINE)
    st.sidebar.markdown(f"### {PLATFORM_NAME}")
    st.sidebar.divider()
    from shared.developer_mode import render_developer_mode_controls

    render_developer_mode_controls()


def page_header(
    eyebrow: str,
    title: str,
    subtitle: str | None = None,
    *,
    hero: bool = False,
) -> None:
    st.caption(eyebrow)
    if hero:
        st.title(title)
    else:
        st.header(title)
    if subtitle:
        st.markdown(subtitle)


def render_stable_dataframe(
    df: pd.DataFrame | None,
    *,
    height: int | None = None,
    column_config: dict[str, Any] | None = None,
    max_visible_rows: int | None = DEFAULT_MAX_VISIBLE_ROWS,
    key: str | None = None,
    container: Any | None = None,
    on_select: str | None = None,
    selection_mode: str | None = None,
) -> Any:
    """Render a result table sized to its rows; scroll vertically only when needed.

    Returns the value of the underlying st.dataframe call (None unless on_select is
    set, in which case it's the selection event — e.g. `event.selection.rows`).
    """
    table = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if table.empty:
        return None

    resolved_height = height
    if resolved_height is None:
        resolved_height = dataframe_height_for_rows(len(table), max_visible_rows=max_visible_rows)

    kwargs: dict[str, Any] = {
        "use_container_width": True,
        "hide_index": True,
    }
    if resolved_height is not None:
        kwargs["height"] = resolved_height
    if column_config:
        kwargs["column_config"] = column_config
    if key is not None:
        kwargs["key"] = key
    if on_select is not None:
        kwargs["on_select"] = on_select
    if selection_mode is not None:
        kwargs["selection_mode"] = selection_mode
    target = container if container is not None else st
    try:
        return target.dataframe(table, **kwargs)
    except TypeError:
        kwargs.pop("height", None)
        kwargs.pop("key", None)
        kwargs.pop("on_select", None)
        kwargs.pop("selection_mode", None)
        return target.dataframe(table, **kwargs)

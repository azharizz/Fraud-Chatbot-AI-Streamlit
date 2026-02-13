import streamlit as st

_THEME_CSS = """\
<style>
    /* ── Color tokens ─────────────────────────────────────────── */
    :root {
        --mk-purple: #6B2FA0;
        --mk-purple-dark: #4E1F7A;
        --mk-purple-light: #F3EDFA;
        --mk-purple-50: #F9F5FD;
        --mk-accent: #9B59B6;
        --mk-text: #2D2D2D;
        --mk-text-muted: #6B7280;
        --mk-border: #E5E7EB;
        --mk-bg: #FFFFFF;
        --mk-bg-subtle: #F9FAFB;
    }

    /* ── Main background ──────────────────────────────────────── */
    .stApp {
        background-color: var(--mk-bg) !important;
        color: var(--mk-text);
    }

    /* ── Header ───────────────────────────────────────────────── */
    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, var(--mk-purple), var(--mk-purple-dark)) !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--mk-purple-50) 0%, #FFFFFF 100%) !important;
        border-right: 1px solid var(--mk-border);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--mk-purple) !important;
    }

    /* ── Buttons ──────────────────────────────────────────────── */
    .stButton > button {
        text-align: left;
        font-size: 0.85rem;
        padding: 10px 14px;
        white-space: normal;
        height: auto;
        min-height: 0;
        border: 1px solid var(--mk-border) !important;
        border-radius: 10px !important;
        background-color: var(--mk-bg-subtle) !important;
        color: var(--mk-text) !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--mk-purple-light) !important;
        border-color: var(--mk-purple) !important;
        color: var(--mk-purple) !important;
        box-shadow: 0 2px 8px rgba(107, 47, 160, 0.15);
    }

    /* ── Chat messages ────────────────────────────────────────── */
    .stChatMessage {
        border-radius: 12px;
        border: 1px solid var(--mk-border);
        background-color: var(--mk-bg) !important;
    }

    /* ── Metric cards ─────────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--mk-purple) !important;
    }

    /* ── Dataframe ────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--mk-border);
    }

    /* ── Expanders ────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        color: var(--mk-purple) !important;
        font-weight: 600;
    }

    /* ── Chat input ───────────────────────────────────────────── */
    .stChatInput > div {
        border-color: var(--mk-border) !important;
        border-radius: 12px !important;
    }

    .stChatInput > div:focus-within {
        border-color: var(--mk-purple) !important;
        box-shadow: 0 0 0 2px rgba(107, 47, 160, 0.2) !important;
    }

    /* ── Divider ──────────────────────────────────────────────── */
    hr { border-color: var(--mk-border) !important; }

    /* ── Code blocks ──────────────────────────────────────────── */
    .stCodeBlock {
        border-radius: 8px;
    }

    /* ── Spinner ──────────────────────────────────────────────── */
    .stSpinner > div { color: var(--mk-purple) !important; }

    /* ── Toggle / Checkbox ────────────────────────────────────── */
    .stCheckbox label span { color: var(--mk-text) !important; }
</style>
"""


def apply_theme() -> None:
    """Inject the Mekari Purple theme CSS into the Streamlit page."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)

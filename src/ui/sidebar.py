from typing import Any

import streamlit as st


EXAMPLE_QUESTIONS = [
    "How does the monthly fraud rate fluctuate over the two-year period?",
    "Which merchant categories exhibit the highest incidence of fraudulent transactions?",
    "What are the primary methods by which credit card fraud is committed?",
    "What are the core components of an effective fraud detection system, according to the authors?",
    "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
    "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
]


def render_sidebar() -> str | None:
    """Render the sidebar and return selected example question (if any)."""
    selected_question: str | None = None

    with st.sidebar:
        st.markdown("## 🔍 Fraud Analysis Chatbot")
        st.markdown(
            "Ask questions about credit card fraud data and research documents."
        )
        st.divider()

        # Example questions
        st.markdown("### 💡 Example Questions")
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if st.button(
                q,
                key=f"example_{i}",
                use_container_width=True,
            ):
                selected_question = q

        st.divider()

        # Dataset info
        with st.expander("📊 Dataset Info", expanded=False):
            st.markdown(
                """
                **Transaction Data**
                - **Records**: ~1,852,394 transactions
                - **Period**: Jan 2019 – Dec 2020
                - **Fraud rate**: ~0.6%
                - **Categories**: 14 merchant categories

                **Research Documents**
                - 📄 *Understanding Credit Card Frauds* (Bhatla et al., 2003)
                - 📄 *2024 Report on Payment Fraud* (EBA/ECB, Aug 2024)
                """
            )

        # Settings
        with st.expander("⚙️ Settings", expanded=False):
            st.toggle(
                "Show quality scores",
                value=True,
                key="show_quality_scores",
            )
            st.toggle(
                "Show SQL queries",
                value=True,
                key="show_sql_queries",
            )
            st.toggle(
                "Show retrieved sources",
                value=True,
                key="show_sources",
            )
            st.divider()
            st.toggle(
                "🔗 Multi-source synthesis",
                value=True,
                key="enable_synthesis",
                help="When both SQL and document tools are used, synthesize a unified answer.",
            )
            st.toggle(
                "⚡ Streaming responses",
                value=True,
                key="enable_streaming",
                help="Show response tokens as they are generated.",
            )

        st.divider()
        st.caption("Built with PydanticAI + OpenAI + DuckDB + FAISS")

    return selected_question

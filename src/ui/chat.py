from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from src.models.scoring import QualityScore

CHART_COLORS = ["#6B2FA0", "#9B59B6", "#BB8FCE", "#D2B4DE", "#E8DAEF"]
CHART_TEMPLATE = "plotly_white"


class ChatRenderer:
    """Class-based Streamlit chat rendering with unique element keys."""

    def __init__(self) -> None:
        self._chart_counter = 0

    def _next_chart_key(self, prefix: str = "chart") -> str:
        """Generate a unique key for Streamlit elements to avoid duplicate ID errors."""
        self._chart_counter += 1
        return f"{prefix}_{self._chart_counter}"

    def render_quality_badge(self, score: QualityScore) -> None:
        """Render a colored quality score badge with expandable breakdown."""
        if not st.session_state.get("show_quality_scores", True):
            return

        overall = score.overall
        if overall >= 0.7:
            color = "🟢"
            label = "High"
        elif overall >= 0.4:
            color = "🟡"
            label = "Medium"
        else:
            color = "🔴"
            label = "Low"

        st.markdown(f"**Quality**: {color} {overall:.2f} ({label})")

        with st.expander("📋 Score Breakdown", expanded=False):
            cols = st.columns(3)
            cols[0].metric("Faithfulness", f"{score.faithfulness:.2f}")
            cols[1].metric("Relevance", f"{score.relevance:.2f}")
            cols[2].metric("Confidence", f"{score.confidence:.2f}")
            if score.faithfulness_reason:
                st.caption(f"💬 {score.faithfulness_reason}")
            if score.validation_passed is not None:
                if score.validation_passed:
                    st.success(f"✅ Verified: {score.validation_reason}")
                else:
                    st.warning(f"⚠️ {score.validation_reason}")

    def render_sql_details(
        self,
        sql_query: str | None,
        sql_results: list[dict] | None,
        columns: list[str] | None,
    ) -> None:
        """Render SQL query and tabular results with auto-visualization."""
        if not sql_query and not sql_results:
            return

        if sql_query and st.session_state.get("show_sql_queries", True):
            with st.expander("🔧 SQL Query", expanded=False):
                st.code(sql_query, language="sql")

        if sql_results and columns:
            df = pd.DataFrame(sql_results)
            st.dataframe(df, width="stretch", hide_index=True)
            self._auto_chart(df, columns)

    def render_rag_sources(
        self,
        sources: list[dict[str, Any]] | None,
        retrieved_chunks: list[str] | None,
    ) -> None:
        """Render retrieved document sources with attribution."""
        if not st.session_state.get("show_sources", True):
            return
        if not sources:
            return

        with st.expander("📚 Retrieved Sources", expanded=False):
            for i, src in enumerate(sources):
                score_pct = src.get("score", 0) * 100
                st.markdown(
                    f"**[{i+1}]** {src.get('source', 'Unknown')} — "
                    f"Page {src.get('page', '?')} "
                    f"(relevance: {score_pct:.1f}%)"
                )
                if retrieved_chunks and i < len(retrieved_chunks):
                    text = retrieved_chunks[i]
                    st.caption(text[:300] + "..." if len(text) > 300 else text)
                if i < len(sources) - 1:
                    st.divider()

    def render_chat_history(self) -> None:
        """Render all messages in the chat history."""
        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant" and "metadata" in msg:
                    meta = msg["metadata"]
                    self.render_sql_details(
                        meta.get("sql_query"),
                        meta.get("sql_results"),
                        meta.get("sql_columns"),
                    )
                    self.render_rag_sources(
                        meta.get("sources"),
                        meta.get("retrieved_chunks"),
                    )
                    if "quality_score" in meta and meta["quality_score"]:
                        self.render_quality_badge(QualityScore(**meta["quality_score"]))

    def _auto_chart(self, df: pd.DataFrame, columns: list[str]) -> None:
        """Auto-detect the best chart type based on the data columns."""
        if df.empty or len(columns) < 2:
            return

        time_col = None
        for c in columns:
            cl = c.lower()
            if any(kw in cl for kw in ["month", "day", "date", "year", "time", "week"]):
                time_col = c
                break

        value_cols = [
            c for c in columns
            if any(kw in c.lower() for kw in [
                "count", "rate", "amount", "total", "avg", "sum", "fraud",
                "pct", "percentage",
            ])
            and c != time_col
        ]

        if time_col and value_cols:
            fig = go.Figure()
            for i, vc in enumerate(value_cols[:3]):
                fig.add_trace(go.Scatter(
                    x=df[time_col], y=df[vc],
                    mode="lines+markers", name=vc,
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                    marker=dict(size=5),
                ))
            fig.update_layout(
                xaxis_title=time_col,
                yaxis_title=value_cols[0] if len(value_cols) == 1 else "Value",
                template=CHART_TEMPLATE, height=400,
                margin=dict(l=40, r=40, t=30, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#2D2D2D"),
            )
            st.plotly_chart(fig, width="stretch", key=self._next_chart_key("line"))

        elif not time_col and value_cols:
            cat_col = next(
                (c for c in columns if c.lower() not in {vc.lower() for vc in value_cols}),
                columns[0],
            )
            if cat_col in df.columns and value_cols[0] in df.columns:
                fig = px.bar(
                    df.head(20), x=cat_col, y=value_cols[0],
                    color=value_cols[0],
                    color_continuous_scale=["#E8DAEF", "#9B59B6", "#6B2FA0"],
                    template=CHART_TEMPLATE,
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=40, r=40, t=30, b=40),
                    xaxis_tickangle=-45,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#2D2D2D"),
                )
                st.plotly_chart(fig, width="stretch", key=self._next_chart_key("bar"))

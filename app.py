import asyncio
import logging
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.core.llm_client import LLMClient
from src.agent.router import FraudRouter
from src.data.database import FraudDatabase
from src.data.vectorstore import VectorStore
from src.models.agent import AgentDeps, AgentResponse
from src.models.source_type import SourceType
from src.scoring.quality import QualityScorer
from src.scoring.validation import AnswerValidator
from src.ui.chat import ChatRenderer
from src.ui.sidebar import render_sidebar
from src.ui.theme import apply_theme

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")


st.set_page_config(
    page_title="Fraud Q&A Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI()

@st.cache_resource
def get_db() -> FraudDatabase:
    return FraudDatabase.connect()

@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore.load()


if "messages" not in st.session_state:
    st.session_state.messages = []

selected_question = render_sidebar()

st.markdown("# 🔍 Fraud Analysis Chatbot")
st.markdown(
    "Ask questions about credit card fraud transaction data (2019-2020) "
    "or fraud research documents."
)
st.divider()

renderer = ChatRenderer()
renderer.render_chat_history()

chat_input = st.chat_input("Ask a question about fraud...")
question = selected_question or chat_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            client = get_openai_client()
            db = get_db()
            vs = get_vector_store()
            llm = LLMClient(client)

            deps = AgentDeps(
                con=db.connection,
                openai_client=client,
                faiss_index=vs.index,
                chunks=vs.chunks,
            )

            router = FraudRouter(llm, db, vs)
            scorer = QualityScorer(llm)
            validator = AnswerValidator()

            enable_synthesis = st.session_state.get("enable_synthesis", True)
            enable_streaming = st.session_state.get("enable_streaming", True)

            response: AgentResponse | None = None

            if enable_streaming:
                response_placeholder = st.empty()
                stream_state = {"text": "", "response": None}

                async def _stream():
                    async for item in router.run_stream(
                        question, deps,
                        message_history=st.session_state.messages,
                        enable_synthesis=enable_synthesis,
                    ):
                        if isinstance(item, str):
                            stream_state["text"] += item
                            response_placeholder.markdown(stream_state["text"] + "▌")
                        elif isinstance(item, AgentResponse):
                            stream_state["response"] = item

                with st.spinner("🤔 Analyzing your question..."):
                    asyncio.run(_stream())

                response = stream_state["response"]

                if response and response.answer != stream_state["text"]:
                    response_placeholder.markdown(response.answer)
                else:
                    response_placeholder.markdown(stream_state["text"])
            else:
                with st.spinner("🤔 Analyzing your question..."):
                    response = asyncio.run(router.run(
                        question, deps,
                        message_history=st.session_state.messages,
                        enable_synthesis=enable_synthesis,
                    ))
                st.markdown(response.answer)

            if response:
                renderer.render_sql_details(response.sql_query, response.sql_results, response.sql_columns)
                renderer.render_rag_sources(response.sources, response.retrieved_chunks)

                context = ""
                if response.sql_results:
                    context = str(response.sql_results[:20])
                if response.retrieved_chunks:
                    context += "\n".join(response.retrieved_chunks)

                quality = scorer.score(
                    question=question,
                    answer=response.answer,
                    context=context or response.answer,
                    source_type=response.source_type,
                    similarity_scores=response.similarity_scores,
                    sql_success=response.sql_results is not None and len(response.sql_results) > 0,
                    sql_row_count=len(response.sql_results) if response.sql_results else 0,
                )

                val_passed, val_reason = validator.validate(
                    answer=response.answer,
                    source_type=response.source_type,
                    sql_results=response.sql_results,
                    retrieved_chunks=response.retrieved_chunks,
                )
                quality.validation_passed = val_passed
                quality.validation_reason = val_reason

                renderer.render_quality_badge(quality)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "metadata": {
                        "sql_query": response.sql_query,
                        "sql_results": response.sql_results,
                        "sql_columns": response.sql_columns,
                        "sources": response.sources,
                        "retrieved_chunks": response.retrieved_chunks,
                        "quality_score": quality.model_dump(),
                        "source_type": response.source_type.value,
                    },
                })

        except Exception as e:
            error_msg = f"❌ An error occurred: {str(e)}"
            st.error(error_msg)
            logger.error("Error processing question: %s", e, exc_info=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })

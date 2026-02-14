import logging
from typing import AsyncIterator

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai import Agent, RunContext

from src.agent.prompts import ROUTER_SYSTEM_PROMPT
from src.agent.sql_tool import SQLTool
from src.agent.rag_tool import RAGTool
from src.agent.synthesis import ResultSynthesizer
from src.core.config import MIN_QUESTION_LENGTH, MAX_QUESTION_LENGTH
from src.core.llm_client import LLMClient
from src.data.database import FraudDatabase
from src.data.vectorstore import VectorStore
from src.models.agent import AgentDeps, AgentResponse
from src.models.source_type import SourceType
from src.models.tools import SQLToolResult, RAGToolResult

logger = logging.getLogger(__name__)


class FraudRouter:
    """PydanticAI-based router that dispatches questions to SQL or RAG tools.

    Encapsulates the agent singleton, tool instances, and all response-building
    logic. Eliminates the module-level global `_agent`.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        database: FraudDatabase,
        vector_store: VectorStore,
    ) -> None:
        self._llm = llm_client
        self._sql_tool = SQLTool(llm_client, database)
        self._rag_tool = RAGTool(llm_client, vector_store)
        self._synthesizer = ResultSynthesizer(llm_client)
        self._agent = self._create_agent()

    def _create_agent(self) -> Agent[AgentDeps, str]:
        """Build the PydanticAI agent with registered tools."""
        sql_tool = self._sql_tool
        rag_tool = self._rag_tool

        a = Agent(
            model="openai:gpt-4o-mini",
            system_prompt=ROUTER_SYSTEM_PROMPT,
            deps_type=AgentDeps,
            retries=2,
        )

        @a.tool
        async def query_fraud_database(ctx: RunContext[AgentDeps], question: str) -> str:
            """Query the fraud transaction database using SQL.

            Use this for questions about transaction data, statistics, trends, counts,
            amounts, rates from the fraud dataset (2019-2020, ~1.85M transactions).
            """
            logger.info("SQL Tool called with: %s", question)
            result = sql_tool.run(question)
            ctx.deps.tool_outputs["sql"] = result

            if not result.success:
                return f"SQL query failed: {result.error}"
            if not result.rows:
                return "Query executed successfully but returned no results."

            lines = [f"SQL Query: {result.sql_query}", ""]
            lines.append(f"Results ({result.row_count} rows):")
            lines.append(" | ".join(result.columns))
            lines.append("-" * 60)
            for row in result.rows[:50]:
                lines.append(" | ".join(str(row.get(c, "")) for c in result.columns))
            if result.row_count > 50:
                lines.append(f"... and {result.row_count - 50} more rows")
            return "\n".join(lines)

        @a.tool
        async def search_fraud_documents(ctx: RunContext[AgentDeps], question: str) -> str:
            """Search the fraud research documents for information.

            Use this for questions about fraud concepts, methods, prevention techniques,
            regulatory findings, EBA/ECB report data, cross-border statistics.
            """
            logger.info("RAG Tool called with: %s", question)
            result = rag_tool.run(
                question=question,
                client=ctx.deps.openai_client,
            )
            ctx.deps.tool_outputs["rag"] = result

            if not result.success:
                return f"Document search failed: {result.error}"
            return result.answer

        return a

    def _sql_is_unanswerable(self, sql: SQLToolResult | None) -> bool:
        """Check if SQL returned an UNANSWERABLE result."""
        if not sql or not sql.success:
            return False
        if sql.rows and len(sql.rows) == 1:
            first_row = sql.rows[0]
            values = [str(v).upper() for v in first_row.values()]
            return any("UNANSWERABLE" in v for v in values)
        return False

    def _fallback_to_rag(self, question: str, deps: AgentDeps) -> RAGToolResult | None:
        """Invoke the RAG tool as a fallback when SQL can't answer."""
        logger.info("SQL returned UNANSWERABLE; falling back to RAG for: %s", question)
        try:
            rag_result = self._rag_tool.run(
                question=question,
                client=deps.openai_client,
            )
            deps.tool_outputs["rag"] = rag_result
            return rag_result
        except Exception as e:
            logger.error("RAG fallback failed: %s", e, exc_info=True)
            return None

    async def run(
        self,
        question: str,
        deps: AgentDeps,
        message_history: list[dict[str, str]] | None = None,
        enable_synthesis: bool = True,
    ) -> AgentResponse:
        """Run the agent synchronously and return a structured response."""
        error = self._validate_input(question)
        if error:
            return AgentResponse(answer=error, source_type=SourceType.ERROR, error=error)

        try:
            deps.tool_outputs = {}
            history = self._build_message_history(message_history)

            result = await self._agent.run(question, deps=deps, message_history=history)
            answer = result.output if isinstance(result.output, str) else str(result.output)

            sql = deps.tool_outputs.get("sql")
            rag = deps.tool_outputs.get("rag")

            # Fallback: if SQL returned UNANSWERABLE and RAG wasn't called,
            # automatically invoke RAG since the answer may exist in documents
            if self._sql_is_unanswerable(sql) and rag is None:
                rag = self._fallback_to_rag(question, deps)
                if rag and rag.success and rag.answer:
                    answer = rag.answer

            source_type = self._infer_source_type(sql, rag)

            if enable_synthesis and source_type == SourceType.BOTH and sql and rag:
                synthesized = self._synthesizer.synthesize(question, sql, rag)
                if synthesized:
                    answer = synthesized

            return self._build_response(answer, sql, rag)

        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            return self._error_response(e)

    async def run_stream(
        self,
        question: str,
        deps: AgentDeps,
        message_history: list[dict[str, str]] | None = None,
        enable_synthesis: bool = True,
    ) -> AsyncIterator[str | AgentResponse]:
        """Stream agent output as text deltas, then yield the full AgentResponse."""
        error = self._validate_input(question)
        if error:
            yield AgentResponse(answer=error, source_type=SourceType.ERROR, error=error)
            return

        try:
            deps.tool_outputs = {}
            history = self._build_message_history(message_history)

            full_text = ""
            async with self._agent.run_stream(question, deps=deps, message_history=history) as stream:
                async for delta in stream.stream_text(delta=True):
                    full_text += delta
                    yield delta

            sql = deps.tool_outputs.get("sql")
            rag = deps.tool_outputs.get("rag")

            # Fallback: if SQL returned UNANSWERABLE and RAG wasn't called,
            # automatically invoke RAG since the answer may exist in documents
            if self._sql_is_unanswerable(sql) and rag is None:
                rag = self._fallback_to_rag(question, deps)
                if rag and rag.success and rag.answer:
                    full_text = rag.answer

            source_type = self._infer_source_type(sql, rag)

            if enable_synthesis and source_type == SourceType.BOTH and sql and rag:
                synthesized = self._synthesizer.synthesize(question, sql, rag)
                if synthesized:
                    full_text = synthesized

            yield self._build_response(full_text, sql, rag)

        except Exception as e:
            logger.error("Agent stream error: %s", e, exc_info=True)
            yield self._error_response(e)

    @staticmethod
    def _validate_input(question: str) -> str | None:
        q = question.strip() if question else ""
        if not q:
            return "Please enter a question."
        if len(q) < MIN_QUESTION_LENGTH:
            return f"Question is too short (minimum {MIN_QUESTION_LENGTH} characters)."
        if len(q) > MAX_QUESTION_LENGTH:
            return f"Question is too long (maximum {MAX_QUESTION_LENGTH} characters)."
        return None

    @staticmethod
    def _build_message_history(
        message_history: list[dict[str, str]] | None,
    ) -> list[ModelMessage] | None:
        if not message_history:
            return None
        recent = message_history[-6:]
        pydantic_history = [
            ModelRequest(parts=[UserPromptPart(content=msg["content"])])
            for msg in recent
            if msg["role"] == "user"
        ]
        return pydantic_history or None

    @staticmethod
    def _infer_source_type(
        sql: SQLToolResult | None,
        rag: RAGToolResult | None,
    ) -> SourceType:
        has_sql = sql is not None
        has_rag = rag is not None
        if has_sql and has_rag:
            return SourceType.BOTH
        if has_sql:
            return SourceType.SQL
        return SourceType.RAG

    @staticmethod
    def _build_response(
        answer: str,
        sql: SQLToolResult | None,
        rag: RAGToolResult | None,
    ) -> AgentResponse:
        source_type = FraudRouter._infer_source_type(sql, rag)
        return AgentResponse(
            answer=answer,
            source_type=source_type,
            sql_query=sql.sql_query if sql and sql.success else None,
            sql_results=sql.rows if sql and sql.success else None,
            sql_columns=sql.columns if sql and sql.success else None,
            retrieved_chunks=rag.retrieved_chunks if rag and rag.success else None,
            similarity_scores=rag.similarity_scores if rag and rag.success else None,
            sources=rag.sources if rag and rag.success else None,
        )

    @staticmethod
    def _error_response(error: str | Exception) -> AgentResponse:
        msg = str(error)
        return AgentResponse(
            answer=f"I encountered an error processing your question: {msg}",
            source_type=SourceType.ERROR,
            error=msg,
        )

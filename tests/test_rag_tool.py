import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

from src.core.llm_client import LLMClient
from src.data.vectorstore import VectorStore
from src.agent.rag_tool import RAGTool


@pytest.fixture(scope="module")
def vector_store():
    return VectorStore.load()


@pytest.fixture(scope="module")
def llm_client():
    return LLMClient(OpenAI())


@pytest.fixture(scope="module")
def openai_client():
    return OpenAI()


@pytest.fixture(scope="module")
def rag_tool(llm_client, vector_store):
    return RAGTool(llm_client, vector_store)


class TestVectorStoreSearch:

    def test_search_basic(self, vector_store, openai_client):
        results = vector_store.search("credit card fraud types", client=openai_client, top_k=3)
        assert len(results) > 0
        assert results[0].text
        assert results[0].score > 0

    def test_search_with_bhatla_filter(self, vector_store, openai_client):
        results = vector_store.search(
            "fraud methods", client=openai_client, top_k=3, source_filter="bhatla",
        )
        for r in results:
            assert r.metadata.source == "bhatla"

    def test_search_with_eba_filter(self, vector_store, openai_client):
        results = vector_store.search(
            "SCA regulation", client=openai_client, top_k=3, source_filter="eba_ecb_2024",
        )
        for r in results:
            assert r.metadata.source == "eba_ecb_2024"

    def test_search_returns_metadata(self, vector_store, openai_client):
        results = vector_store.search("fraud prevention", client=openai_client, top_k=1)
        assert results
        meta = results[0].metadata
        assert meta.source
        assert meta.page > 0


class TestRAGTool:

    def test_rag_tool_basic(self, rag_tool, openai_client):
        result = rag_tool.run("What are the main types of credit card fraud?", client=openai_client)
        assert result.success
        assert len(result.answer) > 50
        assert result.sources

    def test_rag_tool_bhatla_specific(self, rag_tool, openai_client):
        result = rag_tool.run(
            "What does Bhatla say about fraud detection systems?",
            client=openai_client,
        )
        assert result.success
        assert any("Bhatla" in s.get("source", "") for s in result.sources)

    def test_rag_tool_eba_specific(self, rag_tool, openai_client):
        result = rag_tool.run(
            "What is the cross-border fraud share according to the EBA/ECB report?",
            client=openai_client,
        )
        assert result.success
        assert result.similarity_scores
        assert all(0.0 <= s <= 1.0 for s in result.similarity_scores)

    def test_rag_tool_source_filter_detection(self):
        assert RAGTool._detect_source_filter("What does the EBA report say?") == "eba_ecb_2024"
        assert RAGTool._detect_source_filter("Bhatla paper findings") == "bhatla"
        assert RAGTool._detect_source_filter("General question") is None

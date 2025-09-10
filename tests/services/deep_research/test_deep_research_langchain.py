import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apex.services.deep_research.deep_research_langchain import (
    DeepResearchLangchain,
    _CustomEmbeddings,
)


@pytest.fixture
def mock_websearch():
    """Fixture for a mocked WebSearchBase."""
    return AsyncMock()


@pytest.fixture
def mock_llm_embed():
    """Fixture for a mocked LLMEmbed."""
    return AsyncMock()


@pytest.fixture
def mock_chat_openai():
    """Fixture for a mocked ChatOpenAI."""
    return MagicMock()


@pytest.fixture
def deep_research_langchain(mock_websearch, mock_llm_embed, mock_chat_openai):
    """Fixture for an instance of DeepResearchLangchain with mocked dependencies."""
    with (
        patch("apex.services.deep_research.deep_research_langchain.LLMEmbed", return_value=mock_llm_embed),
        patch("apex.services.deep_research.deep_research_langchain.ChatOpenAI", return_value=mock_chat_openai),
        patch(
            "apex.services.deep_research.deep_research_langchain.PythonREPL",
            return_value=MagicMock(run=MagicMock(return_value="2\n")),
        ),
    ):
        return DeepResearchLangchain(
            key="test_key",
            base_url="http://test.url",
            emb_base_url="http://test.emb.url",
            summary_model="summary_model",
            research_model="research_model",
            compression_model="compression_model",
            final_model="final_model",
            websearch=mock_websearch,
        )


@pytest.mark.asyncio
async def test_custom_embeddings_aembed_documents(mock_llm_embed):
    """Test the aembed_documents method of _CustomEmbeddings."""
    custom_embeddings = _CustomEmbeddings(mock_llm_embed)
    texts = ["text1", "text2"]
    await custom_embeddings.aembed_documents(texts)
    mock_llm_embed.invoke.assert_called_once_with(inputs=texts)


@pytest.mark.asyncio
async def test_custom_embeddings_aembed_query(mock_llm_embed):
    """Test the aembed_query method of _CustomEmbeddings."""
    mock_llm_embed.invoke.return_value = [[1.0, 2.0, 3.0]]
    custom_embeddings = _CustomEmbeddings(mock_llm_embed)
    text = "query text"
    result = await custom_embeddings.aembed_query(text)
    mock_llm_embed.invoke.assert_called_once_with(inputs=[text])
    assert result == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_invoke_with_documents_in_body(deep_research_langchain, mock_websearch):
    """When body contains documents, agent can directly produce a final report without websearch."""
    messages = [{"role": "user", "content": "test question"}]
    body = {"documents": [{"page_content": "doc1"}, {"page_content": "doc2"}]}

    with (
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_agent_chain"
        ) as mock_build_agent_chain,
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_final_chain"
        ) as mock_build_final_chain,
    ):
        agent_chain = AsyncMock()
        return_value = json.dumps({"thought": "enough info", "final_answer": "final_report"})
        agent_chain.ainvoke.return_value = return_value
        mock_build_agent_chain.return_value = agent_chain

        final_chain_mock = AsyncMock()
        final_chain_mock.ainvoke.return_value = return_value
        mock_build_final_chain.return_value = final_chain_mock

        result = await deep_research_langchain.invoke(messages, body)

        mock_websearch.search.assert_not_called()
        assert result[0] == return_value


@pytest.mark.asyncio
async def test_invoke_with_websearch(deep_research_langchain, mock_websearch):
    """Agent chooses websearch then produces final answer."""
    messages = [{"role": "user", "content": "test question"}]
    mock_websearch.search.return_value = [MagicMock(content="web_doc", url="http://a.com", title="A")]

    with (
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_agent_chain"
        ) as mock_build_agent_chain,
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_final_chain"
        ) as mock_build_final_chain,
    ):
        agent_chain = AsyncMock()
        agent_chain.ainvoke.side_effect = [
            (
                '{"thought": "need info", "action": {"tool": "websearch", '
                '"input": {"query": "test query", "max_results": 3}}}'
            ),
            '{"thought": "done", "final_answer": "final_answer"}',
        ]
        mock_build_agent_chain.return_value = agent_chain

        final_chain_mock = AsyncMock()
        final_chain_mock.ainvoke.return_value = "final_answer"
        mock_build_final_chain.return_value = final_chain_mock

        result = await deep_research_langchain.invoke(messages)

        mock_websearch.search.assert_called_once_with(query="test query", max_results=3)
        assert result[0] == "final_answer"


@pytest.mark.asyncio
async def test_invoke_no_websearch_needed_final_answer(deep_research_langchain, mock_websearch):
    """Agent can produce a final report without calling websearch."""
    messages = [{"role": "user", "content": "test question"}]

    with (
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_agent_chain"
        ) as mock_build_agent_chain,
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_final_chain"
        ) as mock_build_final_chain,
    ):
        agent_chain = AsyncMock()
        return_value = json.dumps({"thought": "enough info", "final_answer": "final_report"})
        agent_chain.ainvoke.return_value = return_value
        mock_build_agent_chain.return_value = agent_chain

        final_chain_mock = AsyncMock()
        final_chain_mock.ainvoke.return_value = return_value
        mock_build_final_chain.return_value = final_chain_mock

        result = await deep_research_langchain.invoke(messages)

        mock_websearch.search.assert_not_called()
        assert result[0] == return_value


@pytest.mark.asyncio
async def test_full_invoke_flow_with_multiple_actions(deep_research_langchain, mock_websearch):
    """Agent performs multiple websearch actions before final answer; tool_history and traces are recorded."""
    messages = [{"role": "user", "content": "test question"}]

    # Two rounds of search results
    mock_websearch.search.side_effect = [
        [
            MagicMock(content="doc A", url="http://a.com", title="A"),
            MagicMock(content="doc B", url="http://b.com", title="B"),
        ],
        [MagicMock(content="doc C", url="http://c.com", title="C")],
    ]

    with (
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_agent_chain"
        ) as mock_build_agent_chain,
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_final_chain"
        ) as mock_build_final_chain,
    ):
        agent_chain = AsyncMock()
        agent_chain.ainvoke.side_effect = [
            (
                '{"thought": "need more info", "action": {"tool": "websearch", '
                '"input": {"query": "Q1", "max_results": 2}}}'
            ),
            (
                '{"thought": "still need more", "action": {"tool": "websearch", '
                '"input": {"query": "Q2", "max_results": 1}}}'
            ),
            '{"thought": "complete", "final_answer": "final_report"}',
        ]
        mock_build_agent_chain.return_value = agent_chain

        final_chain_mock = AsyncMock()
        final_chain_mock.ainvoke.return_value = "final_report"
        mock_build_final_chain.return_value = final_chain_mock

        result = await deep_research_langchain.invoke(messages)

        # Two tool uses
        assert mock_websearch.search.call_count == 2
        mock_websearch.search.assert_any_call(query="Q1", max_results=2)
        mock_websearch.search.assert_any_call(query="Q2", max_results=1)

        # Final answer returned
        assert result[0] == "final_report"
        # Tool history recorded
        assert len(result[1]) == 2
        assert result[1][0]["tool"] == "websearch"
        # Reasoning traces present
        assert isinstance(result[2], list)


@pytest.mark.asyncio
async def test_invoke_with_python_repl(deep_research_langchain):
    """Agent chooses python_repl then produces final answer."""
    with (
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_agent_chain"
        ) as mock_build_agent_chain,
        patch(
            "apex.services.deep_research.deep_research_langchain.DeepResearchLangchain._build_final_chain"
        ) as mock_build_final_chain,
    ):
        agent_chain = AsyncMock()
        agent_chain.ainvoke.side_effect = [
            ('{"thought": "compute needed", "action": {"tool": "python_repl", "input": {"code": "print(1+1)"}}}'),
            '{"thought": "done", "final_answer": "final_answer"}',
        ]
        mock_build_agent_chain.return_value = agent_chain

        final_chain_mock = AsyncMock()
        final_chain_mock.ainvoke.return_value = "final_answer"
        mock_build_final_chain.return_value = final_chain_mock

        result = await deep_research_langchain.invoke([{"role": "user", "content": "q"}])

        # Tool history includes python_repl usage
        assert any(t["tool"] == "python_repl" for t in result[1])
        assert result[0] == "final_answer"

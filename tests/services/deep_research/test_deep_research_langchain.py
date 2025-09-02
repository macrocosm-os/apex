from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

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
    """Test invoke method when documents are provided in the body."""
    messages = [{"role": "user", "content": "test question"}]
    docs = [{"page_content": "doc1"}, {"page_content": "doc2"}]
    body = {"documents": docs}

    with (
        patch.object(
            deep_research_langchain, "_create_vector_store", new_callable=AsyncMock
        ) as mock_create_vector_store,
        patch.object(deep_research_langchain, "_create_compression_retriever") as mock_create_compression_retriever,
        patch.object(deep_research_langchain, "_create_summary_chain") as mock_create_summary_chain,
        patch.object(deep_research_langchain, "_create_research_chain") as mock_create_research_chain,
        patch("apex.services.deep_research.deep_research_langchain.PromptTemplate") as mock_prompt_template,
        patch("apex.services.deep_research.deep_research_langchain.StrOutputParser"),
    ):
        mock_compression_retriever = MagicMock()
        mock_compression_retriever.ainvoke = AsyncMock(return_value=[Document(page_content="compressed_doc")])
        mock_create_compression_retriever.return_value = mock_compression_retriever

        mock_summary_chain = MagicMock()
        mock_summary_chain.ainvoke = AsyncMock(return_value="summary")
        mock_create_summary_chain.return_value = mock_summary_chain

        mock_research_chain = MagicMock()
        mock_research_chain.ainvoke = AsyncMock(return_value="research_report")
        mock_create_research_chain.return_value = mock_research_chain

        final_chain = MagicMock()
        final_chain.ainvoke = AsyncMock(return_value="final_answer")
        mock_prompt_template.return_value.__or__.return_value.__or__.return_value = final_chain

        await deep_research_langchain.invoke(messages, body)

        mock_websearch.search.assert_not_called()
        mock_create_vector_store.assert_called_once()


@pytest.mark.asyncio
async def test_invoke_with_websearch(deep_research_langchain, mock_websearch):
    """Test invoke method when no documents are in the body, falling back to websearch."""
    messages = [{"role": "user", "content": "test question"}]
    mock_websearch.search.return_value = [MagicMock(content="web_doc", url="http://a.com")]

    with (
        patch.object(
            deep_research_langchain, "_create_vector_store", new_callable=AsyncMock
        ) as mock_create_vector_store,
        patch.object(deep_research_langchain, "_create_compression_retriever") as mock_create_compression_retriever,
        patch.object(deep_research_langchain, "_create_summary_chain") as mock_create_summary_chain,
        patch.object(deep_research_langchain, "_create_research_chain") as mock_create_research_chain,
        patch("apex.services.deep_research.deep_research_langchain.PromptTemplate") as mock_prompt_template,
        patch("apex.services.deep_research.deep_research_langchain.StrOutputParser"),
    ):
        mock_compression_retriever = MagicMock()
        mock_compression_retriever.ainvoke = AsyncMock(return_value=[Document(page_content="compressed_doc")])
        mock_create_compression_retriever.return_value = mock_compression_retriever

        mock_summary_chain = MagicMock()
        mock_summary_chain.ainvoke = AsyncMock(return_value="summary")
        mock_create_summary_chain.return_value = mock_summary_chain

        mock_research_chain = MagicMock()
        mock_research_chain.ainvoke = AsyncMock(return_value="research_report")
        mock_create_research_chain.return_value = mock_research_chain

        final_chain = MagicMock()
        final_chain.ainvoke = AsyncMock(return_value="final_answer")
        mock_prompt_template.return_value.__or__.return_value.__or__.return_value = final_chain

        await deep_research_langchain.invoke(messages)

        mock_websearch.search.assert_called_once_with(query="test question", max_results=5)
        mock_create_vector_store.assert_called_once()


@pytest.mark.asyncio
async def test_invoke_no_documents_found(deep_research_langchain, mock_websearch):
    """Test invoke when no documents are found from any source."""
    messages = [{"role": "user", "content": "test question"}]
    mock_websearch.search.return_value = []

    result = await deep_research_langchain.invoke(messages)

    assert result[0] == "Could not find any information on the topic."
    assert result[1] == deep_research_langchain.tool_history
    assert isinstance(result[2], list)


@pytest.mark.asyncio
async def test_full_invoke_flow(deep_research_langchain, mock_websearch):
    """Test the full, successful execution flow of the invoke method."""
    messages = [{"role": "user", "content": "test question"}]
    question = messages[-1]["content"]
    web_docs = [MagicMock(content="web_doc", url="http://a.com")]
    compressed_docs = [Document(page_content="compressed_doc")]
    summary = "summary"
    research_report = "research_report"
    final_answer = "final_answer"

    mock_websearch.search.return_value = web_docs

    with (
        patch("apex.services.deep_research.deep_research_langchain.FAISS") as mock_faiss,
        patch(
            "apex.services.deep_research.deep_research_langchain.ContextualCompressionRetriever"
        ) as mock_compression_retriever_class,
        patch("apex.services.deep_research.deep_research_langchain.LLMChainFilter") as mock_llm_chain_filter,
        patch.object(deep_research_langchain, "_create_summary_chain") as mock_create_summary_chain,
        patch.object(deep_research_langchain, "_create_research_chain") as mock_create_research_chain,
        patch("apex.services.deep_research.deep_research_langchain.PromptTemplate") as mock_prompt_template,
        patch("apex.services.deep_research.deep_research_langchain.StrOutputParser"),
    ):
        mock_vector_store = AsyncMock()
        mock_faiss.afrom_documents = AsyncMock(return_value=mock_vector_store)

        mock_compression_retriever = AsyncMock()
        mock_compression_retriever.ainvoke.return_value = compressed_docs
        mock_compression_retriever_class.return_value = mock_compression_retriever

        mock_summary_chain = AsyncMock()
        mock_summary_chain.ainvoke.return_value = summary
        mock_create_summary_chain.return_value = mock_summary_chain

        mock_research_chain = AsyncMock()
        mock_research_chain.ainvoke.return_value = research_report
        mock_create_research_chain.return_value = mock_research_chain

        final_chain = AsyncMock()
        final_chain.ainvoke.return_value = final_answer
        mock_prompt_template.return_value.__or__.return_value.__or__.return_value = final_chain

        result = await deep_research_langchain.invoke(messages)

        mock_websearch.search.assert_called_once_with(query=question, max_results=5)
        mock_faiss.afrom_documents.assert_called_once()
        mock_llm_chain_filter.from_llm.assert_called_once()
        mock_compression_retriever.ainvoke.assert_called_once_with(question)
        mock_summary_chain.ainvoke.assert_called_once_with({"context": compressed_docs, "question": question})
        mock_research_chain.ainvoke.assert_called_once_with({"context": compressed_docs, "question": question})
        final_chain.ainvoke.assert_called_once_with(
            {"summary": summary, "research_report": research_report, "question": question}
        )
        assert result[0] == final_answer
        assert result[1] == deep_research_langchain.tool_history
        assert isinstance(result[2], list)

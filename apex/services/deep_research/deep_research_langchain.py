from typing import Any

from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from apex.common.config import Config
from apex.services.deep_research.deep_research_base import DeepResearchBase
from apex.services.llm.llm_embed import LLMEmbed
from apex.services.websearch.websearch_base import WebSearchBase
from apex.services.websearch.websearch_tavily import WebSearchTavily


class DeepResearchLangchain(DeepResearchBase):
    def __init__(
        self,
        key: str,
        base_url: str,
        emb_base_url: str,
        summary_model: str,
        research_model: str,
        compression_model: str,
        final_model: str,
        websearch: WebSearchBase,
        emb_key: str | None = None,
        summary_base_url: str | None = None,
        research_base_url: str | None = None,
        compression_base_url: str | None = None,
        final_base_url: str | None = None,
        summary_key: str | None = None,
        research_key: str | None = None,
        compression_key: str | None = None,
        final_key: str | None = None,
    ):
        self.websearch = websearch
        self.tool_history: list[dict[str, str]] = []
        self.embedding_model = LLMEmbed(
            base_url=emb_base_url,
            key=emb_key if emb_key is not None else key,
        )
        self.summary_model = ChatOpenAI(
            model_name=summary_model,
            openai_api_key=summary_key if summary_key is not None else key,
            openai_api_base=summary_base_url if summary_base_url is not None else base_url,
            max_retries=3,
            temperature=0.01,
        )
        self.research_model = ChatOpenAI(
            model_name=research_model,
            openai_api_key=research_key if research_key is not None else key,
            openai_api_base=research_base_url if research_base_url is not None else base_url,
            max_retries=3,
            temperature=0.01,
        )
        self.compression_model = ChatOpenAI(
            model_name=compression_model,
            openai_api_key=compression_key if compression_key is not None else key,
            openai_api_base=compression_base_url if compression_base_url is not None else base_url,
            max_retries=3,
            temperature=0.01,
        )
        self.final_model = ChatOpenAI(
            model_name=final_model,
            openai_api_key=final_key if final_key is not None else key,
            openai_api_base=final_base_url if final_base_url is not None else base_url,
            max_retries=3,
            temperature=0.01,
        )

    async def _create_vector_store(self, documents: list[Document]) -> BaseRetriever:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        docs = text_splitter.split_documents(documents)

        custom_embeddings = _CustomEmbeddings(self.embedding_model)

        vector_store: VectorStore = await FAISS.afrom_documents(docs, custom_embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 12})

    def _create_summary_chain(self) -> RunnableSerializable[dict[str, Any], str]:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Summarize the following context to answer the question.
Context: {context}
Question: {question}
Helpful Answer:
""",
        )
        return prompt | self.summary_model | StrOutputParser()

    def _create_compression_retriever(self, retriever: BaseRetriever) -> BaseRetriever:
        compressor = LLMChainFilter.from_llm(self.compression_model)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
        )

    def _create_research_chain(self) -> RunnableSerializable[dict[str, Any], str]:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Generate a comprehensive research report based on the provided context.
Context: {context}
Question: {question}
Research Report:
""",
        )
        return prompt | self.research_model | StrOutputParser()

    async def invoke(
        self, messages: list[dict[str, str]], body: dict[str, Any] | None = None
    ) -> tuple[str, list[dict[str, str]]]:  # type: ignore[override]
        # Clear tool history for each new invocation
        self.tool_history = []
        question = messages[-1]["content"]
        documents: list[Document] = []
        if body and "documents" in body and body["documents"]:
            documents = [
                Document(page_content=doc["page_content"])
                for doc in body["documents"]
                if doc and "page_content" in doc and doc["page_content"] is not None
            ]

        if not documents:
            # Track websearch in tool history
            self.tool_history.append({"tool": "websearch", "args": question})
            websites = await self.websearch.search(query=question, max_results=5)
            for website in websites:
                if website.content:
                    documents.append(Document(page_content=str(website.content), metadata={"url": website.url}))

        if not documents:
            return "Could not find any information on the topic.", self.tool_history

        retriever = await self._create_vector_store(documents)
        if not retriever:
            return "Could not create a vector store from the documents.", self.tool_history

        compression_retriever = self._create_compression_retriever(retriever)

        summary_chain = self._create_summary_chain()
        research_chain = self._create_research_chain()

        compressed_docs: list[Document] = await compression_retriever.ainvoke(question)

        summary: str = await summary_chain.ainvoke({"context": compressed_docs, "question": question})

        research_report: str = await research_chain.ainvoke({"context": compressed_docs, "question": question})

        final_prompt = PromptTemplate(
            input_variables=["summary", "research_report", "question"],
            template="""Based on the following summary and research report, provide a final answer to the question.
Summary: {summary}
Research Report: {research_report}
Question: {question}
Final Answer:
""",
        )
        final_chain = final_prompt | self.final_model | StrOutputParser()

        final_answer: str = await final_chain.ainvoke(
            {"summary": summary, "research_report": research_report, "question": question}
        )
        return final_answer, self.tool_history


class _CustomEmbeddings(Embeddings):  # type: ignore
    def __init__(self, model: LLMEmbed):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("This model only supports async embedding.")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("This model only supports async embedding.")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.model.invoke(inputs=texts)

    async def aembed_query(self, text: str) -> list[float]:
        embeddings = await self.model.invoke(inputs=[text])
        if not embeddings:
            return []
        return embeddings[0]


if __name__ == "__main__":
    import asyncio
    import time

    config = Config.from_file(path="config/testnet.yaml")

    websearch = WebSearchTavily(**config.websearch.kwargs)
    deep_researcher = DeepResearchLangchain(**config.deep_research.kwargs, websearch=websearch)

    # Create a dummy request.
    dummy_messages = [{"role": "user", "content": "What is the purpose of subnet 1 in Bittensor?"}]
    dummy_body: dict[str, Any] = {}

    # Run the invoke method.
    async def main() -> None:
        timer_start = time.perf_counter()
        result, tool_history = await deep_researcher.invoke(dummy_messages, dummy_body)
        logger.debug("Answer:", result)
        logger.debug("Tool History:", tool_history)
        timer_end = time.perf_counter()
        logger.debug(f"Time elapsed: {timer_end - timer_start:.2f}s")

    asyncio.run(main())

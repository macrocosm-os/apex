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
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
            max_tokens=800,
        )
        self.research_model = ChatOpenAI(
            model_name=research_model,
            openai_api_key=research_key if research_key is not None else key,
            openai_api_base=research_base_url if research_base_url is not None else base_url,
            max_retries=3,
            temperature=0.01,
            max_tokens=1200,
        )
        self.compression_model = ChatOpenAI(
            model_name=compression_model,
            openai_api_key=compression_key if compression_key is not None else key,
            openai_api_base=compression_base_url if compression_base_url is not None else base_url,
            max_retries=3,
            temperature=0.01,
            max_tokens=600,
        )
        self.final_model = ChatOpenAI(
            model_name=final_model,
            openai_api_key=final_key if final_key is not None else key,
            openai_api_base=final_base_url if final_base_url is not None else base_url,
            max_retries=3,
            temperature=0.01,
            max_tokens=1600,
        )
        # Caution: PythonREPL can execute arbitrary code on the host machine.
        # Use with caution and consider sandboxing for untrusted inputs.
        self.python_repl = PythonREPL()

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
The report should be long-form (800-1200 words) and include the sections:
- Executive Summary
- Key Findings
- Evidence (quote or paraphrase context with attributions)
- Limitations and Uncertainties
- Conclusion

Explain reasoning explicitly in prose. Prefer depth over breadth.

Context: {context}
Question: {question}
Research Report:
""",
        )
        return prompt | self.research_model | StrOutputParser()

    async def invoke(
        self, messages: list[dict[str, str]], body: dict[str, Any] | None = None
    ) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:  # type: ignore[override]
        # Agentic, iterative deep research with a single websearch tool.
        self.tool_history = []
        reasoning_traces: list[dict[str, Any]] = []

        question = messages[-1]["content"]

        # Seed notes with any provided documents
        notes: list[str] = []
        if body and "documents" in body and body["documents"]:
            for doc in body["documents"]:
                if doc and "page_content" in doc and doc["page_content"] is not None:
                    content = str(doc["page_content"])[:1000]
                    notes.append(f"Provided document snippet: {content}")

        # Iterative loop using research model to choose actions
        max_iterations = 20
        step_index = 0

        # Track discovered sources from websearch for citations
        collected_sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        def _render_sources(max_items: int = 12) -> str:
            if not collected_sources:
                return "(none)"
            lines: list[str] = []
            for i, src in enumerate(collected_sources[:max_items], start=1):
                title = src.get("title") or "untitled"
                url = src.get("url") or ""
                lines.append(f"[{i}] {title} - {url}")
            return "\n".join(lines)

        def _render_notes(max_items: int = 8) -> str:
            if not notes:
                return "(none yet)"
            clipped = notes[-max_items:]
            return "\n".join(f"- {item}" for item in clipped)

        def _build_agent_chain() -> RunnableSerializable[dict[str, Any], str]:
            prompt = PromptTemplate(
                input_variables=["question", "notes", "sources"],
                template=(
                    "You are DeepResearcher, a meticulous, tool-using research agent.\n"
                    "You can use exactly these tools: websearch, python_repl.\n\n"
                    "Tool: websearch\n"
                    "- description: Search the web for relevant information.\n"
                    "- args: keys: 'query' (string), 'max_results' (integer <= 10)\n\n"
                    "Tool: python_repl\n"
                    "- description: A Python shell for executing Python commands.\n"
                    "- note: Print values to see output, e.g., `print(...)`.\n"
                    "- args: keys: 'code' (string: valid python command).\n\n"
                    "Follow an iterative think-act-observe loop. "
                    "Prefer rich internal reasoning over issuing many tool calls.\n"
                    "Spend time thinking: produce substantial, explicit reasoning in each 'thought'.\n"
                    "Avoid giving a final answer too early. Aim for at least 6 detailed thoughts before finalizing,\n"
                    "unless the question is truly trivial. "
                    "If no tool use is needed in a step, still provide a reflective 'thought'\n"
                    "that evaluates evidence, identifies gaps, and plans the next step.\n\n"
                    "Always respond in strict JSON. Use one of the two schemas:\n\n"
                    "1) Action step (JSON keys shown with dot-paths):\n"
                    "- thought: string\n"
                    "- action.tool: 'websearch' | 'python_repl'\n"
                    "- action.input: for websearch -> {{query: string, max_results: integer}}\n"
                    "- action.input: for python_repl -> {{code: string}}\n\n"
                    "2) Final answer step:\n"
                    "- thought: string\n"
                    "- final_answer: string\n\n"
                    "In every step, make 'thought' a detailed paragraph (120-200 words) that:\n"
                    "- Summarizes what is known and unknown so far\n"
                    "- Justifies the chosen next action or decision not to act\n"
                    "- Evaluates evidence quality and cites source numbers when applicable\n"
                    "- Identifies risks, uncertainties, and alternative hypotheses\n\n"
                    "Executive Summary, Key Findings, Evidence, Limitations, Conclusion.\n"
                    "Use inline numeric citations like [1], [2] that refer to Sources.\n"
                    "Include a final section titled 'Sources' listing the numbered citations.\n\n"
                    "Question:\n{question}\n\n"
                    "Notes and observations so far:\n{notes}\n\n"
                    "Sources (use these for citations):\n{sources}\n\n"
                    "Respond with JSON only."
                ),
            )
            return prompt | self.research_model | StrOutputParser()

        agent_chain = _build_agent_chain()

        while step_index < max_iterations:
            step_index += 1
            agent_output: str = await agent_chain.ainvoke(
                {
                    "question": question,
                    "notes": _render_notes(),
                    "sources": _render_sources(),
                }
            )

            parsed = self._safe_parse_json(agent_output)
            if parsed is None:
                reasoning_traces.append(
                    {
                        "step": f"iteration-{step_index}",
                        "model": getattr(self.research_model, "model_name", "unknown"),
                        "output": agent_output,
                        "error": "Failed to parse JSON from agent output",
                    }
                )
                # Add a note to steer next iteration toward valid JSON
                notes.append("Agent output was not valid JSON. Please respond with valid JSON per schema.")
                continue

            thought = str(parsed.get("thought", ""))

            # Final answer branch
            if "final_answer" in parsed:
                final_answer = str(parsed.get("final_answer", ""))
                reasoning_traces.append(
                    {
                        "step": f"iteration-{step_index}",
                        "model": getattr(self.research_model, "model_name", "unknown"),
                        "thought": thought,
                        "final_answer": final_answer,
                    }
                )
                return final_answer, self.tool_history, reasoning_traces

            # Action branch (only websearch supported)
            action = parsed.get("action") or {}
            if action.get("tool") == "websearch":
                action_input = action.get("input") or {}
                query = str(action_input.get("query", question))
                try:
                    max_results = int(action_input.get("max_results", 5))
                except Exception:
                    max_results = 5
                max_results = max(1, min(10, max_results))

                self.tool_history.append({"tool": "websearch", "args": query})
                websites = await self.websearch.search(query=query, max_results=max_results)

                observations: list[str] = []
                for idx, website in enumerate(websites[:max_results]):
                    if website.content:
                        snippet = str(website.content)[:500]
                        observations.append(
                            f"Result {idx + 1}: {website.title or website.url or 'untitled'}\n{snippet}"
                        )
                    # Track source metadata for citations
                    url = getattr(website, "url", "") or ""
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        collected_sources.append(
                            {
                                "url": url,
                                "title": website.title or "",
                            }
                        )

                observation_text = "\n\n".join(observations) if observations else "No results returned."
                notes.append(f"Thought: {thought}")
                notes.append(f'Observation from websearch (q="{query}"):\n{observation_text}')
                reasoning_traces.append(
                    {
                        "step": f"iteration-{step_index}",
                        "model": getattr(self.research_model, "model_name", "unknown"),
                        "thought": thought,
                        "action": {"tool": "websearch", "query": query, "max_results": max_results},
                        "observation": observation_text[:1000],
                    }
                )
                continue

            if action.get("tool") == "python_repl":
                action_input = action.get("input") or {}
                code = str(action_input.get("code", "")).strip()
                # Record the tool use (truncate long code for history)
                self.tool_history.append({"tool": "python_repl", "args": code[:200]})

                if not code:
                    observation_text = "python_repl received empty code."
                else:
                    try:
                        # PythonREPL returns only printed output (may include trailing newline)
                        repl_output = self.python_repl.run(code)
                        observation_text = repl_output if repl_output else "(no output)"
                    except Exception as e:  # noqa: BLE001
                        observation_text = f"Error while executing code: {e}"

                notes.append(f"Thought: {thought}")
                notes.append(f"Observation from python_repl:\n{observation_text}")
                reasoning_traces.append(
                    {
                        "step": f"iteration-{step_index}",
                        "model": getattr(self.research_model, "model_name", "unknown"),
                        "thought": thought,
                        "action": {"tool": "python_repl", "code": code[:500]},
                        "observation": observation_text[:1000],
                    }
                )
                continue

            # Unknown action or schema
            reasoning_traces.append(
                {
                    "step": f"iteration-{step_index}",
                    "model": getattr(self.research_model, "model_name", "unknown"),
                    "thought": thought,
                    "error": f"Unsupported action or schema: {action}",
                }
            )
            notes.append("Agent returned an unsupported action. Use the websearch tool or provide final_answer.")

        # Fallback: if loop ends without final answer, ask final model to synthesize from notes
        final_prompt = PromptTemplate(
            input_variables=["question", "notes", "sources"],
            template=(
                "You are a senior researcher. Write a research report with sections:\n"
                "Executive Summary, Key Findings, Evidence, Limitations, Conclusion.\n"
                "Use inline numeric citations like [1], [2] that refer to Sources.\n"
                "At the end, include a 'Sources' section listing the numbered citations.\n\n"
                "Question:\n{question}\n\n"
                "Notes:\n{notes}\n\n"
                "Sources:\n{sources}\n\n"
                "Research Report:"
            ),
        )
        final_chain = final_prompt | self.final_model | StrOutputParser()
        final_report: str = await final_chain.ainvoke(
            {
                "question": question,
                "notes": _render_notes(12),
                "sources": _render_sources(20),
            }
        )
        reasoning_traces.append(
            {
                "step": "final-fallback",
                "model": getattr(self.final_model, "model_name", "unknown"),
                "output": final_report,
            }
        )
        return final_report, self.tool_history, reasoning_traces

    def _safe_parse_json(self, text: str) -> dict[str, Any] | None:
        """Attempt to parse a JSON object from model output.

        Tries full parse, fenced code extraction, and best-effort substring extraction.
        """
        import json
        import re

        # Direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
            return None
        except Exception:
            pass

        # Extract first JSON code fence
        fence_match = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", text)
        if fence_match:
            candidate = fence_match.group(1)
            try:
                obj2 = json.loads(candidate)
                if isinstance(obj2, dict):
                    return obj2
                return None
            except Exception:
                pass

        # Heuristic: find first '{' and last '}' and try parse
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate2 = text[start : end + 1]
            try:
                obj3 = json.loads(candidate2)
                if isinstance(obj3, dict):
                    return obj3
                return None
            except Exception:
                return None
        return None


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
    dummy_messages = [
        {
            "role": "user",
            "content": """In the study of convex sets, why might two closed convex sets fail to have a strictly
            separating hyperplane, even if they are disjoint? What geometric or topological properties could
            prevent strict separation, and how does this contrast
            with the case where strict separation is possible? Can you provide an intuitive example where such
            a scenario occurs, and explain the underlying reasoning?""",
        }
    ]
    dummy_body: dict[str, Any] = {}

    # Run the invoke method.
    async def main() -> None:
        timer_start = time.perf_counter()
        result, tool_history, reasoning_traces = await deep_researcher.invoke(dummy_messages, dummy_body)
        print(f"Answer: {result}")
        print(f"Tool History: {tool_history}")
        print(f"Reasoning Traces: {reasoning_traces}")
        timer_end = time.perf_counter()
        print(f"Time elapsed: {timer_end - timer_start:.2f}s")

    asyncio.run(main())

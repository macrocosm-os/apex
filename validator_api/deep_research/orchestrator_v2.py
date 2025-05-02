import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from validator_api.deep_research.utils import extract_content_from_stream, parse_llm_json, with_retries
from validator_api.serializers import CompletionsRequest, WebRetrievalRequest
from validator_api.web_retrieval import web_retrieval


def make_chunk(text):
    chunk = json.dumps({"choices": [{"delta": {"content": text}}]})
    return f"data: {chunk}\n\n"


def get_current_datetime_str() -> str:
    """Returns a nicely formatted string of the current date and time"""
    return datetime.now().strftime("%B %d, %Y")


class LLMQuery(BaseModel):
    """Records a single LLM API call with its inputs and outputs"""

    messages: list[dict]  # The input messages
    raw_response: str  # The raw response from the LLM
    parsed_response: Any | None = None  # The parsed response (if applicable)
    step_name: str  # Name of the step that made this query
    timestamp: float  # When the query was made
    model: str  # Which model was used


async def search_web(question: str, n_results: int = 2, completions=None) -> dict:
    """
    Takes a natural language question, generates an optimized search query, performs web search,
    and returns a referenced answer based on the search results.

    Args:
        question: The natural language question to answer
        n_results: Number of search results to retrieve
        completions: Function to make completions request

    Returns:
        dict containing the answer, references, and search metadata
    """
    # Generate optimized search query
    query_prompt = """Given a natural language question, generate an optimized web search query.
    Focus on extracting key terms and concepts while removing unnecessary words.
    Format your response as a single line containing only the optimized search query."""

    messages = [{"role": "system", "content": query_prompt}, {"role": "user", "content": question}]

    optimized_query, query_record = await make_mistral_request(
        messages, "optimize_search_query", completions=completions
    )

    # Perform web search
    for i in range(3):
        try:
            search_results = await web_retrieval(WebRetrievalRequest(search_query=optimized_query, n_results=n_results))
            if search_results.results:
                break
        except BaseException:
            logger.warning(f"Try {i+1} failed")
    if not search_results.results:
        search_results = {"results": []}

    # Generate referenced answer
    answer_prompt = f"""Based on the provided search results, generate a concise but well-structured answer to the question.
    Include inline references to sources using markdown format [n] where n is the source number.

    Question: {question}

    Search Results:
    {json.dumps([{
        'index': i + 1,
        'content': result.content,
        'url': result.url
    } for i, result in enumerate(search_results.results)], indent=2)}

    Format your response as a JSON object with the following structure:
    {{
        "answer": "Your detailed answer with inline references [n]",
        "references": [
            {{
                "number": n,
                "url": "Source URL"
            }}
        ]
    }}"""

    messages = [
        {"role": "system", "content": answer_prompt},
        {"role": "user", "content": "Please generate a referenced answer based on the search results."},
    ]

    raw_answer, answer_record = await make_mistral_request_with_json(
        messages, "generate_referenced_answer", completions=completions
    )
    try:
        answer_data = parse_llm_json(raw_answer)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Mistral API response as JSON: {e}")
        raise

    return {
        "question": question,
        "optimized_query": optimized_query,
        "answer": answer_data["answer"],
        "references": answer_data["references"],
        "raw_results": [{"snippet": r.content, "url": r.url} for r in search_results.results],
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(json.JSONDecodeError),
)
async def make_mistral_request_with_json(
    messages: list[dict],
    step_name: str,
    completions: Callable[[CompletionsRequest], Awaitable[StreamingResponse]],
):
    """Makes a request to Mistral API and records the query"""
    raw_response, query_record = await make_mistral_request(messages, step_name, completions)
    try:
        parse_llm_json(raw_response)  # Test if the response is jsonable
        return raw_response, query_record
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Mistral API response as JSON: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(BaseException),
)
async def make_mistral_request(
    messages: list[dict], step_name: str, completions: Callable[[CompletionsRequest], Awaitable[StreamingResponse]]
) -> tuple[str, LLMQuery]:
    """Makes a request to Mistral API and records the query"""

    model = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"
    temperature = 0.15
    top_p = 1
    max_tokens = 128000
    sample_params = {
        "top_p": top_p,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": False,
    }
    logger.info(f"Making request to Mistral API with model: {model}")
    request = CompletionsRequest(
        messages=messages,
        model=model,
        stream=False,
        sampling_parameters=sample_params,
    )
    # Iterate over the response then collect the content
    response = await completions(request)
    response_content = response.choices[0].message.content  # await extract_content_from_stream(response)
    logger.info(f"Response content: {response_content}")
    if not response_content:
        raise ValueError(f"No response content received from Mistral API, response: {response}")
    if "Error" in response_content:
        raise ValueError(f"Error in Mistral API response: {response_content}")
    query_record = LLMQuery(
        messages=messages, raw_response=response_content, step_name=step_name, timestamp=time.time(), model=model
    )

    return response_content, query_record


class Step(BaseModel):
    title: str
    content: str
    next_step: str | None = None
    summary: str | None = None

    def __str__(self):
        return f"Title: {self.title}\nContent: {self.content}\nNext Step: {self.next_step}\nSummary: {self.summary}"


class StepManager(BaseModel):
    steps: list[Step]

    def __str__(self):
        output = "Here is the list of steps that were already completed:\n\n"
        for i, step in enumerate(self.steps):
            output += f"Step {i+1}:\n{step}\n\n"
        return output


class Tool(ABC):
    """Base class for tools that can be used by the orchestrator"""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does and how to use it"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters"""
        pass


class WebSearchTool(Tool):
    """Tool for performing web searches and getting referenced answers"""

    def __init__(self, completions=None):
        self.completions = completions

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return """Searches the web to answer a question. Provides a referenced answer with citations.
        Input parameters:
        - question: The natural language question to answer
        - n_results: (optional) Number of search results to use (default: 5)

        Returns a dictionary containing:
        - question: Original question asked
        - optimized_query: Search query used
        - answer: Detailed answer with inline references [n]
        - references: List of numbered references with titles and URLs
        - raw_results: Raw search results used"""

    async def execute(self, question: str, n_results: int = 2) -> dict:
        return await search_web(question=question, n_results=n_results, completions=self.completions)


class ToolRequest(BaseModel):
    """A request to execute a specific tool"""

    tool_name: str
    parameters: dict
    purpose: str  # Why this tool execution is needed for the current step


class ToolResult(BaseModel):
    """Result of executing a tool"""

    tool_name: str
    parameters: dict
    result: Any
    purpose: str


class OrchestratorV2(BaseModel):
    todo_list: str | None = None
    current_step: int | None = None
    user_messages: str | None = None
    max_steps: int = 10
    completed_steps: StepManager = StepManager(steps=[])
    query_history: list[LLMQuery] = []
    tool_history: list[ToolResult] = []
    completions: Optional[Callable[[CompletionsRequest], Awaitable[StreamingResponse]]] = None
    tools: dict[str, Tool] = {}

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize tools with the completions function
        self.tools = {"web_search": WebSearchTool(completions=self.completions)}

    async def assess_question_suitability(
        self, question: str, completions: Callable[[CompletionsRequest], Awaitable[StreamingResponse]] = None
    ) -> dict:
        logger.info(f"assess_question_suitability: {question}")
        logger.info(f"completions: {completions}")
        logger.info(f"self.completions: {self.completions}")

        """
        Assesses whether a question is suitable for deep research or if it can be answered directly.

        Args:
            question: The user's question to assess

        Returns:
            dict containing assessment results with:
            - is_suitable: Boolean indicating if deep research is needed
            - reason: Explanation of the assessment
            - direct_answer: Simple answer if question doesn't need deep research
        """

        assessment_prompt = f"""You are part of Apex, a Deep Research Assistant. Your purpose is to assess whether a question is suitable for deep research or if it can be answered directly. The current date and time is {get_current_datetime_str()}.

                            Task:
                            Evaluate the given question and determine if it:

                            1. Requires deep research (complex topics, factual research, analysis of multiple sources, or needs verification through web search)
                            2. Can be answered directly (simple questions, greetings, opinions, or well-known facts that do not require research)

                            # Definitions
                            ## Deep research questions typically:
                            - Seek factual information that may require up-to-date or verified data (e.g., prices, event times, current status)
                            - Involve complex topics with nuance, such as technical processes, system design, or multi-step methodologies
                            - Request detailed breakdowns, plans, or analysis grounded in domain-specific knowledge (e.g., engineering, AI development)
                            - Require synthesis of information from multiple or external sources
                            - Involve comparing different perspectives, approaches, or technologies
                            - Would reasonably benefit from web search, expert resources, or tool use to provide a comprehensive answer

                            ## Questions NOT suitable for deep research include:
                            - Simple greetings or conversational remarks (e.g., "How are you?", "Hello")
                            - Basic opinions that don't require factual grounding or research
                            - Simple, well-known facts that don't need verification (e.g., "The sky is blue")
                            - Requests for purely imaginative content like poems, stories, or fictional narratives
                            - Personal questions about the AI assistant (e.g., "What's your favorite color?")
                            - Questions with obvious or unambiguous answers that don't benefit from external tools or elaboration

                            Response Format:
                            Format your response as a JSON object with the following structure:
                            {{
                                "is_suitable": boolean,  // true if deep research or a web search is needed, false if not
                                "reason": "Brief explanation of why the question does or doesn't need deep research",
                                "direct_answer": "If the question doesn't need deep research, provide a direct answer here. Otherwise, null."
                            }}
                            """

        messages = [
            {"role": "system", "content": assessment_prompt},
            {"role": "user", "content": question},
        ]

        assessment_result, query_record = await make_mistral_request_with_json(
            messages, "assess_question_suitability", completions=self.completions
        )

        try:
            assessment_data = parse_llm_json(assessment_result)
            query_record.parsed_response = assessment_data
            return assessment_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse question assessment output as JSON: {e}")
            return {
                "is_suitable": True,
                "reason": "Unable to assess question suitability due to parsing error. Proceeding with deep research.",
                "direct_answer": None,
            }

    @with_retries(max_retries=3)
    async def plan_tool_executions(self) -> list[ToolRequest]:
        """Uses mistral LLM to plan which tools to execute for the current step"""
        logger.info(f"Planning tool executions for step {self.current_step}")

        tools_description = "\n\n".join([f"Tool: {name}\n{tool.description}" for name, tool in self.tools.items()])
        # TODO: Remove the 2 tools at a time constraint
        prompt = f"""You are planning the use of tools to gather information for the current step in a complex task. The current date and time is {get_current_datetime_str()}.

                    Available Tools:
                    {tools_description}

                    Current todo list (✓ marks completed steps):
                    {self.todo_list}

                    Previous steps completed:
                    {self.completed_steps}

                    Your task is to determine what tool executions, if any, are needed for the next unchecked step in the todo list.
                    You can request multiple executions of the same tool with different parameters if needed. Constrain yourself to only using 2 tools at a time.

                    Format your response as a JSON array of tool requests, where each request has:
                    - tool_name: Name of the tool to execute
                    - parameters: Dictionary of parameters for the tool
                    - purpose: Why this tool execution is needed for the current step

                    If no tools are needed, return an empty array.

                    Example response:
                    [
                        {{
                            "tool_name": "web_search",
                            "parameters": {{"question": "What are the latest developments in quantum computing?"}},
                            "purpose": "To gather recent information about quantum computing advances"
                        }}
                    ]
                  """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Please plan the necessary tool executions for the next step."},
        ]

        plan_output, query_record = await make_mistral_request_with_json(
            messages, f"plan_tools_step_{self.current_step}", completions=self.completions
        )

        try:
            try:
                tool_requests = parse_llm_json(plan_output)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool planning output as JSON: {e}. Plan output: {plan_output}")
                raise
            query_record.parsed_response = tool_requests
            self.query_history.append(query_record)

            # Validate tool requests
            validated_requests = []
            for req in tool_requests:
                if req["tool_name"] not in self.tools:
                    logger.warning(f"Ignoring request for unknown tool: {req['tool_name']}")
                    continue
                validated_requests.append(ToolRequest(**req))

            if validated_requests:
                logger.info(f"Planned {len(validated_requests)} tool executions")
            return validated_requests

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool planning output as JSON: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing required key in tool planning output: {e}")
            raise

    async def execute_tools(self, tool_requests: list[ToolRequest]) -> list[ToolResult]:
        """Executes the requested tools concurrently and records their results"""

        async def execute_single_tool(request: ToolRequest) -> ToolResult | None:
            """Helper function to execute a single tool and handle exceptions"""
            logger.info(f"Executing {request.tool_name} - Purpose: {request.purpose}")
            tool = self.tools[request.tool_name]

            try:
                result = await tool.execute(**request.parameters)
                return ToolResult(
                    tool_name=request.tool_name, parameters=request.parameters, result=result, purpose=request.purpose
                )
            except Exception as e:
                logger.error(f"Failed to execute {request.tool_name}: {e}")
                return None

        # Execute all tool requests concurrently
        tool_results = await asyncio.gather(*[execute_single_tool(request) for request in tool_requests])

        # Filter out None results (from failed executions) and record successful results
        results = [result for result in tool_results if result is not None]
        self.tool_history.extend(results)

        return results

    async def run(self, messages):
        logger.info("Starting orchestration run")
        self.user_messages = messages

        # Always take the last user message as the question
        question = messages[-1]["content"]
        logger.info(f"self.completions: {self.completions}")
        # First assess if the question is suitable for deep research
        question_assessment = await self.assess_question_suitability(question, self.completions)

        # If the question is not suitable for deep research, return a direct answer
        if not question_assessment["is_suitable"]:
            logger.info(f"Question not suitable for deep research: {question_assessment['reason']}")
            yield make_chunk(question_assessment["direct_answer"])
            yield "data: [DONE]\n\n"
            return

        # Continue with deep research process
        yield make_chunk("## Generating Research Plan\n")
        await self.generate_todo_list()
        yield make_chunk(f"## Research Plan\n{self.todo_list}\n")

        for step in range(self.max_steps):
            self.current_step = step + 1
            logger.info(f"Step {step + 1}/{self.max_steps}")

            # Plan and execute tools for this step
            yield make_chunk(f"\n## Step {step + 1}: Planning Tools\n")
            tool_requests = await self.plan_tool_executions()
            if tool_requests:
                for request in tool_requests:
                    yield make_chunk(f"\n## Executing {request.tool_name}\n{request.purpose}\n")
                results = await self.execute_tools(tool_requests)
                for result in results:
                    yield make_chunk(f"\n### Tool Results\n{result.tool_name} execution complete\n")

            yield make_chunk(f"\n## Analyzing Step {step + 1}\n")
            thinking_result = await self.do_thinking()
            yield make_chunk(f"\n## Step {step + 1} Summary\n{thinking_result.summary}\n")

            if thinking_result.next_step == "generate_final_answer":
                logger.info("Generating final answer")
                yield make_chunk("\n## Generating Final Answer\n")
                final_answer = await self.generate_final_answer()
                yield make_chunk(f"\n## Final Answer\n{final_answer}\n")
                return

            yield make_chunk("\n## Updating Research Plan\n")
            await self.update_todo_list()
            yield make_chunk("\n## Research Plan Updated\n")

        yield make_chunk("\n## Generating Final Answer\n")
        final_answer = await self.generate_final_answer()
        yield make_chunk(f"\n# Final Answer\n{final_answer}\n")
        yield "data: [DONE]\n\n"

    @with_retries(max_retries=3)
    async def generate_todo_list(self):
        """Uses mistral LLM to generate a todo list for the Chain of Thought process"""
        logger.info("Generating initial todo list")

        prompt = """Based on the conversation history provided, create a focused step-by-step todo list that outlines the thought process needed to find the answer to the user's question. Focus on information gathering, analysis, and validation steps.

                    Key principles:
                    1. Break down the problem into clear analytical steps
                    2. Focus on what information needs to be gathered and analyzed
                    3. Include validation steps to verify findings
                    4. Consider what tools might be needed at each step
                    5. DO NOT include report writing or summarization in the steps - that will be handled in the final answer

                    Format your response as a numbered list where each item follows this structure:
                    1. [Analysis/Research Task]: What needs to be investigated or analyzed
                    - Information needed: What specific data or insights we need to gather
                    - Approach: How we'll gather this information (e.g., which tools might help)
                    - Validation: How we'll verify the information is accurate and complete

                    Your todo list should focus purely on the steps needed to find and validate the answer, not on presenting it.
                 """

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Here is the conversation history to base the todo list on:\n{self.user_messages}",
            },
        ]

        response, query_record = await make_mistral_request(
            messages, "generate_todo_list", completions=self.completions
        )
        self.query_history.append(query_record)
        self.todo_list = response
        return self.todo_list

    @with_retries(max_retries=3)
    async def do_thinking(self) -> Step:
        """Uses mistral LLM to generate thinking/reasoning tokens in line with the todo list"""
        logger.info(f"Analyzing step {self.current_step}")

        prompt = f"""
                  You are a systematic problem solver working through a complex task step by step. The current date and time is {get_current_datetime_str()}. You have a todo list to follow, and you're currently on step {self.current_step}. Your goal is to think deeply about this step and provide clear, logical reasoning.

Here is your todo list (✓ marks completed steps):
{self.todo_list}

Find the first unchecked item in the todo list (items without a ✓) and analyze that step. Provide your response in the following JSON format:
{{
    "thinking_step_title": "Title of the current todo list step being analyzed",
    "thoughts": "Your detailed analysis and reasoning about this step, including:
                - Step-by-step reasoning process
                - Consideration of edge cases and potential issues
                - References to previous steps if relevant
                - Validation of your approach
                - Summary of the process that clearly states the answer to the todo list step",
    "summary": "A concise summary of your conclusions and key takeaways from this step",
    "next_action": "Either 'continue_thinking' if there are more unchecked todo steps to process, or 'generate_final_answer' if all steps are checked"
}}"""

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Here is the conversation history to base your thinking on:\n{self.user_messages}",
            },
        ]

        thinking_output, query_record = await make_mistral_request_with_json(
            messages, f"thinking_step_{self.current_step}", completions=self.completions
        )

        try:
            thinking_dict = parse_llm_json(thinking_output)
            query_record.parsed_response = thinking_dict
            self.query_history.append(query_record)

            step = Step(
                title=thinking_dict["thinking_step_title"],
                content=thinking_dict["thoughts"],
                next_step=thinking_dict["next_action"],
                summary=thinking_dict["summary"],
            )
            logger.info(f"Completed analysis: {step.title}")
            self.completed_steps.steps.append(step)
            return step
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse thinking output as JSON: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing required key in thinking output: {e}")
            raise

    @with_retries(max_retries=3)
    async def update_todo_list(self):
        """Uses mistral LLM to update the todo list based on the steps taken"""
        logger.info("Updating todo list")

        prompt = f"""You are responsible for reviewing and updating the todo list based on the latest thinking step.

Current todo list:
{self.todo_list}

Latest completed thinking step:
{self.completed_steps.steps[-1]}

Previous completed steps:
{self.completed_steps.steps[:-1]}

Your task is to:
1. Review the current todo list and completed steps
2. Mark completed items with a checkmark (✓) at the start of the line
3. Determine if any new tasks have emerged from the latest analysis
4. Assess if any existing tasks need to be modified based on new insights
5. Check if any tasks are now redundant or can be removed
6. Ensure task dependencies are still accurate

When marking items as complete:
- Add a "✓ " at the start of any numbered item that has been fully addressed in the completed steps
- The checkmark should be added before the number, like this: "✓ 1. [Task Name]"
- If a task was partially completed, do not add a checkmark
- Keep the original numbering intact, just add the checkmark before the number
- Maintain any existing checkmarks from previous updates

Format your response in the following JSON structure:
{{
    "updated_todo_list": "The complete, updated todo list with checkmarks for completed items",
    "changes_made": [
        "list of specific changes made to the todo list and why"
    ],
    "next_step_number": number,
    "rationale": "Brief explanation of why these updates were necessary"
}}"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Here is the conversation history for context:\n{self.user_messages}"},
        ]

        updated_todo, query_record = await make_mistral_request_with_json(
            messages, f"update_todo_list_step_{self.current_step}", completions=self.completions
        )

        try:
            updated_todo_dict = parse_llm_json(updated_todo)
            query_record.parsed_response = updated_todo_dict
            self.query_history.append(query_record)

            self.todo_list = updated_todo_dict["updated_todo_list"]
            logger.info(f"Updated todo list with {len(updated_todo_dict['changes_made'])} changes")
            return updated_todo_dict
        except BaseException as e:
            logger.error(f"Failed to parse updated todo list as JSON: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing required key in updated todo list: {e}")
            raise

    @with_retries(max_retries=3)
    async def generate_final_answer(self):
        """Uses mistral LLM to generate a final answer to the user's request"""
        logger.info("Generating final answer")
        logger.debug(f"Completed steps for final answer:\n{self.completed_steps}")

        prompt = f"""You are tasked with providing a clear, direct answer to the user's original question based on the analysis performed. The current date and time is {get_current_datetime_str()}. Your goal is to synthesize all the information gathered into a helpful response.

Original user question:
{self.user_messages}

Analysis performed:
TODO list (✓ marks completed steps):
{self.todo_list}

Completed thinking steps:
{self.completed_steps}

Tool execution history:
{json.dumps([{
    'tool': result.tool_name,
    'purpose': result.purpose,
    'result': result.result
} for result in self.tool_history], indent=2)}

Your task is to:
1. Review all the information gathered
2. Synthesize the findings into a clear answer
3. Directly address the user's original question
4. Include relevant supporting evidence and citations
5. Acknowledge any limitations or uncertainties

Format your response as a JSON object with the following structure:
{{
    "direct_answer": "A clear, concise answer to the user's question",
    "detailed_explanation": "A more detailed explanation with supporting evidence and reasoning",
    "sources_and_evidence": [
        {{
            "point": "Key point or claim made",
            "evidence": "Evidence supporting this point",
            "source": "Where this information came from (if applicable)"
        }}
    ],
    "limitations": [
        "Any limitations, caveats, or uncertainties in the answer"
    ]
}}

Focus on providing a helpful, accurate answer to what the user actually asked."""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Please generate a final answer based on the analysis performed."},
        ]

        final_answer, query_record = await make_mistral_request_with_json(
            messages, "generate_final_answer", completions=self.completions
        )
        logger.debug(f"Generated final answer:\n{final_answer}")

        try:
            final_answer_dict = parse_llm_json(final_answer)
            query_record.parsed_response = final_answer_dict
            self.query_history.append(query_record)

            return final_answer_dict
        except BaseException as e:
            logger.error(f"Failed to parse final answer as JSON: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing required key in final answer: {e}")
            raise


if __name__ == "__main__":

    async def main():
        orchestrator = OrchestratorV2()
        try:
            # We would need a real completions function here, but since this is just an example,
            # we'll use None and it will fail gracefully
            async for chunk in orchestrator.run(
                messages=[{"role": "user", "content": "How can I implement a prompt engineering project?"}],
                completions=None,
            ):
                print(chunk)
        except Exception as e:
            print(f"An error occurred: {e}")

    asyncio.run(main())

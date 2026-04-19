from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from Day_33.config import GROQ_API_KEY, MODEL_NAME, validate_config


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


@tool
def count_characters(text: str) -> int:
    """Count the number of characters in the provided text."""
    return len(text)


def get_tools() -> List:
    """Return the list of Python functions exposed as LangChain tools."""
    return [add_numbers, multiply_numbers, count_characters]


def get_llm() -> ChatGroq:
    """Create the Groq chat model."""
    validate_config()
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0,
    )


def build_tool_map(tools: List) -> Dict[str, object]:
    """Map tool names to callable tool objects."""
    return {tool_item.name: tool_item for tool_item in tools}


def run_tool_calling_agent(user_query: str) -> str:
    """
    Let the LLM decide when to call Python functions as tools.
    The loop continues until the model returns a final text response.
    """
    tools = get_tools()
    tool_map = build_tool_map(tools)
    llm_with_tools = get_llm().bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Use tools whenever a calculation "
                "or text operation is needed. After using tools, provide a clear final answer."
            )
        ),
        HumanMessage(content=user_query),
    ]

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            selected_tool = tool_map[tool_call["name"]]
            tool_result = selected_tool.invoke(tool_call["args"])

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )


def main() -> None:
    print("LangChain Tool Calling Demo")
    print("Ask something like: 'What is (12 + 8) * 3?'")
    print("Type 'exit' to stop.\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() == "exit":
            print("Session ended.")
            break

        if not user_query:
            continue

        try:
            answer = run_tool_calling_agent(user_query)
            print(f"AI: {answer}\n")
        except Exception as error:
            print(f"Error: {error}\n")


if __name__ == "__main__":
    main()

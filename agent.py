import json
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --- State ---

class State(TypedDict):
    messages: Annotated[list, add_messages]
    validation_error: str | None  # set if validation fails


# --- Tools ---

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def get_word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)


# LangChain tools are executed locally by ToolNode
langchain_tools = [add, multiply, get_word_length]

# OpenAI built-in tools are executed server-side (no local execution needed)
openai_tools = [{"type": "web_search_preview"}]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(langchain_tools + openai_tools)


def extract_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


# --- Nodes ---

def call_llm(state: State) -> State:
    """Send messages to the LLM and get a response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State) -> str:
    """Route to tools if the LLM called one, otherwise validate."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "validate"


def validate(state: State) -> State:
    """
    Validate the LLM's response before returning it to the user.
    Add your own rules here — this example checks the response is not empty
    and does not contain harmful content.
    """
    last = state["messages"][-1]
    text = extract_text(last.content)

    # Rule 1: response must not be empty
    if not text.strip():
        print("[Validation] Empty response — asking LLM to retry.")
        from langchain_core.messages import HumanMessage
        return {"messages": [HumanMessage(content="Your response was empty. Please try again.")], "validation_error": "empty"}

    # Rule 2: block harmful content (extend this list as needed)
    blocked = ["ignore previous instructions", "jailbreak"]
    for phrase in blocked:
        if phrase in text.lower():
            print("[Validation] Blocked response detected.")
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content="I can't help with that.")], "validation_error": "blocked"}

    # Passed — clear any previous error
    return {"validation_error": None}


def after_validate(state: State) -> Literal["llm", "__end__"]:
    """If validation failed with a retryable error, go back to LLM, otherwise end."""
    if state.get("validation_error") == "empty":
        return "llm"
    return END


# --- Graph ---

tool_node = ToolNode(langchain_tools)

graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.add_node("tools", tool_node)
graph.add_node("validate", validate)

graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")
graph.add_conditional_edges("validate", after_validate)

agent = graph.compile()


# --- History ---

HISTORY_FILE = "history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


history = load_history()


def run(prompt: str) -> str:
    history.append({"role": "user", "content": prompt})
    result = agent.invoke({"messages": history})
    last = result["messages"][-1]
    text = extract_text(last.content)
    history.append({"role": "assistant", "content": text})
    save_history(history)
    return text


if __name__ == "__main__":
    print("Agent ready. Type 'exit' to quit.\n")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue
        print(f"Agent: {run(prompt)}\n")

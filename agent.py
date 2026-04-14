import json
import math
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# --- State ---

class State(TypedDict):
    messages: Annotated[list, add_messages]
    validation_error: str | None
    steps: int  # number of LLM invocations; enforces MAX_STEPS limit


# --- System Prompt ---

SYSTEM_PROMPT = """You are a capable assistant that solves problems through multiple steps.

When tackling a task:
1. You MUST always call `think` first before using any other tool. No exceptions.
2. Call tools as many times as needed — you are not limited to a single action.
3. After each tool result, call `think` again if the result requires interpretation before the next step.
4. Chain results together: use the output of one tool as input to the next.
5. Only produce a final answer once you have gathered and verified all necessary information.

Available tools:
- `think`: Record your reasoning or plan (no side effects — use freely to think out loud).
- `add`, `multiply`: Arithmetic on two numbers.
- `calculate`: Evaluate any mathematical expression (supports +, -, *, /, **, sqrt, log, pi, etc.).
- `get_word_length`: Count characters in a string.

Always show your work. For complex problems, break them into steps and tackle each one.

At the end of every response, add a short "**Tools used:**" section that lists each tool you called and one sentence explaining why you called it."""


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


@tool
def think(thought: str) -> str:
    """Use this tool to reason through a problem step by step.
    Write your plan, intermediate conclusions, or working notes here.
    This has no side effects and does not affect the outside world."""
    return thought


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    Supports operators (+, -, *, /, **, //, %) and math functions
    (sqrt, sin, cos, tan, log, log10, exp, abs, round, pi, e, etc.).
    Examples: '2 ** 10', 'sqrt(144) + 5 * 3', 'pi * 4 ** 2'"""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# LangChain tools are executed locally by ToolNode
langchain_tools = [add, multiply, get_word_length, think, calculate]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(langchain_tools)

MAX_STEPS = 15


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
    messages = list(state["messages"])
    # Prepend system prompt if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm.invoke(messages)
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


def should_continue(state: State) -> str:
    """Route to tools if the LLM made tool calls, otherwise validate.
    Enforce MAX_STEPS to prevent infinite loops."""
    if state.get("steps", 0) >= MAX_STEPS:
        print(f"[Agent] Step limit ({MAX_STEPS}) reached — ending.")
        return "validate"
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
    result = agent.invoke({"messages": history, "steps": 0})
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

"""
REST API Agent Starters — LangGraph vs CrewAI
=============================================
Two complete, runnable agents targeting a demo placeholder API.
Uses Google Gemini free tier (15 RPM / 1500 RPD).

Requirements:
    pip install langgraph langchain-google-genai langchain-core crewai requests

Get your free Gemini API key (no credit card needed):
    https://aistudio.google.com  → "Get API key"

Set your key before running:
    export GOOGLE_API_KEY="AIza..."

Demo target: https://jsonplaceholder.typicode.com  (no auth required)
"""

import os
import json
import time
import random
import functools
import requests
from typing import TypedDict

# ─────────────────────────────────────────────────────────────
# RATE LIMIT PROTECTION  (Option C — throttle + retry on 429)
# ─────────────────────────────────────────────────────────────

# Minimum seconds between any two LLM calls (throttle).
# Gemini free tier allows 15 RPM → 1 call per 4s is safe.
THROTTLE_SECONDS = 4

# How many times to retry after a 429 before giving up.
MAX_RATE_RETRIES = 5

_last_call_time = 0.0   # module-level clock for throttle


def throttle():
    """Sleep just long enough to stay under 15 RPM."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < THROTTLE_SECONDS:
        wait = THROTTLE_SECONDS - elapsed
        print(f"  [throttle] waiting {wait:.1f}s to respect rate limit...")
        time.sleep(wait)
    _last_call_time = time.time()


def with_rate_limit_retry(fn):
    """
    Decorator — if the wrapped call raises a 429 / RateLimitError,
    back off exponentially with jitter and retry up to MAX_RATE_RETRIES times.
    Works for both LangChain LLM calls and raw requests.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RATE_RETRIES + 1):
            throttle()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = (
                    "429" in err
                    or "rate limit" in err
                    or "resource_exhausted" in err   # Gemini's error code
                    or "quota" in err
                )
                if is_rate_limit and attempt < MAX_RATE_RETRIES:
                    # Exponential back-off with jitter: 10s, 20s, 40s ... +/- 2s
                    wait = (2 ** attempt) * 10 + random.uniform(0, 2)
                    print(f"  [rate limit] 429 hit — retry {attempt + 1}/{MAX_RATE_RETRIES} in {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    raise   # re-raise if not a rate limit or retries exhausted
    return wrapper


# ─────────────────────────────────────────────────────────────
# SHARED: Demo placeholder API (JSONPlaceholder)
# ─────────────────────────────────────────────────────────────

DEMO_BASE_URL = "https://jsonplaceholder.typicode.com"


# ══════════════════════════════════════════════════════════════
# 1.  LANGGRAPH AGENT
#     Graph-based: explicit nodes, conditional edges, retry loop
# ══════════════════════════════════════════════════════════════

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ── State schema ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list        # conversation history
    endpoint: str         # REST path to call
    method: str           # HTTP verb
    payload: dict         # body for POST/PUT
    api_response: dict    # raw response from the API
    retry_count: int      # 5xx retry counter
    final_answer: str     # output shown to the user


API_MAX_RETRIES = 3       # retries for 5xx API errors (separate from rate limit)


# ── Tool: HTTP call ──────────────────────────────────────────

@tool
def call_rest_api(endpoint: str, method: str = "GET", payload: dict = None) -> str:
    """
    Call the demo REST API at jsonplaceholder.typicode.com.

    Args:
        endpoint: path like /posts/1 or /users
        method: HTTP verb — GET, POST, PUT, DELETE
        payload: optional JSON body for POST/PUT

    Returns:
        JSON string with status_code and body.
    """
    url = f"{DEMO_BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    try:
        if method.upper() == "GET":
            resp = requests.get(url, headers=headers, timeout=10)
        elif method.upper() == "POST":
            resp = requests.post(url, json=payload or {}, headers=headers, timeout=10)
        elif method.upper() == "PUT":
            resp = requests.put(url, json=payload or {}, headers=headers, timeout=10)
        elif method.upper() == "DELETE":
            resp = requests.delete(url, headers=headers, timeout=10)
        else:
            return json.dumps({"error": f"Unsupported method: {method}"})

        return json.dumps({
            "status_code": resp.status_code,
            "body": resp.json() if resp.content else {},
        })
    except Exception as e:
        return json.dumps({"error": str(e), "status_code": 500})


# ── Gemini LLM setup ─────────────────────────────────────────

lg_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",      # free tier; swap "gemini-1.5-pro" for more power
    temperature=0,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
).bind_tools([call_rest_api])

lg_llm_plain = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)


# ── Node: planner ────────────────────────────────────────────

@with_rate_limit_retry
def _planner_invoke(messages):
    return lg_llm.invoke(messages)

def planner_node(state: AgentState) -> dict:
    system = (
        "You are a REST API planning agent. "
        "Given the user's request, call the correct endpoint on the demo API "
        f"at {DEMO_BASE_URL}. "
        "Available resources: /posts, /users, /todos, /albums, /comments. "
        "Use the call_rest_api tool to execute the call."
    )
    messages = [{"role": "system", "content": system}, *state["messages"]]
    response = _planner_invoke(messages)
    return {"messages": state["messages"] + [response]}


# ── Node: tool executor ──────────────────────────────────────

def tool_executor_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    results = []
    api_response = state.get("api_response", {})

    for tc in last.tool_calls:
        if tc["name"] == "call_rest_api":
            raw = call_rest_api.invoke(tc["args"])
            parsed = json.loads(raw)
            api_response = parsed
            results.append(ToolMessage(content=raw, tool_call_id=tc["id"]))

    return {
        "messages": state["messages"] + results,
        "api_response": api_response,
        "retry_count": state.get("retry_count", 0),
    }


# ── Node: API retry (5xx back-off) ───────────────────────────

def api_retry_node(state: AgentState) -> dict:
    count = state.get("retry_count", 0) + 1
    wait = 2 ** count
    print(f"  [api-retry] 5xx — attempt {count}/{API_MAX_RETRIES}, waiting {wait}s...")
    time.sleep(wait)
    hint = HumanMessage(
        content=f"The API returned a server error. Please retry. Attempt {count}/{API_MAX_RETRIES}."
    )
    return {"messages": state["messages"] + [hint], "retry_count": count}


# ── Node: formatter ──────────────────────────────────────────

@with_rate_limit_retry
def _formatter_invoke(messages):
    return lg_llm_plain.invoke(messages)

def formatter_node(state: AgentState) -> dict:
    api_data = json.dumps(state.get("api_response", {}), indent=2)
    messages = [
        {"role": "system", "content": "Summarise this API response in plain English. Be concise."},
        {"role": "user", "content": f"API response:\n{api_data}"},
    ]
    response = _formatter_invoke(messages)
    return {"final_answer": response.content}


# ── Edge routing ─────────────────────────────────────────────

def should_use_tool(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tool"
    return "format"

def should_retry_api(state: AgentState) -> str:
    status = state.get("api_response", {}).get("status_code", 200)
    retries = state.get("retry_count", 0)
    if status >= 500 and retries < API_MAX_RETRIES:
        return "api_retry"
    return "format"


# ── Build the graph ──────────────────────────────────────────

def build_langgraph_agent():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("execute_tool", tool_executor_node)
    graph.add_node("api_retry", api_retry_node)
    graph.add_node("format", formatter_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", should_use_tool, {
        "execute_tool": "execute_tool",
        "format": "format",
    })
    graph.add_conditional_edges("execute_tool", should_retry_api, {
        "api_retry": "api_retry",
        "format": "format",
    })
    graph.add_edge("api_retry", "planner")
    graph.add_edge("format", END)

    return graph.compile(checkpointer=MemorySaver())


def run_langgraph_agent(user_query: str) -> str:
    agent = build_langgraph_agent()
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "endpoint": "",
        "method": "GET",
        "payload": {},
        "api_response": {},
        "retry_count": 0,
        "final_answer": "",
    }

    print("\n" + "=" * 60)
    print("LANGGRAPH AGENT  (Gemini 1.5 Flash)")
    print("=" * 60)
    print(f"Query: {user_query}\n")

    result = agent.invoke(initial_state, config=config)
    answer = result.get("final_answer", "(no answer produced)")
    print(f"Answer:\n{answer}")
    return answer


# ══════════════════════════════════════════════════════════════
# 2.  CREWAI AGENT
#     Role-based crew: Researcher -> API Caller -> Parser -> Writer
# ══════════════════════════════════════════════════════════════

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool


# ── Shared HTTP tool for CrewAI ──────────────────────────────

class RestApiTool(BaseTool):
    name: str = "REST API caller"
    description: str = (
        "Makes HTTP requests to jsonplaceholder.typicode.com. "
        "Input: JSON string with keys — endpoint (str), method (str, default GET), "
        "payload (dict, optional)."
    )

    def _run(self, input_str: str) -> str:
        try:
            args = json.loads(input_str)
        except json.JSONDecodeError:
            args = {"endpoint": input_str.strip(), "method": "GET"}

        url = f"{DEMO_BASE_URL}{args.get('endpoint', '/posts')}"
        method = args.get("method", "GET").upper()
        payload = args.get("payload", {})

        try:
            if method == "GET":
                resp = requests.get(url, timeout=10)
            elif method == "POST":
                resp = requests.post(url, json=payload, timeout=10)
            elif method == "PUT":
                resp = requests.put(url, json=payload, timeout=10)
            elif method == "DELETE":
                resp = requests.delete(url, timeout=10)
            else:
                return json.dumps({"error": f"Unknown method {method}"})

            return json.dumps({
                "status_code": resp.status_code,
                "body": resp.json() if resp.content else {},
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})


rest_tool = RestApiTool()

# CrewAI uses litellm's provider/model format for Gemini
GEMINI_MODEL = "gemini/gemini-1.5-flash"


# ── Agents ───────────────────────────────────────────────────

researcher = Agent(
    role="API Researcher",
    goal="Determine the correct REST endpoint and HTTP method for the user's request.",
    backstory=(
        "You know the JSONPlaceholder API inside-out. You translate natural language "
        "into precise API call plans: endpoint path, HTTP method, and any payload."
    ),
    verbose=True,
    allow_delegation=False,
    llm=GEMINI_MODEL,
)

api_caller = Agent(
    role="API Caller",
    goal="Execute the REST API call and return the raw response.",
    backstory=(
        "You specialise in reliable HTTP requests. You take an API plan and execute "
        "it with your REST tool, returning the complete response."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[rest_tool],
    llm=GEMINI_MODEL,
)

parser = Agent(
    role="Response Parser",
    goal="Extract and structure the most relevant fields from the API response.",
    backstory=(
        "You are a data wrangler who reads raw JSON and surfaces only what matters."
    ),
    verbose=True,
    allow_delegation=False,
    llm=GEMINI_MODEL,
)

writer = Agent(
    role="Technical Writer",
    goal="Present the parsed data as a clean, plain-English answer.",
    backstory=(
        "You turn structured data into friendly summaries — no jargon, no raw JSON."
    ),
    verbose=True,
    allow_delegation=False,
    llm=GEMINI_MODEL,
)


# ── Crew builder ─────────────────────────────────────────────

def build_crewai_crew(user_query: str) -> Crew:
    research_task = Task(
        description=(
            f"User request: '{user_query}'\n\n"
            "Identify the correct JSONPlaceholder endpoint and HTTP method. "
            'Output a JSON plan: {"endpoint": "...", "method": "...", "payload": {}}'
        ),
        expected_output="A JSON string with endpoint, method, and optional payload.",
        agent=researcher,
    )

    call_task = Task(
        description=(
            "Using the researcher's API plan, call the REST API tool with that "
            "plan as a JSON string input. Return the full response."
        ),
        expected_output="Raw JSON response with status_code and body.",
        agent=api_caller,
        context=[research_task],
    )

    parse_task = Task(
        description=(
            "Given the API response, extract the most relevant fields. "
            "Summarise the first 3 items if it's a list; key fields if it's an object."
        ),
        expected_output="A structured summary of the important data.",
        agent=parser,
        context=[call_task],
    )

    write_task = Task(
        description=(
            f"Original question: '{user_query}'\n\n"
            "Write a concise, plain-English answer using the parsed data."
        ),
        expected_output="A clear, friendly answer to the user's question.",
        agent=writer,
        context=[parse_task],
    )

    return Crew(
        agents=[researcher, api_caller, parser, writer],
        tasks=[research_task, call_task, parse_task, write_task],
        process=Process.sequential,
        verbose=True,
    )


@with_rate_limit_retry
def _crew_kickoff(crew: Crew):
    return crew.kickoff()


def run_crewai_agent(user_query: str) -> str:
    print("\n" + "=" * 60)
    print("CREWAI AGENT  (Gemini 1.5 Flash)")
    print("=" * 60)
    print(f"Query: {user_query}\n")

    crew = build_crewai_crew(user_query)
    result = _crew_kickoff(crew)

    answer = str(result)
    print(f"\nFinal Answer:\n{answer}")
    return answer


# ══════════════════════════════════════════════════════════════
# 3.  RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY is not set.")
        print("Get a free key at https://aistudio.google.com -> 'Get API key'")
        print("Then run:  export GOOGLE_API_KEY='AIza...'")
        exit(1)

    # ── Pick your query ──────────────────────────────────────
    QUERY = "Get the first 3 posts and tell me their titles."
    # QUERY = "Fetch user number 2 and tell me their name and email."
    # QUERY = "Create a new post with title 'Hello World' and body 'Testing the API'."
    # QUERY = "Get all todos for user 1 and tell me how many are completed."

    print("\n  Running both agents on the same query...")
    print(f'    Query: "{QUERY}"')
    print(f"    Throttle: {THROTTLE_SECONDS}s between LLM calls")
    print(f"    Rate limit retries: up to {MAX_RATE_RETRIES} x exponential back-off\n")

    lg_answer = run_langgraph_agent(QUERY)
    crewai_answer = run_crewai_agent(QUERY)

    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 60)
    print(f"\n[LangGraph]\n{lg_answer}")
    print(f"\n[CrewAI]\n{crewai_answer}")

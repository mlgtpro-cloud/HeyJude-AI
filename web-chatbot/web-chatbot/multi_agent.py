"""Multi-agent system.

  1. ORCHESTRATOR (the main LLM) reads the request and decides which specialist agents to run,
     and what each one should do.
  2. Each SPECIALIST AGENT runs its own agent loop with its own instructions and its own tools.
  3. The MAIN LLM receives every agent's output and writes the final answer in the chat.

To add a new agent, add one entry to AGENTS. Nothing else needs to change.
"""
import json
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from agent_core import run_agent
from tools import DEFAULT_TZ, TOOLS

BOT_NAME = os.getenv("BOT_NAME", "HeyJude AI")
CURRENCY = os.getenv("CURRENCY", "MYR")
ORIGIN = os.getenv("ORIGIN", "Kuala Lumpur (KUL)")
MAX_WORKERS = 4  # most specialist agents per request (protects the free API limits)
MAX_NOTE = 3000  # characters of each agent's output passed on


def _common():
    today = datetime.now(ZoneInfo(DEFAULT_TZ)).strftime("%A, %d %B %Y")
    return (f"Today is {today}. Defaults if the user does not say otherwise: currency {CURRENCY}, "
            f"home city/airport {ORIGIN}. Never invent prices, schedules, facts or URLs. If something cannot "
            "be found, write 'not found'. Label estimates as estimates. Give sources as full https URLs, never as "
            "citation markers like 【3†L1-L9】. Keep it concise.")


# ---------------------------------------------------------------- the agents
AGENTS = {
    "researcher": {
        "name": "Researcher", "short": "Researches the web",
        "desc": "General web research on any topic: facts, news, comparisons, how-tos, product information.",
        "tools": ["web_search", "fetch_webpage"], "steps": 6, "last": False,
        "prompt": ("You are the Researcher agent. Complete your task by searching the web (web_search, and "
                   "fetch_webpage to read a result). Report the key facts with source URLs and note where "
                   "sources disagree. Use at most 4 tool calls. Maximum 300 words."),
    },
    "travel": {
        "name": "Travel Agent", "short": "Searches flights & hotels",
        "desc": "Travel specialist: real flight prices and times (with airline and time-window filters), hotels, booking links.",
        "tools": ["search_flights", "web_search", "fetch_webpage", "flight_links", "hotel_links"], "steps": 8, "last": False,
        "prompt": ("You are the Travel Agent. FLIGHTS: always call search_flights for real prices and times "
                   "(use IATA codes or city names; Tokyo is NRT,HND). Pass any airline preference and departure "
                   "time window from your task as the airlines, depart_after and depart_before arguments. If no "
                   "dates are given, assume a 5-night trip starting about 6 weeks from today and say the dates are "
                   "assumed. Make at most 2 search_flights calls. Copy the options from the tool result into a "
                   "markdown table: Airline, Flight, Departs, Arrives, Stops, Price. If the tool says no flight "
                   "matched the time window, say so and show the closest options it listed. If the tool says live "
                   "prices are not set up, say exactly that, give NO prices, times or airlines from memory, and "
                   "only give the flight_links. Never write a table row whose price is 'not found': leave it out. "
                   "HOTELS: use web_search (at most 3 web calls) and list only hotels for which you found a price "
                   "per night and a source URL: name, area, price per night, URL. Then call hotel_links and "
                   "flight_links and list those links. Maximum 350 words."),
    },
    "places": {
        "name": "Places Agent", "short": "Finds attractions",
        "desc": "Travel specialist: attractions, activities, food and local tips for a destination.",
        "tools": ["web_search", "fetch_webpage"], "steps": 6, "last": False,
        "prompt": ("You are the Places Agent. Find places to visit for your task: 6 to 10 attractions or "
                   "activities, each with a one-line reason, typical cost and time needed when known. Add local "
                   "food to try and getting-around tips. Use web_search (and fetch_webpage). At most 4 tool "
                   "calls. Maximum 300 words."),
    },
    "weather": {
        "name": "Weather Agent", "short": "Checks weather & what to wear",
        "desc": "Weather forecast or typical weather for a place and dates, and what to wear or pack (trips and outdoor plans).",
        "tools": ["get_weather", "web_search"], "steps": 5, "last": False,
        "prompt": ("You are the Weather Agent. Give the weather for the place and dates in your task and say "
                   "what to wear and pack. Call get_weather (a real forecast, next 16 days only). For later "
                   "dates use web_search for the typical weather in that month and say it is typical, not a "
                   "forecast. Output a weather summary (temperatures, rain) and then a clothing and packing "
                   "list. At most 3 tool calls. Maximum 200 words."),
    },
    "analyst": {
        "name": "Analyst", "short": "Does the calculations",
        "desc": "Numbers for any topic: costs, budgets, comparisons, calculations, date maths.",
        "tools": ["calculator", "get_current_datetime"], "steps": 6, "last": False,
        "prompt": ("You are the Analyst agent. Do the number work in your task: costs, budgets, comparisons, "
                   "dates. Use the calculator tool for EVERY calculation. Show a markdown table with the "
                   "inputs and results and flag any figure that is only an assumption. Maximum 250 words."),
    },
    "writer": {
        "name": "Writer", "short": "Writes the plan or document",
        "desc": ("Writes any plan, itinerary, report, essay or document from the other agents' notes. "
                 "It runs after the other agents and has no tools."),
        "tools": [], "steps": 1, "last": True,
        "prompt": ("You are the Writer agent. Write what your task asks for (for example an itinerary, plan "
                   "or report) using ONLY the notes from the other agents and the user's request. Organize it "
                   "clearly (for example day by day with morning, afternoon and evening). Do not add facts "
                   "that are not in the notes. Maximum 450 words."),
    },
    "coder": {
        "name": "Coder", "short": "Writes and debugs code",
        "desc": "Programming: writes, explains and debugs code, and checks documentation when needed.",
        "tools": ["web_search", "fetch_webpage"], "steps": 5, "last": False,
        "prompt": ("You are the Coder agent. Write, explain or debug code for your task. Give complete, runnable "
                   "code in code blocks, say how to run it, and point out likely bugs. Use web_search (and "
                   "fetch_webpage) only to check documentation when needed, at most 3 tool calls. "
                   "Maximum 400 words not counting code."),
    },
    "explainer": {
        "name": "Explainer", "short": "Teaches a concept",
        "desc": "Teaches a concept to a student: a simple explanation, an example and a common mistake. No tools.",
        "tools": [], "steps": 1, "last": False,
        "prompt": ("You are the Explainer agent. Explain the topic in your task clearly for a student: simple "
                   "words first, then a short example, then one common mistake. Maximum 300 words."),
    },
    "critic": {
        "name": "Critic", "short": "Checks the other agents' work",
        "desc": "Reviews the other agents' work for errors, gaps and contradictions. It runs after the others and has no tools.",
        "tools": [], "steps": 1, "last": True,
        "prompt": ("You are the Critic agent. Check the other agents' notes against the user's request. List errors, "
                   "missing parts, unsupported claims and contradictions, and say what is solid. Be specific and "
                   "short. Do not rewrite the answer. Maximum 200 words."),
    },
}

# The travel agents are examples of domain specialists. They are off by default (set ENABLE_TRAVEL_AGENTS=1 to use them).
if os.getenv("ENABLE_TRAVEL_AGENTS", "0") != "1":
    for _key in ("travel", "places", "weather"):
        AGENTS.pop(_key, None)

MAX_CUSTOM = 2  # custom agents the orchestrator may invent for a request
CUSTOM_TOOLS = ["web_search", "fetch_webpage", "calculator", "get_current_datetime"]

ORCHESTRATOR_PROMPT = (
    "You are the Orchestrator of a general-purpose multi-agent system that can help with any topic: study, "
    "coding, research, writing, business, data and more. Decide which specialist agents the user's request "
    "needs, and give each one a specific task that contains every relevant detail from the request (names, "
    "dates, numbers, budget, preferences). Available agents:\n{agents}\n\n"
    "If the request needs a specialty that none of them covers, you may add a custom agent in the form "
    "{{\"agent\": \"custom\", \"role\": \"<role name>\", \"task\": \"<instruction>\"}} (at most {custom}). "
    "Return ONLY JSON in this form: {{\"agents\": [{{\"agent\": \"<key>\", \"task\": \"<instruction>\"}}]}}. "
    "Choose 0 to {max} agents in total and only the ones really needed. Use an empty list if the request can "
    "be answered directly. Put the agents that read the others' notes (writer, critic) last, writer before "
    "critic. No text outside the JSON."
)

MAIN_PROMPT = (
    f"You are the Main LLM of {BOT_NAME}, a multi-agent system. If asked your name, say you are {BOT_NAME}. "
    "You receive the user's request and the outputs of the "
    "specialist agents. Combine everything into ONE clear, well-organized final answer in Markdown that "
    "directly answers the request. Keep the agents' facts, numbers and links exactly and do not invent new "
    "facts. Never add flights, prices or times that are not in the agent outputs; if the Travel Agent says live prices are unavailable, say so clearly near the top. Where agents disagree or something was 'not found', say so. If an agent output says the agent failed, mention that this part is missing. End with a short 'Check before you "
    "rely on this' list of assumptions. If no agent outputs are provided, just answer the user directly."
)


# ------------------------------------------------------------------- helpers
def _tools(names):
    return [t for t in TOOLS if t["function"]["name"] in names] or None


def _quiet(gen):
    """Run an agent without showing its text. Only status messages are passed on;
    the caller decides what to do if the result says it failed."""
    while True:
        try:
            ev = next(gen)
        except StopIteration as stop:
            return stop.value
        if ev["t"] == "status":
            yield ev


def _tagged(gen, key, final):
    """Mark an agent's events with its stage key. The main LLM's text is the chat answer (no tag).
    Error events are not passed on: the result says whether the agent failed."""
    while True:
        try:
            ev = next(gen)
        except StopIteration as stop:
            return stop.value
        if ev["t"] == "error":
            continue
        if ev["t"] in ("tool", "tool_result") or (ev["t"] == "text" and not final):
            ev["s"] = key
        yield ev


def _custom_spec(role):
    return {"name": role, "short": "Specialist created for this request", "desc": "", "tools": CUSTOM_TOOLS,
            "steps": 6, "last": False,
            "prompt": (f"You are the {role} agent, a specialist created for this request. Complete your task "
                       "carefully. Use the tools when they help and report facts with source URLs. "
                       "Maximum 300 words.")}


def parse_plan(text, fallback_task, specs):
    """Turn the orchestrator's JSON into [(agent_key, task), ...]. Custom agents are added to specs.
    Falls back to the Researcher."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        items = json.loads(m.group(0)).get("agents", []) if m else None
    except (ValueError, AttributeError):
        items = None
    if items is None or not isinstance(items, list):
        return [("researcher", fallback_task)]
    plan, seen, custom = [], set(), 0
    for it in items:
        if not isinstance(it, dict):
            continue
        key, task = it.get("agent"), it.get("task")
        if not (isinstance(task, str) and task.strip()):
            continue
        if key == "custom":
            role = re.sub(r"[^\w &-]", "", str(it.get("role", ""))).strip()[:30]
            if not role or custom >= MAX_CUSTOM:
                continue
            custom += 1
            key = f"custom{custom}"
            specs[key] = _custom_spec(role)
        elif key not in AGENTS or key in seen:
            continue
        plan.append((key, task.strip()))
        seen.add(key)
    plan = plan[:MAX_WORKERS]
    plan.sort(key=lambda kv: (specs[kv[0]]["last"], kv[0] == "critic"))  # writer after the rest, critic last of all
    return plan


def _stage(key, name, desc, status, final=False):
    return {"t": "stage", "key": key, "name": name, "desc": desc, "status": status, "final": final}


# ---------------------------------------------------------------------- main
def run_multi(cfg, messages):
    """Run the whole system. Yields event dicts for the chat page."""
    request = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    common = _common()
    specs = dict(AGENTS)  # the fixed agents, plus any custom agent the orchestrator creates

    # 1. Orchestrator: which agents, and what should each do?
    yield _stage("orch", "Orchestrator", "Plans which agents to run", "start")
    agent_list = "\n".join(f'- "{k}": {a["desc"]}' for k, a in AGENTS.items())
    system = ORCHESTRATOR_PROMPT.format(agents=agent_list, max=MAX_WORKERS, custom=MAX_CUSTOM) + "\n\n" + common
    result = yield from _quiet(run_agent(cfg, system, messages, None, 1))
    if result["ok"]:
        plan = parse_plan(result["text"], request, specs)
        note = ""
    else:  # planner failed: fall back to one Researcher agent instead of giving up
        plan = [("researcher", request)]
        note = f"The planner failed ({result.get('error', 'unknown error')[:120]}). Using the Researcher only.\n\n"
    if plan:
        shown = note + "\n".join(f"- **{specs[k]['name']}**: {task}" for k, task in plan)
    else:
        shown = "No specialist agents are needed. The main LLM will answer directly."
    yield {"t": "text", "d": shown, "s": "orch"}
    yield {"t": "stage", "key": "orch", "status": "done"}

    # 2. Specialist agents, one after another
    notes = {}
    for key, task in plan:
        a = specs[key]
        yield _stage(key, a["name"], a["short"], "start")
        ctx = f"USER REQUEST:\n{request}\n\nYOUR TASK:\n{task}"
        if a["last"] and notes:  # only the writer needs the other agents' notes
            ctx += "\n\nNOTES FROM THE OTHER AGENTS:\n" + "\n\n".join(
                f"## {specs[k]['name']}\n{v[:MAX_NOTE]}" for k, v in notes.items())
        gen = run_agent(cfg, a["prompt"] + "\n\n" + common, [{"role": "user", "content": ctx}],
                        _tools(a["tools"]), a["steps"])
        result = yield from _tagged(gen, key, False)
        if result["ok"]:
            notes[key] = result["text"]
        else:  # keep going: the other agents' work is still useful
            msg = f"{a['name']} failed: {result.get('error', 'unknown error')[:200]}"
            notes[key] = f"[{msg}]"
            yield {"t": "text", "d": "Warning: " + msg, "s": key}
        yield {"t": "stage", "key": key, "status": "done"}

    # 3. Main LLM: combine every agent's output into the final answer
    yield _stage("main", "Main LLM", "Combines everything into the final answer", "start", final=True)
    if notes:
        body = "\n\n".join(f"## {specs[k]['name']}\n{v[:MAX_NOTE]}" for k, v in notes.items())
        main_msgs = [{"role": "user", "content": f"USER REQUEST:\n{request}\n\nAGENT OUTPUTS:\n{body}"}]
    else:
        main_msgs = messages
    result = yield from _tagged(run_agent(cfg, MAIN_PROMPT + "\n\n" + common, main_msgs, None, 1), "main", True)
    if not result["ok"]:
        if not notes:
            yield {"t": "error", "d": result.get("error", "The main LLM failed.")}
            return
        raw = "\n\n".join(f"## {specs[k]['name']}\n{v}" for k, v in notes.items())
        yield {"t": "text", "d": "The main LLM could not write the final answer "
               f"({result.get('error', 'unknown error')[:120]}). Here are the raw agent outputs:\n\n{raw}"}
    yield {"t": "stage", "key": "main", "status": "done"}
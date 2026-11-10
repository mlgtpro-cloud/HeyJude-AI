"""Auto mode: one small, fast model call decides how each message should be handled.

  chat  - answer directly (simple questions, explanations, writing, code help, easy maths)
  agent - one model that can call tools (live data, web lookups, exact maths, flight search)
  multi - orchestrator + specialist agents + main LLM (big tasks with several parts)

If the router fails or is unsure, it falls back to "agent": that costs the same as chat
when no tool is needed, so a wrong guess is cheap.
"""
import json
import re

from agent_core import run_agent

VALID = ("chat", "agent", "multi")

ROUTER_PROMPT = (
    "You are a router. Decide how to handle the user's latest message. Modes:\n"
    "- chat: simple questions, explanations, greetings and small talk, writing, coding help, easy maths, "
    "anything that can be answered well from general knowledge alone.\n"
    "- agent: needs one or a few tools: current or real-time information (news, today's date or "
    "time, prices, latest versions), looking something up on the web, reading a web page, or big or exact "
    "calculations.\n"
    "- multi: a large task made of several different parts that need different specialists, whose results "
    "are then combined: for example planning a project with research, costs and a schedule, "
    "comparing several products or tools with research and costs, building a study plan with resources and a "
    "schedule, or a multi-part research report.\n"
    "If the user attached an image, choose chat unless they also ask for live data or a big task: the model reads "
    "images itself, and there is no OCR tool.\n"
    "Always choose the simplest mode that can do the job well. If unsure between two, choose the simpler one.\n"
    "Examples: 'whats 1+1' -> chat. 'hello' -> chat. 'explain recursion' -> chat. 'write a python function' "
    "-> chat. 'what is the latest version of Python' -> agent. 'latest news about X' -> agent. 'what is 98765 x 43210' -> "
    "agent. "
    "'plan a software project with research, costs and a schedule' -> multi. 'compare the best laptops under RM3000 for programming' -> multi. 'build a study plan for my data structures exam with resources and a schedule' -> multi.\n"
    "Return ONLY JSON: {\"mode\": \"chat\" or \"agent\" or \"multi\", \"reason\": \"<max 6 words>\"}"
)


def _text(content):
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    t = str(content or "")
    return t if len(t) <= 800 else t[:400] + "\n...\n" + t[-400:]  # keep the start and the end (the question)


def route(cfg, messages, has_image=False):
    """Return (mode, short_reason)."""
    idx = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=None)
    if idx is None:
        return "chat", "empty message"
    note = "LATEST USER MESSAGE:\n" + _text(messages[idx]["content"])
    if has_image:
        note += "\n\n(The user attached an image.)"
    prev = next((m for m in reversed(messages[:idx]) if m["role"] == "assistant"), None)
    if prev:
        note += "\n\nPREVIOUS ASSISTANT REPLY (start):\n" + _text(prev["content"])[:200]

    gen = run_agent(cfg, ROUTER_PROMPT, [{"role": "user", "content": note}], None, 1)
    chunks = []
    while True:
        try:
            ev = next(gen)
        except StopIteration as stop:
            result = stop.value
            break
        if ev["t"] == "text":
            chunks.append(ev["d"])
    if not result.get("ok"):
        return "agent", "router unavailable"
    m = re.search(r"\{.*?\}", "".join(chunks), re.S)
    try:
        data = json.loads(m.group(0))
        mode, reason = data.get("mode"), str(data.get("reason", ""))[:60]
    except (AttributeError, ValueError):
        return "agent", "unclear request"
    if mode not in VALID:
        return "agent", "unclear request"
    if has_image and mode == "multi":  # multi-agent mode is text only
        mode = "agent"
    return mode, reason
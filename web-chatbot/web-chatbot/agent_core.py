"""The reusable agent engine.

call_model()  - one streaming call to the LLM API. Retries by itself on rate limits (429),
                temporary server errors (5xx) and dropped connections.
run_agent()   - the agent loop: model -> tool call -> tool result -> model ... -> answer.
                Also recovers from the model making a broken tool call.
Both app.py (normal chat) and multi_agent.py use this.
"""
import json
import re
import time

import requests

from tools import run_tool

MAX_RETRIES = 3
RETRY_STATUS = (500, 502, 503, 504)  # temporary server problems: worth retrying


def _retry_after(r):
    """How many seconds the API asks us to wait after a 429."""
    try:
        return float(r.headers.get("retry-after")) + 1
    except (TypeError, ValueError):
        pass
    m = re.search(r"(?:(\d+)m)?\s*([\d.]+)s", r.text or "")
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
    return 20.0


def friendly_error(status, text):
    t = (text or "")[:300]
    if status == 429 and ("per day" in t.lower() or "(TPD)" in t):
        m = re.search(r"try again in ([\dhms.]+)", t)
        return ("[Error 429] The free daily limit for this model is used up"
                + (f" (it frees up again in about {m.group(1).rstrip('.')})" if m else "")
                + ". Wait a little, or set a different MODEL in .env and restart.")
    if status == 413 or "too large" in t.lower():
        nums = re.search(r"Limit (\d+), Requested (\d+)", text or "")
        model = re.search(r"model `([^`]+)`", text or "")
        if nums:
            return (f"[Error {status}] This request is too big for the free limit of "
                    f"{model.group(1) if model else 'this model'}: it allows about {int(nums.group(1)):,} tokens "
                    f"per minute and this request needs about {int(nums.group(2)):,}. "
                    "Attach a shorter file, ask a shorter question, or try again later.")
        return (f"[Error {status}] The request was too big for the free API limit. "
                f"Try a shorter request, or wait a minute and try again. ({t[:120]})")
    return f"[Error {status}] {t}"


def _can_fallback(cfg, payload):
    """A second model has its own daily limit. It cannot read images, so skip it for those."""
    fb = cfg.get("fallback_model")
    has_image = any(isinstance(m.get("content"), list) for m in payload["messages"])
    return bool(fb) and fb != payload["model"] and not has_image


_SWITCHED = {}  # model -> (backup model, time until we try the main model again)
SWITCH_MINUTES = 10


def call_model(cfg, payload):
    """Stream one model call. Yields ("text", str), ("wait", seconds),
    ("calls", [..]) when the model wants tools, or ("error", str)."""
    sw = _SWITCHED.get(payload["model"])
    if sw and sw[1] > time.time() and _can_fallback(cfg, payload):  # remembered: do not waste a failed call
        cfg["model"] = sw[0]
        payload = {**payload, "model": sw[0]}
    for attempt in range(MAX_RETRIES + 1):
        calls, wait, got_text, switch = {}, None, False, False
        try:
            with requests.post(
                f"{cfg['base_url']}/chat/completions",
                headers=cfg["headers"], json=payload, stream=True, timeout=90,
            ) as r:
                r.encoding = "utf-8"
                if r.status_code == 429:
                    w = _retry_after(r)
                    if w > 60 and _can_fallback(cfg, payload):  # a long wait: a daily limit is used up
                        switch = True
                    elif w <= 60 and attempt < MAX_RETRIES:
                        wait = w
                elif r.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                    wait = 2.0 * (attempt + 1)
                if wait is None and not switch:
                    if r.status_code != 200:
                        yield ("error", friendly_error(r.status_code, r.text))
                        return
                    for line in r.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            delta = json.loads(chunk)["choices"][0]["delta"]
                        except (KeyError, IndexError, ValueError):
                            continue
                        if delta.get("content"):
                            got_text = True
                            yield ("text", delta["content"])
                        for tc in delta.get("tool_calls") or []:  # tool calls arrive in pieces
                            slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name") and not slot["name"]:
                                slot["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["args"] += fn["arguments"]
        except requests.RequestException as e:
            if not got_text and attempt < MAX_RETRIES:  # safe to retry: nothing was shown yet
                wait = 3.0 * (attempt + 1)
            else:
                yield ("error", f"[Connection error] {e}")
                return
        if switch:
            old, cfg["model"] = payload["model"], cfg["fallback_model"]  # stays switched for the rest of the run
            _SWITCHED[old] = (cfg["model"], time.time() + SWITCH_MINUTES * 60)
            payload = {**payload, "model": cfg["model"]}
            yield ("note", f"Daily limit reached for {old}. Switching to {cfg['model']}...")
            continue
        if wait is not None:
            yield ("wait", wait)
            time.sleep(wait)
            continue
        if calls:
            yield ("calls", [calls[i] for i in sorted(calls)])
        return


def _status(seconds):
    return {"t": "status", "d": f"Waiting {int(seconds)}s (rate limit or temporary error), then retrying..."}


def _shrink(msgs, level):
    """Make a too-big request smaller. Level 1 keeps only the newest user message and what follows it.
    Level 2 also cuts long texts to half."""
    system, rest = msgs[0], msgs[1:]
    last_user = max((i for i, m in enumerate(rest) if m["role"] == "user"), default=0)
    rest = rest[last_user:]
    if level >= 2:
        def half(m):
            c = m.get("content")
            if isinstance(c, str) and len(c) > 1500:
                return {**m, "content": c[: len(c) // 2] + "\n[shortened]"}
            if isinstance(c, list):
                return {**m, "content": [({**p, "text": p["text"][: len(p["text"]) // 2] + "\n[shortened]"}
                                          if p.get("type") == "text" and len(p.get("text", "")) > 1500 else p)
                                         for p in c]}
            return m
        rest = [half(m) for m in rest]
    return [system] + rest


def run_agent(cfg, system, messages, tool_defs=None, max_steps=6, run=run_tool):
    """The agent loop. Yields event dicts; returns {"text": answer, "ok": bool, "error": str}."""
    msgs = [{"role": "system", "content": system}] + list(messages)
    use_tools = bool(tool_defs)
    for step in range(max_steps):
        tool_fail = shrinks = 0
        while True:  # retry this step if the model made a broken tool call or the request was too big
            payload = {"model": cfg["model"], "stream": True, "messages": msgs}
            if use_tools:
                payload["tools"] = tool_defs
                payload["tool_choice"] = "auto"
            text, calls, err = [], None, None
            for kind, val in call_model(cfg, payload):
                if kind == "text":
                    text.append(val)
                    yield {"t": "text", "d": val}
                elif kind == "wait":
                    yield _status(val)
                elif kind == "note":
                    yield {"t": "status", "d": val}
                elif kind == "calls":
                    calls = val
                else:
                    err = val
            if err and use_tools and "tool_use_failed" in err:
                tool_fail += 1
                if tool_fail >= 2:
                    use_tools = False
                    yield {"t": "status", "d": "The model made a broken tool call. Answering without tools..."}
                continue
            if err and shrinks < 2 and (err.startswith("[Error 413]") or "too big" in err):
                shrinks += 1
                msgs = _shrink(msgs, shrinks)
                yield {"t": "status", "d": "The request was too big. Retrying with a shorter version..."}
                continue
            break
        if err:
            yield {"t": "error", "d": err}
            return {"text": "", "ok": False, "error": err}
        if not calls:  # no tool requested: the answer is complete
            return {"text": "".join(text), "ok": True}
        for n, c in enumerate(calls):
            c["id"] = c["id"] or f"call_{step}_{n}"
        msgs.append({
            "role": "assistant",
            "content": "".join(text) or None,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                for c in calls
            ],
        })
        for c in calls:  # run each tool and give the result back to the model
            yield {"t": "tool", "id": c["id"], "name": c["name"], "args": c["args"] or "{}"}
            result = run(c["name"], c["args"])
            yield {"t": "tool_result", "id": c["id"], "result": result[:800]}
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    # Tool steps used up: make one last call with tools switched off, so the model must answer
    msgs.append({"role": "user", "content": "The tool limit is reached. Using only the information gathered "
                 "so far, write your final answer now. Do not call any tools."})
    final = {"model": cfg["model"], "stream": True, "messages": msgs}
    if use_tools:
        final["tools"] = tool_defs
        final["tool_choice"] = "none"
    text = []
    for kind, val in call_model(cfg, final):
        if kind == "text":
            text.append(val)
            yield {"t": "text", "d": val}
        elif kind == "wait":
            yield _status(val)
        elif kind == "note":
            yield {"t": "status", "d": val}
        elif kind == "error":
            yield {"t": "error", "d": val}
            return {"text": "", "ok": False, "error": val}
    if not text:
        note = "\n\n_(Stopped: too many tool steps.)_"
        yield {"t": "text", "d": note}
        return {"text": note, "ok": True}
    return {"text": "".join(text), "ok": True}
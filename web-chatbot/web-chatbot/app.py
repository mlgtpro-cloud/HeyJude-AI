import hmac
import json
import os
from io import BytesIO

from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, stream_with_context

load_dotenv()

from agent_core import run_agent  # noqa: E402  (imported after load_dotenv so .env settings apply)
from tools import TOOLS  # noqa: E402
from multi_agent import run_multi  # noqa: E402
from router import route  # noqa: E402

API_KEY = os.getenv("API_KEY", "")
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "")  # optional: protects a public website so strangers cannot use your free API quota
USER_NAME = os.getenv("USER_NAME", "")  # optional: shown in the greeting, e.g. "Good morning, Lem"
BASE_URL = os.getenv("BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
# If the main model hits its daily free limit, switch to this one (each model has its own daily limit)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", os.getenv("MODEL", "openai/gpt-oss-120b"))  # model for the Auto router
MULTI_MODEL = os.getenv("MULTI_MODEL", os.getenv("MODEL", "openai/gpt-oss-120b"))  # model for Multi-agent mode
# Used automatically when the latest message contains an image
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
BOT_NAME = os.getenv("BOT_NAME", "HeyJude AI")
# Who the assistant is. It is sent at the start of every chat, so a new chat is still HeyJude AI.
IDENTITY = (
    f"Your name is {BOT_NAME}. You are an AI assistant "
    + (f"built by {USER_NAME}" if USER_NAME else "built as a student project")
    + f" that runs on the AI model {MODEL}. If asked who you are or what your name is, say you are {BOT_NAME}. "
    f"If asked which model powers you, say {MODEL}. Never say that you are ChatGPT."
)
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Answer clearly and concisely. "
    "Use markdown and code blocks when they help. Write simple arithmetic in plain text; use LaTeX "
    "($...$ or $$...$$) only for real formulas.",
)
AGENT_PROMPT = (
    "You can use tools. Use `calculator` for any arithmetic instead of computing in your head, "
    "`get_current_datetime` for today's date or the time, `web_search` for recent or uncertain facts, "
    "`fetch_webpage` to read a page found by web_search, `get_weather` for a real forecast, and "
    "`search_flights` for real flight prices and times, and `flight_links` / `hotel_links` for live booking-search links. Call a tool only when it helps, "
    "otherwise answer directly. Use at most 4 tool calls, then write your answer. "
    "When you use web results, give the full source URLs as markdown links, never citation markers like 【3†L1-L9】."
)
MAX_HISTORY = 20  # last N messages sent to the model
MAX_FILE_CHARS = int(os.getenv("MAX_FILE_CHARS", "9000"))  # how much of an attached file is sent (the free API limits request size)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))  # older chat history sent along (the free API limits request size)
MAX_TEXT = 30000  # max characters per message (room for pasted files)
MAX_IMAGE = 5_000_000  # max size of one base64 image string
MAX_STEPS = 6  # max model calls per question (agent loop limit)

app = Flask(__name__)


def clean_content(content, keep_images):
    """Accept a string, or a list of text/image parts (OpenAI format)."""
    if isinstance(content, str):
        return content[:MAX_TEXT]
    if not isinstance(content, list):
        return ""
    texts, images = [], []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(str(part.get("text", ""))[:MAX_TEXT])
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if keep_images and url.startswith("data:image/") and len(url) < MAX_IMAGE:
                images.append({"type": "image_url", "image_url": {"url": url}})
            else:
                texts.append("[an image was attached here]")
    text = "\n".join(texts)
    if images:
        return [{"type": "text", "text": text}] + images
    return text


@app.before_request
def require_password():
    """If ACCESS_PASSWORD is set, the browser asks for it (any user name works)."""
    if not ACCESS_PASSWORD:
        return None
    auth = request.authorization
    if auth and hmac.compare_digest((auth.password or "").encode(), ACCESS_PASSWORD.encode()):
        return None
    return Response("Password needed", 401, {"WWW-Authenticate": 'Basic realm="HeyJude AI"'})


def to_line(ev):
    """One line of the response stream (newline-delimited JSON)."""
    return json.dumps(ev, ensure_ascii=False) + "\n"


def _size(m):
    c = m["content"]
    return len(c) if isinstance(c, str) else sum(len(p.get("text", "")) for p in c if p.get("type") == "text")


def fit_context(messages):
    """Keep the newest messages that fit the size budget. The newest message is always kept."""
    kept, total = [messages[-1]], _size(messages[-1])
    for m in reversed(messages[:-1]):
        if total + _size(m) > MAX_CONTEXT_CHARS:
            break
        kept.append(m)
        total += _size(m)
    kept.reverse()
    while len(kept) > 1 and kept[0]["role"] != "user":  # a chat must start with the user
        kept.pop(0)
    return kept


@app.get("/")
def index():
    return render_template("index.html", model=MODEL, user_name=USER_NAME, file_chars=MAX_FILE_CHARS)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    if mode not in ("auto", "chat", "agent", "multi"):
        mode = "auto"
    raw = data.get("messages", [])[-MAX_HISTORY:]
    messages, has_image = [], False
    for i, m in enumerate(raw):
        if m.get("role") not in ("user", "assistant") or not m.get("content"):
            continue
        # only the newest user message keeps its image (saves tokens); multi-agent mode is text only
        keep = mode != "multi" and i == len(raw) - 1 and m["role"] == "user"
        content = clean_content(m["content"], keep)
        if not content:
            continue
        if isinstance(content, list):
            has_image = True
        messages.append({"role": m["role"], "content": content})

    if not messages:
        return {"error": "Send a message first."}, 400
    if not API_KEY:
        return {"error": "API_KEY is missing. Add it to your .env file and restart."}, 500

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last = messages[-1]["content"]
    has_file = isinstance(last, str) and last.startswith("File: ")  # a PDF, Word or text file was attached
    # keep the request small: an image is sent alone (it costs many tokens), otherwise only recent history
    messages = messages[-1:] if has_image else fit_context(messages)

    def generate():
        try:
            use = mode
            if mode == "auto":
                if has_image or has_file:  # reading a picture or file needs no tools: plain chat is smaller and safer
                    use, reason = "chat", "image attached" if has_image else "file attached"
                else:  # a quick router call picks the best mode for this message (router.py)
                    use, reason = route({"base_url": BASE_URL, "headers": headers, "model": ROUTER_MODEL,
                                         "fallback_model": FALLBACK_MODEL}, messages, has_image)
                yield to_line({"t": "route", "mode": use, "reason": reason})
            if use == "multi":  # orchestrator + specialist agents + main LLM (multi_agent.py)
                cfg = {"base_url": BASE_URL, "headers": headers, "model": MULTI_MODEL, "fallback_model": FALLBACK_MODEL}
                events = run_multi(cfg, messages)
            else:  # single agent (with tools) or plain chat
                cfg = {"base_url": BASE_URL, "headers": headers, "model": VISION_MODEL if has_image else MODEL,
                       "fallback_model": FALLBACK_MODEL}
                system = IDENTITY + " " + SYSTEM_PROMPT + (" " + AGENT_PROMPT if use == "agent" else "")
                events = run_agent(cfg, system, messages, TOOLS if use == "agent" else None, MAX_STEPS)
            for ev in events:
                yield to_line(ev)
            yield to_line({"t": "done"})  # tells the page the answer finished normally
        except Exception as e:  # never let a bug cut the stream silently
            app.logger.exception("chat request failed")
            yield to_line({"t": "error", "d": f"[Server error] {e}"})

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.post("/api/extract")
def extract():
    """Pull plain text out of a PDF or Word (.docx) upload."""
    f = request.files.get("file")
    if not f:
        return {"error": "No file received."}, 400
    data = f.read()
    if len(data) > 8 * 1024 * 1024:
        return {"error": "File is too big (max 8 MB)."}, 400
    name = (f.filename or "").lower()
    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(data))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        elif name.endswith(".docx"):
            from docx import Document

            doc = Document(BytesIO(data))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(c.text for c in row.cells))
            text = "\n".join(parts)
        else:
            return {"error": "Only .pdf and .docx files are handled here."}, 400
    except Exception:
        return {"error": "Could not read that file. It may be damaged or password-protected."}, 400
    text = text.strip()
    if not text:
        return {"error": "No readable text found (scanned PDF?). Try attaching a photo of the page instead."}, 400
    return {"text": text[:MAX_FILE_CHARS], "truncated": len(text) > MAX_FILE_CHARS, "limit": MAX_FILE_CHARS}


if __name__ == "__main__":
    # debug=False: the auto-restarting debug server kills a run in progress when any file changes
    app.run(debug=False, threaded=True)
"""Tools the AI agent can call. Each tool is a normal Python function.

TOOLS is the list of tool descriptions sent to the model (OpenAI function-calling format).
run_tool() executes a tool by name when the model asks for it.
"""
import ast
import ipaddress
import json
import math
import operator
import os
import re
import socket
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, urlparse
from zoneinfo import ZoneInfo

import requests

DEFAULT_TZ = os.getenv("TIMEZONE", "Asia/Kuala_Lumpur")

# ---------------------------------------------------------------- calculator
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs,
    "round": round, "floor": math.floor, "ceil": math.ceil,
}
_NAMES = {"pi": math.pi, "e": math.e}


def _eval(node):
    """Evaluate a maths expression safely (no eval(), only whitelisted syntax)."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent is too large")
        result = _BIN[type(node.op)](left, right)
        if isinstance(result, (int, float)) and abs(result) > 1e300:
            raise ValueError("number is too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expression):
    value = _eval(ast.parse(str(expression).strip(), mode="eval"))
    if isinstance(value, float):
        value = int(value) if value.is_integer() else round(value, 10)
    return f"{expression} = {value}"


# ------------------------------------------------------------ date and time
def get_current_datetime(timezone=None):
    name = timezone or DEFAULT_TZ
    try:
        tz = ZoneInfo(name)
    except Exception:
        return f"Unknown timezone '{name}'. Use an IANA name such as Asia/Kuala_Lumpur."
    now = datetime.now(tz)
    return now.strftime("%A, %d %B %Y, %H:%M:%S") + f" ({name}, UTC{now.strftime('%z')})"


# --------------------------------------------------------------- web search
def web_search(query, max_results=5):
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # older package name
            from duckduckgo_search import DDGS
        results = DDGS(timeout=10).text(str(query), max_results=max(1, min(int(max_results), 8)))
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}")
    return "\n\n".join(lines)


# ------------------------------------------------------------- web page reader
class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer"}

    def __init__(self):
        super().__init__()
        self.parts, self.depth = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth and data.strip():
            self.parts.append(data.strip())


def _is_public(url):
    """Block localhost / private network addresses."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
    except Exception:
        return False
    return True


def fetch_webpage(url):
    url = str(url).strip()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StudyChatbot/1.0)"}
    try:
        for _ in range(4):  # follow a few redirects, checking each one
            if not _is_public(url):
                return "Blocked: only public http(s) pages can be opened."
            r = requests.get(url, headers=headers, timeout=15, stream=True, allow_redirects=False)
            if r.is_redirect and r.headers.get("Location"):
                url = urljoin(url, r.headers["Location"])
                continue
            break
        else:
            return "Too many redirects."
        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200:
            return f"Could not open the page (HTTP {r.status_code})."
        if "html" not in ctype and "text/plain" not in ctype:
            return f"Unsupported content type: {ctype or 'unknown'}."
        raw = b"".join(r.iter_content(chunk_size=65536, decode_unicode=False))[:1_000_000]
        html = raw.decode(r.encoding or "utf-8", errors="replace")
    except requests.RequestException as e:
        return f"Could not open the page: {e}"
    if "text/plain" in ctype:
        text = html
    else:
        parser = _TextExtractor()
        parser.feed(html)
        text = " ".join(parser.parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3000] or "The page has no readable text."


# ------------------------------------------------------------------- weather
_WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
        65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
        81: "rain showers", 82: "heavy showers", 95: "thunderstorm", 96: "thunderstorm with hail",
        99: "thunderstorm with hail"}


def get_weather(city, start_date=None, days=5):
    """Real forecast from Open-Meteo (free, no key). Covers the next 16 days only."""
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": str(city), "count": 1}, timeout=10).json()
        if not geo.get("results"):
            return f"City '{city}' was not found."
        loc = geo["results"][0]
        days = max(1, min(int(days), 7))
        params = {"latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto",
                  "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max"}
        if start_date:
            first = date.fromisoformat(str(start_date))
            today = datetime.now(ZoneInfo(loc.get("timezone") or "UTC")).date()
            if first < today or (first - today).days > 15:
                return ("Forecast is only available for the next 16 days. "
                        "Use web_search for the typical weather in that month.")
            last = min(first + timedelta(days=days - 1), today + timedelta(days=15))
            params["start_date"], params["end_date"] = first.isoformat(), last.isoformat()
        else:
            params["forecast_days"] = days
        d = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10).json()["daily"]
    except ValueError:
        return "Error: start_date must look like 2026-12-10."
    except Exception as e:
        return f"Weather lookup failed: {e}"
    lines = [f"Forecast for {loc['name']}, {loc.get('country', '')}:"]
    for i, day in enumerate(d["time"]):
        lines.append(f"{day}: {_WMO.get(d['weathercode'][i], 'mixed')}, "
                     f"{d['temperature_2m_min'][i]:.0f} to {d['temperature_2m_max'][i]:.0f} C, "
                     f"rain chance {d['precipitation_probability_max'][i]}%")
    return "\n".join(lines)


# ------------------------------------------------------- real flight prices
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "")
CURRENCY = os.getenv("CURRENCY", "MYR")
_CITY_AIRPORTS = {
    "tokyo": "NRT,HND", "osaka": "KIX,ITM", "kyoto": "KIX,ITM", "nagoya": "NGO", "fukuoka": "FUK",
    "sapporo": "CTS", "kuala lumpur": "KUL", "penang": "PEN", "singapore": "SIN", "bangkok": "BKK,DMK",
    "seoul": "ICN,GMP", "hong kong": "HKG", "taipei": "TPE", "bali": "DPS", "jakarta": "CGK",
    "london": "LHR,LGW", "paris": "CDG,ORY", "sydney": "SYD", "melbourne": "MEL", "dubai": "DXB",
    "new york": "JFK,EWR,LGA", "los angeles": "LAX",
}


def _airports(text):
    t = str(text).strip()
    if t.lower() in _CITY_AIRPORTS:
        return _CITY_AIRPORTS[t.lower()]
    codes = [c.strip().upper() for c in t.split(",")]
    if all(len(c) == 3 and c.isalpha() for c in codes):
        return ",".join(codes)
    raise ValueError(f"'{t}' is not a known city. Use IATA airport codes such as NRT,HND.")


def _minutes(hhmm):
    h, m = str(hhmm).strip().split(":")
    h, m = int(h), int(m)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError
    return h * 60 + m


def search_flights(origin, destination, depart_date, return_date=None, depart_after=None,
                   depart_before=None, airlines=None, max_results=6):
    """Real flight prices and times from Google Flights (through SerpApi). The time window and
    airline filters are applied here in code, so the AI cannot invent matches."""
    if not SERPAPI_KEY:
        return ("Live flight prices are not set up (no SERPAPI_API_KEY in the .env file). Do NOT guess "
                "prices, airlines or times. Say that live prices are unavailable and give the flight_links "
                "search links instead.")
    try:
        dep_ids, arr_ids = _airports(origin), _airports(destination)
        date.fromisoformat(str(depart_date))
        if return_date:
            date.fromisoformat(str(return_date))
        lo = _minutes(depart_after) if depart_after else None
        hi = _minutes(depart_before) if depart_before else None
    except ValueError as e:
        return str(e) if str(e) else "Error: dates must look like 2026-12-10 and times like 20:00."
    params = {"engine": "google_flights", "departure_id": dep_ids, "arrival_id": arr_ids,
              "outbound_date": str(depart_date), "currency": CURRENCY, "hl": "en",
              "type": 1 if return_date else 2, "api_key": SERPAPI_KEY}
    if return_date:
        params["return_date"] = str(return_date)
    try:
        data = requests.get("https://serpapi.com/search.json", params=params, timeout=45).json()
    except Exception as e:
        return f"Flight search failed: {e}"
    if data.get("error"):
        return f"Flight search error: {data['error']}"

    offers = []
    for item in (data.get("best_flights") or []) + (data.get("other_flights") or []):
        legs = item.get("flights") or []
        if not legs or item.get("price") is None:
            continue
        try:
            dep = datetime.strptime(legs[0]["departure_airport"]["time"], "%Y-%m-%d %H:%M")
            arr = datetime.strptime(legs[-1]["arrival_airport"]["time"], "%Y-%m-%d %H:%M")
        except (KeyError, ValueError):
            continue
        offers.append({
            "price": item["price"], "dep": dep, "arr": arr,
            "names": list(dict.fromkeys(l.get("airline", "?") for l in legs)),
            "numbers": [l.get("flight_number", "") for l in legs],
            "from": legs[0]["departure_airport"].get("id", ""), "to": legs[-1]["arrival_airport"].get("id", ""),
            "via": [l["arrival_airport"].get("id", "") for l in legs[:-1]],
            "minutes": item.get("total_duration"),
        })
    if not offers:
        return "No flights were found for that route and date."

    want = [w.strip().lower() for w in str(airlines or "").split(",") if w.strip()]

    def keep(o):
        m = o["dep"].hour * 60 + o["dep"].minute
        a, b = (lo if lo is not None else 0), (hi if hi is not None else 1439)
        in_window = (a <= m <= b) if a <= b else (m >= a or m <= b)
        codes = [n.split()[0].lower() for n in o["numbers"] if n]
        airline_ok = not want or any(w in n.lower() for w in want for n in o["names"]) or any(w in codes for w in want)
        return in_window and airline_ok

    def line(i, o):
        plus = f" (+{(o['arr'].date() - o['dep'].date()).days}d)" if o["arr"].date() > o["dep"].date() else ""
        stops = "direct" if not o["via"] else f"{len(o['via'])} stop via {','.join(o['via'])}"
        dur = f"{o['minutes'] // 60}h{o['minutes'] % 60:02d}" if o["minutes"] else "?"
        kind = "round trip" if return_date else "one way"
        return (f"{i}. {', '.join(o['names'])} ({' / '.join(o['numbers'])}) | {o['from']} {o['dep']:%H:%M} -> "
                f"{o['to']} {o['arr']:%H:%M}{plus} | {stops} | {dur} | {CURRENCY} {o['price']:,.0f} {kind}")

    head = (f"Live prices from Google Flights (via SerpApi), {dep_ids} -> {arr_ids}, departing {depart_date}"
            + (f", returning {return_date}" if return_date else "") + ". Prices can change; check before booking.")
    insight = data.get("price_insights") or {}
    if insight.get("typical_price_range"):
        a, b = insight["typical_price_range"][:2]
        head += f"\nTypical price range for this route: {CURRENCY} {a:,.0f} to {b:,.0f}."
    shown = sorted((o for o in offers if keep(o)), key=lambda o: o["price"])[:max(1, min(int(max_results), 10))]
    if shown:
        return head + "\n" + "\n".join(line(i, o) for i, o in enumerate(shown, 1))
    others = sorted(offers, key=lambda o: o["dep"])[:8]
    return (head + "\nNo flight matched the requested time window or airline on that date. "
            "Flights that do exist (by departure time):\n" + "\n".join(line(i, o) for i, o in enumerate(others, 1)))


# ------------------------------------------------------------ booking links
def flight_links(origin, destination, depart_date, return_date=None):
    """Links that open live flight searches (the user checks real prices and books there)."""
    q = f"Flights from {origin} to {destination} on {depart_date}"
    if return_date:
        q += f" returning {return_date}"
    out = ["Google Flights: https://www.google.com/travel/flights?q=" + quote(q)]
    o, d = str(origin).strip().lower(), str(destination).strip().lower()
    if len(o) == 3 and len(d) == 3 and o.isalpha() and d.isalpha():
        try:
            path = f"{o}/{d}/{date.fromisoformat(depart_date).strftime('%y%m%d')}/"
            if return_date:
                path += date.fromisoformat(return_date).strftime("%y%m%d") + "/"
            out.append("Skyscanner: https://www.skyscanner.com.my/transport/flights/" + path)
        except ValueError:
            pass
    return "\n".join(out)


def hotel_links(city, checkin, checkout, guests=2):
    return ("Booking.com: https://www.booking.com/searchresults.html?ss=" + quote(str(city))
            + f"&checkin={checkin}&checkout={checkout}&group_adults={int(guests)}")


# ------------------------------------------------------------------ registry
REGISTRY = {
    "calculator": calculator,
    "get_current_datetime": get_current_datetime,
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
    "search_flights": search_flights,
    "get_weather": get_weather,
    "flight_links": flight_links,
    "hotel_links": hotel_links,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate a maths expression exactly. Supports + - * / // % **, parentheses, "
                       "sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil, pi, e.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. 0.18 * 4250 + 12"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_datetime",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {
            "timezone": {"type": "string", "description": "Optional IANA timezone, e.g. Asia/Tokyo. Defaults to the user's timezone."}}}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for recent or uncertain information. Returns titles, URLs and snippets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "description": "1 to 8, default 5"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_webpage",
        "description": "Open a public web page and return its text. Use it to read a page found by web_search.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "search_flights",
        "description": "Search REAL flight prices and times (Google Flights). Filters by departure time window and airline. "
                       "Always use this for flight prices; never guess them.",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string", "description": "City or IATA code(s), e.g. Kuala Lumpur or KUL"},
            "destination": {"type": "string", "description": "City or IATA code(s), e.g. Tokyo or NRT,HND"},
            "depart_date": {"type": "string", "description": "YYYY-MM-DD"},
            "return_date": {"type": "string", "description": "Optional, YYYY-MM-DD (gives round-trip prices)"},
            "depart_after": {"type": "string", "description": "Optional earliest departure time, 24h HH:MM, e.g. 20:00"},
            "depart_before": {"type": "string", "description": "Optional latest departure time, 24h HH:MM, e.g. 21:00"},
            "airlines": {"type": "string", "description": "Optional airline names or 2-letter codes, comma separated, e.g. Malaysia Airlines,MH"},
            "max_results": {"type": "integer", "description": "1 to 10, default 6"}},
            "required": ["origin", "destination", "depart_date"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Real weather forecast for a city (next 16 days only). For later dates use web_search for typical weather.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "start_date": {"type": "string", "description": "Optional, YYYY-MM-DD"},
            "days": {"type": "integer", "description": "1 to 7, default 5"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "flight_links",
        "description": "Build links that open live flight searches (Google Flights, Skyscanner) so the traveller can check real prices and book.",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string", "description": "City or 3-letter airport code, e.g. KUL"},
            "destination": {"type": "string", "description": "City or 3-letter airport code, e.g. NRT"},
            "depart_date": {"type": "string", "description": "YYYY-MM-DD"},
            "return_date": {"type": "string", "description": "Optional, YYYY-MM-DD"}},
            "required": ["origin", "destination", "depart_date"]}}},
    {"type": "function", "function": {
        "name": "hotel_links",
        "description": "Build a link that opens a live hotel search on Booking.com.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "checkin": {"type": "string", "description": "YYYY-MM-DD"},
            "checkout": {"type": "string", "description": "YYYY-MM-DD"},
            "guests": {"type": "integer"}},
            "required": ["city", "checkin", "checkout"]}}},
]


def run_tool(name, raw_args):
    """Run one tool call from the model. Always returns a string (errors included)."""
    try:
        args = json.loads(raw_args) if raw_args else {}
        if not isinstance(args, dict):
            raise ValueError
    except ValueError:
        return "Error: the arguments were not valid JSON."
    fn = REGISTRY.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'."
    try:
        return str(fn(**args))[:3500]
    except TypeError as e:
        return f"Error: bad arguments ({e})."
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Error: {e}"
"""MV2 forecaster: an open-source forecasting bot for the Metaculus FutureEval
bot tournament, by MV2 of Munim, Inc. Its forecasts are produced entirely by
AI, with no human in the loop.

Pipeline per question:
  1. research: one LLM call with the provider's native web search;
  2. context: related Manifold markets (public API, no key);
  3. forecasts: every model in FORECAST_MODELS answers RUNS_PER_MODEL times;
  4. aggregate: mean log-odds (binary), mean probabilities (multiple choice),
     mean CDF (numeric/discrete);
  5. post the forecast once, then the required private comment with reasoning.

Spot peer scoring counts only a bot's last forecast before a question closes,
so a question already answered is re-forecast once more when it is about to
close and the earlier answer has gone stale (see should_forecast).

Standard library only. Configuration is by environment variable:
  METACULUS_TOKEN   bot account token (required to post)
  LLM_API_KEY       key for the OpenAI-compatible endpoint (OPENROUTER_API_KEY also read)
  LLM_BASE_URL      default https://openrouter.ai/api/v1
  FORECAST_MODELS   comma-separated model ids; default: newest frontier
                    Anthropic/OpenAI/Google models listed by the endpoint's /models
  RESEARCH_MODEL    model used for research; default: first forecast model
  RUNS_PER_MODEL    default 2
  TOURNAMENTS       comma-separated ids or slugs; default 33121,minibench
  REFRESH_BEFORE_CLOSE_H  re-forecast an answered question when it closes within this
                          many hours; default 24. Set to 0 to disable re-forecasting.
  MIN_FORECAST_AGE_H      ...but only if the last answer is at least this old; default 12.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

__version__ = "0.1.5"

METACULUS_API = "https://www.metaculus.com/api"
FALL_2026_FUTUREEVAL_ID = 33121
MINIBENCH_ID = "minibench"
BOT_TESTING_AREA_ID = "bot-testing-area"
DEFAULT_TOURNAMENTS = [str(FALL_2026_FUTUREEVAL_ID), MINIBENCH_ID]
FORECAST_TYPES = ["binary", "multiple_choice", "numeric", "discrete"]
USER_AGENT = f"mv2-forecaster/{__version__} (+https://github.com/munimtechnologies/mv2-forecaster)"

BINARY_FLOOR, BINARY_CEIL = 0.02, 0.98
MC_FLOOR = 0.01
CDF_UNIFORM_WEIGHT = 0.02
CDF_OPEN_BOUND_MASS = 0.001
CDF_MIN_STEP = 5e-5
PERCENTILES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
COMMENT_LIMIT = 9000

# Metaculus's spam filter deactivates accounts over links (2026-10-14: two links
# in a bio took down the owner's account), so a posted comment carries none:
# not our own repo, and not the URLs the research model cites.
LINK_RE = re.compile(r"\[([^\]]*)\]\((?:https?://|www\.)[^)]*\)"
                     r"|(?:https?://|www\.)\S+"
                     r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|io|ai|gov|edu|co|uk|news|info|dev|app)\b(?:/\S*)?",
                     re.IGNORECASE)


def strip_links(text):
    """Drop every URL and bare domain; keep a markdown link's label."""
    return LINK_RE.sub(lambda m: m.group(1) or "[link removed]", text or "")

# Spot peer scoring only counts the last forecast before a question closes, so a
# question we have already answered is re-forecast once more near close, when the
# earlier answer has aged. These are the defaults; main() reads env overrides.
REFRESH_BEFORE_CLOSE_H = 24.0
MIN_FORECAST_AGE_H = 12.0


# --------------------------------------------------------------------- HTTP

def http_json(method, url, token=None, body=None, headers=None, timeout=120):
    data = None
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        hdrs["Authorization"] = f"Token {token}"
    if headers:
        hdrs.update(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


# ---------------------------------------------------------------- Metaculus

def list_open_posts(tournament, token):
    """All open single-question posts of a tournament, following pagination."""
    posts, offset = [], 0
    while True:
        params = urllib.parse.urlencode({
            "limit": 100, "offset": offset, "order_by": "-hotness",
            "forecast_type": ",".join(FORECAST_TYPES), "tournaments": tournament,
            "statuses": "open", "include_description": "true",
        })
        page = http_json("GET", f"{METACULUS_API}/posts/?{params}", token)
        results = page.get("results") or []
        posts.extend(p for p in results if p.get("question"))
        if not page.get("next") or not results:
            return posts
        offset += len(results)


def get_post(post_id, token):
    return http_json("GET", f"{METACULUS_API}/posts/{post_id}/", token)


def already_forecast(question):
    try:
        return question["my_forecasts"]["latest"]["forecast_values"] is not None
    except (KeyError, TypeError):
        return False


def _parse_iso(s):
    """Parse a Metaculus ISO timestamp to an aware UTC datetime, or None."""
    if not s:
        return None
    try:
        d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d


def hours_until_close(question, now=None):
    """Hours from now until the question's scheduled close; None if unknown."""
    close = _parse_iso(question.get("scheduled_close_time"))
    if close is None:
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    return (close - now).total_seconds() / 3600.0


def hours_since_last_forecast(question, now=None):
    """Hours since our latest forecast (my_forecasts.latest.start_time is epoch
    seconds); None if there is no forecast or no usable timestamp."""
    try:
        ts = question["my_forecasts"]["latest"]["start_time"]
    except (KeyError, TypeError):
        return None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    return (now.timestamp() - ts) / 3600.0


def should_forecast(question, now=None, before_close_h=REFRESH_BEFORE_CLOSE_H,
                    min_age_h=MIN_FORECAST_AGE_H):
    """Whether to forecast this question now.

    Always forecast a question never answered. For one already answered, only
    re-forecast (a spot-score refresh) when it closes within before_close_h hours
    AND the last answer is at least min_age_h hours old. Missing or unparseable
    timing data falls back to the conservative 'forecast once' behavior, and
    before_close_h <= 0 disables re-forecasting entirely.
    """
    if not already_forecast(question):
        return True
    if before_close_h <= 0:
        return False
    close_in = hours_until_close(question, now)
    age = hours_since_last_forecast(question, now)
    if close_in is None or age is None:
        return False
    return 0 < close_in <= before_close_h and age >= min_age_h


def forecast_payload(question_id, qtype, value):
    payload = {"question": question_id, "source": "api", "probability_yes": None,
               "probability_yes_per_category": None, "continuous_cdf": None}
    if qtype == "binary":
        payload["probability_yes"] = value
    elif qtype == "multiple_choice":
        payload["probability_yes_per_category"] = value
    elif qtype in ("numeric", "discrete"):
        payload["continuous_cdf"] = value
    else:
        raise ValueError(f"unsupported question type: {qtype}")
    return payload


def post_forecast(question_id, qtype, value, token):
    body = [forecast_payload(question_id, qtype, value)]
    return http_json("POST", f"{METACULUS_API}/questions/forecast/", token, body)


def post_private_comment(post_id, text, token):
    body = {"text": text[:COMMENT_LIMIT], "parent": None, "included_forecast": True,
            "is_private": True, "on_post": post_id}
    return http_json("POST", f"{METACULUS_API}/comments/create/", token, body)


# ---------------------------------------------------------------------- LLM

def llm_chat(model, prompt, base_url, api_key, temperature=0.4, web=False, timeout=300):
    body = {"model": model, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    if web:
        body["plugins"] = [{"id": "web", "engine": "native"}]
    resp = http_json("POST", base_url.rstrip("/") + "/chat/completions", body=body,
                     headers={"Authorization": f"Bearer {api_key}",
                              "X-Title": "mv2-forecaster"}, timeout=timeout)
    return resp["choices"][0]["message"]["content"] or ""


# Across FutureEval's first year the single biggest driver of a bot's score was
# whether it used a frontier model (Metaculus pros-vs-bots write-ups, 2026), so
# each provider's top tier is preferred, falling back to the next tier only if
# the endpoint does not list it. Tiers are tried in order; the newest id wins.
_TIER_RULES = [
    ("anthropic/claude-", [("opus",), ("sonnet",)], ("haiku",)),
    ("openai/gpt-", [()], ("mini", "nano", "oss", "pro", "image", "audio", "codex", "chat", "search", "realtime")),
    ("google/gemini-", [("pro",)], ("flash", "image", "tts", "lite", "exp")),
]


def pick_models(models_json):
    """Newest frontier model per provider from an OpenRouter-style /models list.

    For each provider the preference tiers are tried in order; the first tier
    with any match supplies the newest matching id.
    """
    rows = models_json.get("data") or []
    picked = []
    for prefix, pref_tiers, banned in _TIER_RULES:
        chosen = None
        for tier in pref_tiers:
            best = None
            for row in rows:
                mid = str(row.get("id", ""))
                low = mid.lower()
                if not low.startswith(prefix) or ":" in low or "[" in low:
                    continue
                if tier and not any(m in low for m in tier):
                    continue
                if any(b in low for b in banned):
                    continue
                created = row.get("created") or 0
                if best is None or created > best[0]:
                    best = (created, mid)
            if best:
                chosen = best[1]
                break
        if chosen:
            picked.append(chosen)
    return picked


# ------------------------------------------------------------------ context

def manifold_markets(title, limit=5):
    """Open binary Manifold markets that match the question title (best effort)."""
    term = re.sub(r"[^\w\s%$.-]", " ", title)[:120]
    params = urllib.parse.urlencode({"term": term, "limit": limit, "filter": "open",
                                     "contractType": "BINARY"})
    try:
        rows = http_json("GET", f"https://api.manifold.markets/v0/search-markets?{params}", timeout=30)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return []
    out = []
    for row in rows if isinstance(rows, list) else []:
        if row.get("probability") is None:
            continue
        out.append({"question": row.get("question", ""), "probability": row["probability"],
                    "url": row.get("url", "")})
    return out


def format_markets(markets):
    if not markets:
        return "None found."
    return "\n".join(f"- {m['question']}: {m['probability']:.0%} ({m['url']})" for m in markets)


# ------------------------------------------------------------------ prompts

def question_text(q):
    parts = [f"Question: {q.get('title', '')}"]
    for key, label in (("description", "Background"), ("resolution_criteria", "Resolution criteria"),
                       ("fine_print", "Fine print")):
        if q.get(key):
            parts.append(f"{label}:\n{q[key]}")
    if q.get("scheduled_resolve_time"):
        parts.append(f"Scheduled resolution: {q['scheduled_resolve_time']}")
    if q.get("unit"):
        parts.append(f"Unit: {q['unit']}")
    return "\n\n".join(parts)


def research_prompt(q, today):
    return (f"Today is {today}. You are the research assistant of a forecaster. Search the web for "
            "the most recent news, official data and expert statements relevant to the question "
            "below. Report facts with dates and sources, the current status quo, scheduled events "
            "before resolution, and relevant base rates. Do not give a forecast.\n\n" + question_text(q))


def forecast_prompt(q, today, research, markets):
    qtype = q.get("type")
    head = (f"Today is {today}. You are a careful superforecaster. Think about base rates, the status "
            "quo (things usually stay as they are), the time left until resolution, and how the "
            "resolution criteria could be met or missed. Write your reasoning briefly, then give "
            "the answer in the exact format requested.\n\n"
            f"{question_text(q)}\n\nResearch notes:\n{research or 'None.'}\n\n"
            f"Related Manifold markets (may not match this question exactly):\n{format_markets(markets)}\n\n")
    if qtype == "binary":
        return head + "End with one line exactly like: Probability: NN%"
    if qtype == "multiple_choice":
        opts = "\n".join(f"{o}: NN%" for o in q.get("options") or [])
        return head + ("End with one line per option, in this order and with these exact names, "
                       f"probabilities summing to 100%:\n{opts}")
    low, high = numeric_bounds_text(q)
    lines = "\n".join(f"Percentile {p}: X" for p in PERCENTILES)
    return head + (f"{low}\n{high}\nGive wide intervals: unknown unknowns are common. End with these "
                   f"lines, values as plain numbers (no units, no commas), increasing:\n{lines}")


def numeric_bounds_text(q):
    s = q.get("scaling") or {}
    lo_open, hi_open = bound_flags(q)
    low = (f"The question's lower bound is {s.get('range_min')}"
           + (" (open: the outcome can be lower)." if lo_open else " (closed: the outcome cannot be lower)."))
    high = (f"The question's upper bound is {s.get('range_max')}"
            + (" (open: the outcome can be higher)." if hi_open else " (closed: the outcome cannot be higher)."))
    return low, high


# ------------------------------------------------------------------ parsing

_NUM = r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?|-?\.\d+"

# Characters an LLM commonly wraps around a multiple-choice label or puts between
# the label and its colon: whitespace, markdown emphasis (* _ ~ `), a blockquote
# marker or a bullet. Allowed on either side of the colon so that, e.g.,
# "**Option A:** 30%" parses (the colon sits inside the bold markers).
_MC_PAD = r"[\s>*_~`\-•]*"
# An optional leading list marker such as "1." or "2)" before the label, so a
# numbered list ("1. Option A: 30%") is read too; the older ^\W* anchor could not
# cross the leading digit and silently dropped the whole answer.
_MC_LIST = r"(?:\d+[.)]" + _MC_PAD + r")?"


def _to_float(text):
    return float(text.replace(",", ""))


def parse_binary(text):
    # The prompt asks the model to "End with one line exactly like: Probability: NN%",
    # but frontier models often drop the % or give a bare fraction (e.g.
    # "Probability: 73" or "Probability: 0.73"). The old pattern required a literal
    # "%", so those answers returned None and that run was silently discarded from
    # the ensemble -- the same silent-drop failure that v0.1.4 fixed for multiple
    # choice. Make the "%" optional (like parse_multiple_choice) and read the value:
    # an explicit "%" or any value above 1 is a percentage (divide by 100); a bare
    # value in [0, 1] is already a probability.
    hits = re.findall(r"Probability\s*:\s*(" + _NUM + r")\s*(%?)", text, re.I)
    if not hits:
        return None
    raw, pct = hits[-1]
    v = _to_float(raw)
    if pct or v > 1.0:
        v = v / 100.0
    return v if 0.0 <= v <= 1.0 else None


def parse_multiple_choice(text, options):
    probs = {}
    for opt in options:
        pat = (r"^" + _MC_PAD + _MC_LIST + re.escape(str(opt))
               + _MC_PAD + r":" + _MC_PAD + r"(" + _NUM + r")\s*%?")
        hits = re.findall(pat, text, re.I | re.M)
        if not hits:
            return None
        probs[opt] = max(_to_float(hits[-1]), 0.0)
    total = sum(probs.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in probs.items()}


def parse_percentiles(text):
    found = {}
    for p, v in re.findall(r"Percentile\s*(\d+)\s*:\s*\$?(" + _NUM + r")", text, re.I):
        if 0 < int(p) < 100:
            found[int(p)] = _to_float(v)
    return found if len(found) >= 3 else None


# -------------------------------------------------------------- aggregation

def aggregate_binary(probs):
    probs = [min(max(p, 1e-4), 1 - 1e-4) for p in probs]
    mean_logit = sum(math.log(p / (1 - p)) for p in probs) / len(probs)
    p = 1 / (1 + math.exp(-mean_logit))
    return round(min(max(p, BINARY_FLOOR), BINARY_CEIL), 4)


def aggregate_multiple_choice(dists, options):
    mean = {o: sum(d[o] for d in dists) / len(dists) for o in options}
    floored = {o: max(v, MC_FLOOR) for o, v in mean.items()}
    total = sum(floored.values())
    out = {o: round(v / total, 6) for o, v in floored.items()}
    drift = round(1.0 - sum(out.values()), 6)
    top = max(out, key=out.get)
    out[top] = round(out[top] + drift, 6)
    return out


def aggregate_cdfs(cdfs):
    return [sum(c[i] for c in cdfs) / len(cdfs) for i in range(len(cdfs[0]))]


# --------------------------------------------------------------------- CDF

def bound_flags(q):
    s = q.get("scaling") or {}
    lo = q.get("open_lower_bound", s.get("open_lower_bound", False))
    hi = q.get("open_upper_bound", s.get("open_upper_bound", False))
    return bool(lo), bool(hi)


def cdf_size(q):
    if q.get("type") == "discrete":
        return int((q.get("scaling") or {})["inbound_outcome_count"]) + 1
    return 201


def location_to_value(x, range_min, range_max, zero_point):
    """Metaculus maps [0, 1] to the question range, log-scaled when zero_point is set."""
    if zero_point is not None:
        ratio = (range_max - zero_point) / (range_min - zero_point)
        return range_min + (range_max - range_min) * (ratio ** x - 1) / (ratio - 1)
    return range_min + (range_max - range_min) * x


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[hi]
    return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo])


def _cap_steps(steps, cap):
    """Limit each step to cap, moving the excess to steps with headroom."""
    steps = list(steps)
    for _ in range(100):
        excess = sum(max(d - cap, 0.0) for d in steps)
        if excess <= 1e-12:
            break
        steps = [min(d, cap) for d in steps]
        room = [cap - d for d in steps]
        total_room = sum(room)
        if total_room <= 0:
            break
        steps = [d + excess * r / total_room for d, r in zip(steps, room)]
    return steps


def cdf_from_percentiles(percentiles, q):
    """Turn {percentile: value} into a valid Metaculus CDF for q."""
    s = q.get("scaling") or {}
    rmin, rmax, zp = float(s["range_min"]), float(s["range_max"]), s.get("zero_point")
    zp = None if zp is None else float(zp)
    lo_open, hi_open = bound_flags(q)
    n = cdf_size(q)

    pairs = sorted((p / 100.0, v) for p, v in percentiles.items() if 0 < p < 100)
    probs = [p for p, _ in pairs]
    values = sorted(v for _, v in pairs)  # monotone rearrangement
    nudge = (rmax - rmin) * 1e-6
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] + nudge

    use_log = zp is not None and values[0] > zp and rmin > zp

    def t(v):
        return math.log(v - zp) if use_log else v

    xs, ys = [t(v) for v in values], probs[:]
    if t(rmin) < xs[0]:
        xs.insert(0, t(rmin))
        ys.insert(0, ys[0] * 0.5 if lo_open else 0.0)
    if t(rmax) > xs[-1]:
        xs.append(t(rmax))
        ys.append(ys[-1] + (1 - ys[-1]) * 0.5 if hi_open else 1.0)

    locs = [i / (n - 1) for i in range(n)]
    raw = [_interp(t(location_to_value(x, rmin, rmax, zp)), xs, ys) for x in locs]
    for i in range(1, n):
        raw[i] = max(raw[i], raw[i - 1])

    lo_mass = raw[0] if lo_open else 0.0
    hi_mass = 1 - raw[-1] if hi_open else 0.0
    span = raw[-1] - raw[0]
    inner = 1 - lo_mass - hi_mass
    if span <= 1e-12:
        steps = [inner / (n - 1)] * (n - 1)
    else:
        steps = [(raw[i + 1] - raw[i]) * inner / span for i in range(n - 1)]

    w, eps = CDF_UNIFORM_WEIGHT, CDF_OPEN_BOUND_MASS
    scale = 1 - w - eps * (int(lo_open) + int(hi_open))
    offset = eps if lo_open else 0.0
    max_step = 0.19 * 200 / (n - 1)
    steps = _cap_steps(steps, (max_step - w / (n - 1)) / scale)

    g = [lo_mass]
    for d in steps:
        g.append(g[-1] + d)
    cdf = [scale * g[i] + w * locs[i] + offset for i in range(n)]
    if not lo_open:
        cdf[0] = 0.0
    if not hi_open:
        cdf[-1] = 1.0
    return [round(c, 10) for c in cdf]


def cdf_problems(cdf, q):
    """Empty list when cdf meets the Metaculus constraints for q."""
    problems = []
    n = cdf_size(q)
    lo_open, hi_open = bound_flags(q)
    if len(cdf) != n:
        problems.append(f"length {len(cdf)} != {n}")
        return problems
    if lo_open and cdf[0] < 0.001:
        problems.append("open lower bound needs cdf[0] >= 0.001")
    if not lo_open and cdf[0] != 0.0:
        problems.append("closed lower bound needs cdf[0] == 0")
    if hi_open and cdf[-1] > 0.999:
        problems.append("open upper bound needs cdf[-1] <= 0.999")
    if not hi_open and cdf[-1] != 1.0:
        problems.append("closed upper bound needs cdf[-1] == 1")
    steps = [cdf[i + 1] - cdf[i] for i in range(n - 1)]
    if min(steps) < CDF_MIN_STEP - 1e-12:
        problems.append(f"step below {CDF_MIN_STEP}: {min(steps)}")
    if max(steps) > 0.2 * 200 / (n - 1) + 1e-9:
        problems.append(f"step above cap: {max(steps)}")
    return problems


# --------------------------------------------------------------- forecaster

def forecast_question(q, models, research_model, cfg, llm=llm_chat, markets_fn=manifold_markets):
    """Return (value, comment, record) for one question, or raise ValueError."""
    qtype = q.get("type")
    today = cfg.get("today") or dt.date.today().isoformat()
    base, key, runs = cfg["base_url"], cfg["api_key"], cfg["runs_per_model"]
    try:
        research = llm(research_model, research_prompt(q, today), base, key, 0.2, True)
    except Exception as exc:  # research is helpful, not required
        research = ""
        print(f"  research failed: {exc}", file=sys.stderr)
    markets = markets_fn(q.get("title", "")) if qtype == "binary" else []
    prompt = forecast_prompt(q, today, research, markets)

    parsed, rationales = [], []
    for model in models:
        for _ in range(runs):
            try:
                text = llm(model, prompt, base, key, 0.7, False)
            except Exception as exc:
                print(f"  {model} failed: {exc}", file=sys.stderr)
                continue
            if qtype == "binary":
                value = parse_binary(text)
            elif qtype == "multiple_choice":
                value = parse_multiple_choice(text, q.get("options") or [])
            else:
                pct = parse_percentiles(text)
                value = cdf_from_percentiles(pct, q) if pct else None
            if value is not None:
                parsed.append((model, value))
                rationales.append((model, text))
    if not parsed:
        raise ValueError("no usable forecasts")

    values = [v for _, v in parsed]
    if qtype == "binary":
        final = aggregate_binary(values)
        shown = f"{final:.1%}"
        each = ", ".join(f"{m}: {v:.0%}" for m, v in parsed)
    elif qtype == "multiple_choice":
        final = aggregate_multiple_choice(values, q.get("options") or [])
        shown = ", ".join(f"{k}: {v:.0%}" for k, v in final.items())
        each = f"{len(parsed)} runs"
    else:
        final = aggregate_cdfs(values)
        problems = cdf_problems(final, q)
        if problems:
            raise ValueError("invalid CDF: " + "; ".join(problems))
        shown = "median ~ " + median_text(final, q)
        each = f"{len(parsed)} runs"

    comment = strip_links(
        f"MV2 forecaster {__version__}, open source, by MV2 of Munim, Inc. "
        f"This forecast was made entirely by AI, with no human in the loop.\n\n"
        f"Aggregate: {shown}\nRuns: {each}\n\n"
        f"Research summary:\n{strip_links(research)[:2500] or 'none'}\n\n"
        f"Reasoning from {rationales[0][0]}:\n{strip_links(rationales[0][1])[:4000]}")
    record = {"question_id": q.get("id"), "title": q.get("title"), "type": qtype, "today": today,
              "models": models, "runs": [[m, v] for m, v in parsed], "final": final,
              "markets": markets, "version": __version__}
    return final, comment, record


def median_text(cdf, q):
    s = q.get("scaling") or {}
    n = len(cdf)
    for i, c in enumerate(cdf):
        if c >= 0.5:
            v = location_to_value(i / (n - 1), float(s["range_min"]), float(s["range_max"]),
                                  None if s.get("zero_point") is None else float(s["zero_point"]))
            return f"{v:.4g}"
    return "above range"


# ---------------------------------------------------------------------- CLI

def resolve_models(cfg):
    env = [m.strip() for m in (os.environ.get("FORECAST_MODELS") or "").split(",") if m.strip()]
    if env:
        return env
    try:
        listing = http_json("GET", cfg["base_url"].rstrip("/") + "/models",
                            headers={"Authorization": f"Bearer {cfg['api_key']}"}, timeout=60)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        print(f"could not list models: {exc}", file=sys.stderr)
        return []
    return pick_models(listing)


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tournament", action="append", help="tournament id or slug (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="forecast but do not post")
    ap.add_argument("--max-questions", type=int, default=0, help="stop after N forecasts (0 = all)")
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "runs"))
    args = ap.parse_args(argv)

    token = os.environ.get("METACULUS_TOKEN", "").strip()
    api_key = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not token or not api_key:
        print("METACULUS_TOKEN and LLM_API_KEY are not both set; nothing to do.")
        return 0
    cfg = {"base_url": os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
           "api_key": api_key, "runs_per_model": int(os.environ.get("RUNS_PER_MODEL", "2")),
           "refresh_before_close_h": _env_float("REFRESH_BEFORE_CLOSE_H", REFRESH_BEFORE_CLOSE_H),
           "min_forecast_age_h": _env_float("MIN_FORECAST_AGE_H", MIN_FORECAST_AGE_H)}
    tournaments = args.tournament or [
        t.strip() for t in (os.environ.get("TOURNAMENTS") or "").split(",") if t.strip()
    ] or DEFAULT_TOURNAMENTS
    models = resolve_models(cfg)
    if not models:
        print("no models available; set FORECAST_MODELS", file=sys.stderr)
        return 1
    research_model = os.environ.get("RESEARCH_MODEL") or models[0]
    print(f"mv2-forecaster {__version__}: models={models} tournaments={tournaments}")

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "forecasts.jsonl")
    done = failures = 0
    for tournament in tournaments:
        try:
            posts = list_open_posts(tournament, token)
        except (urllib.error.URLError, ValueError, OSError) as exc:
            print(f"{tournament}: listing failed: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"{tournament}: {len(posts)} open questions")
        for post in posts:
            if args.max_questions and done >= args.max_questions:
                break
            try:
                detail = get_post(post["id"], token)
                q = detail["question"]
                if q.get("type") not in FORECAST_TYPES or not should_forecast(
                        q, before_close_h=cfg["refresh_before_close_h"],
                        min_age_h=cfg["min_forecast_age_h"]):
                    continue
                refresh = already_forecast(q)
                print(f"- {q['id']} [{q['type']}] "
                      f"{'(refresh) ' if refresh else ''}{q.get('title', '')[:90]}")
                value, comment, record = forecast_question(q, models, research_model, cfg)
                record["post_id"] = post["id"]
                record["tournament"] = tournament
                record["refresh"] = refresh
                if not args.dry_run:
                    post_forecast(q["id"], q["type"], value, token)
                    for attempt in range(2):
                        try:
                            post_private_comment(post["id"], comment, token)
                            break
                        except (urllib.error.URLError, OSError) as exc:
                            print(f"  comment failed ({attempt + 1}): {exc}", file=sys.stderr)
                            time.sleep(5)
                record["posted"] = not args.dry_run
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
                done += 1
            except Exception as exc:
                failures += 1
                print(f"  skipped post {post.get('id')}: {exc}", file=sys.stderr)
    print(f"done: {done} forecasts, {failures} failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""News sentiment, scored two ways.

1. VADER: a rule-based word list (plus finance words). Free, offline, but it only
   counts positive/negative words and can't tell if a story is about the stock.
2. Claude (LLM): reads each headline like an analyst, gives a score from -1 to +1
   and says whether the story is really about this company. Used when an
   Anthropic API key is available; otherwise the pipeline falls back to VADER.

Every run also appends headlines to data/news_archive.jsonl, so a real history
builds up over time and can later be tested as a model feature.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import RAW

ARCHIVE = RAW.parent / "news_archive.jsonl"
LLM_CACHE = RAW.parent / "llm_scores.json"   # pay once per headline
MODEL = "claude-opus-5-5"

FINANCE_WORDS = {
    "beat": 2.0, "beats": 2.0, "surge": 2.5, "surges": 2.5, "soar": 2.5, "soars": 2.5,
    "rally": 2.0, "rallies": 2.0, "upgrade": 2.0, "upgraded": 2.0, "outperform": 1.8,
    "record": 1.2, "bullish": 2.2, "jump": 1.6, "jumps": 1.6, "gain": 1.3, "gains": 1.3,
    "miss": -2.0, "misses": -2.0, "plunge": -2.8, "plunges": -2.8, "tumble": -2.4,
    "tumbles": -2.4, "slump": -2.2, "downgrade": -2.0, "downgraded": -2.0,
    "bearish": -2.2, "lawsuit": -1.6, "probe": -1.5, "recall": -1.6, "layoffs": -1.5,
    "sell-off": -2.2, "selloff": -2.2, "drop": -1.5, "drops": -1.5, "falls": -1.4,
    "weakness": -1.4, "dragging": -1.2, "tariff": -1.0, "tariffs": -1.0,
}
_an = SentimentIntensityAnalyzer()
_an.lexicon.update(FINANCE_WORDS)


def vader(text):
    return _an.polarity_scores(text)["compound"]


# ------------------------------------------------------------------ LLM scorer
SYSTEM = (
    "You are an equity research assistant. For each numbered news item about a "
    "company, judge how the news would likely affect that company's stock price "
    "over the next few days. score: -1 (clearly negative) to +1 (clearly positive), "
    "0 if neutral or unclear. relevant: true only if the item is mainly about this "
    "company or directly affects its business; false if it mentions the company in "
    "passing or is about another firm or the market in general. reason: at most 12 words."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "number"},
                    "relevant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "score", "relevant", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def _key(ticker, title):
    return hashlib.sha1(f"{ticker}|{title}".encode()).hexdigest()[:16]


def _llm_client():
    """Return a client only when credentials exist."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None


def llm_score_batch(client, ticker, items):
    """Score a list of {title, summary} for one ticker in a single request.
    Returns {index: {...}} or {} on failure."""
    import anthropic

    lines = [f"[{i}] {x['title']} — {(x.get('summary') or '')[:300]}" for i, x in enumerate(items)]
    try:
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            # if a request is ever declined, the API retries it on a fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": f"Company ticker: {ticker}\n\n" + "\n".join(lines)}],
        )
    except anthropic.RateLimitError:
        print(f"  [llm] rate limited on {ticker}; using VADER for now")
        return {}
    except anthropic.APIStatusError as e:
        print(f"  [llm] API error {e.status_code} on {ticker}: {e.message}")
        return {}
    except anthropic.APIConnectionError:
        print(f"  [llm] network error on {ticker}")
        return {}
    if resp.stop_reason != "end_turn":
        print(f"  [llm] {ticker}: stopped with {resp.stop_reason}")
        return {}
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    out = {}
    for r in json.loads(text).get("items", []):
        if 0 <= r["index"] < len(items):
            out[r["index"]] = {"score": max(-1.0, min(1.0, float(r["score"]))),
                               "relevant": bool(r["relevant"]), "reason": r["reason"]}
    return out


def add_llm_scores(news):
    """Attach `llm` to each headline, using the on-disk cache when possible.
    Returns True if any LLM scores exist."""
    cache = json.loads(LLM_CACHE.read_text()) if LLM_CACHE.exists() else {}
    client = _llm_client()
    missing = 0
    for t, items in news.items():
        todo = [i for i, x in enumerate(items) if _key(t, x["title"]) not in cache]
        missing += len(todo)
        if todo and client:
            got = llm_score_batch(client, t, [items[i] for i in todo])
            for j, i in enumerate(todo):
                if j in got:
                    cache[_key(t, items[i]["title"])] = got[j]
        for x in items:
            x["llm"] = cache.get(_key(t, x["title"]))
    LLM_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    if missing and not client:
        print("  [llm] no Anthropic credentials found; news scored with VADER only")
    return any(x["llm"] for items in news.values() for x in items)


# ------------------------------------------------------------------ archive
def archive(news):
    """Append new headlines (deduplicated) to the JSONL archive."""
    seen = set()
    if ARCHIVE.exists():
        for line in ARCHIVE.read_text().splitlines():
            r = json.loads(line)
            seen.add(_key(r["ticker"], r["title"]))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    added = 0
    with ARCHIVE.open("a") as f:
        for t, items in news.items():
            for x in items:
                k = _key(t, x["title"])
                if k in seen:
                    continue
                seen.add(k)
                rec = {"ticker": t, "date": x["date"], "fetched": now, "title": x["title"],
                       "source": x["source"], "vader": round(vader(x["title"] + ". " + (x.get("summary") or "")[:300]), 3),
                       "llm": x.get("llm")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                added += 1
    return added


def archive_daily(tickers):
    """Daily average sentiment per ticker from the archive."""
    if not ARCHIVE.exists():
        return {t: [] for t in tickers}
    rows = [json.loads(l) for l in ARCHIVE.read_text().splitlines()]
    # headlines archived before an API key existed pick up later LLM scores from the cache
    cache = json.loads(LLM_CACHE.read_text()) if LLM_CACHE.exists() else {}
    for r in rows:
        r["llm"] = r.get("llm") or cache.get(_key(r["ticker"], r["title"]))
    out = {}
    for t in tickers:
        by_day = {}
        for r in rows:
            if r["ticker"] != t or not r["date"]:
                continue
            d = by_day.setdefault(r["date"][:10], {"v": [], "l": []})
            d["v"].append(r["vader"])
            if r.get("llm") and r["llm"]["relevant"]:
                d["l"].append(r["llm"]["score"])
        out[t] = [{"day": k, "n": len(v["v"]), "vader": round(sum(v["v"]) / len(v["v"]), 3),
                   "llm": round(sum(v["l"]) / len(v["l"]), 3) if v["l"] else None}
                  for k, v in sorted(by_day.items())]
    return out


# ------------------------------------------------------------------ main entry
def score_news(news):
    has_llm = add_llm_scores(news)
    added = archive(news)
    print(f"  news: {added} new headlines archived; LLM scores {'on' if has_llm else 'off'}")
    daily = archive_daily(list(news))
    out = {"_meta": {"llm": has_llm, "model": MODEL if has_llm else None,
                     "archive_rows": sum(1 for _ in ARCHIVE.open()) if ARCHIVE.exists() else 0}}
    for t, items in news.items():
        rows = []
        for x in items:
            l = x.get("llm")
            rows.append({"title": x["title"], "date": x["date"][:16].replace("T", " "),
                         "source": x["source"],
                         "score": round(vader(x["title"] + ". " + (x.get("summary") or "")[:300]), 3),
                         "llm": round(l["score"], 2) if l else None,
                         "relevant": l["relevant"] if l else None,
                         "reason": l["reason"] if l else None})
        rows.sort(key=lambda r: r["date"], reverse=True)
        avg = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
        rel = [r["llm"] for r in rows if r["llm"] is not None and r["relevant"]]
        out[t] = {"avg": round(avg, 3), "n": len(rows),
                  "llm_avg": round(sum(rel) / len(rel), 3) if rel else None,
                  "llm_relevant": len(rel), "items": rows[:25], "daily": daily[t]}
    return out

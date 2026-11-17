"""Claude as forecaster: can a large language model beat the ML model and the
"always guess up" baseline at predicting a stock's next 5 trading days?

Leak control:
  * Only dates AFTER Claude's training-data cutoff (Jun 2026, per Anthropic's
    model overview page) are tested, so Claude cannot remember what happened.
  * Every 5th trading day, so 5-day targets don't overlap (independent samples).

Two arms:
  anon  : numbers only, company and date hidden. Tests pure reasoning on data.
  named : same numbers plus company name and date. Tests whether Claude's
          background knowledge of the company helps.

Usage:
    python llm_forecast.py --dry-run        # show sample prompt + cost estimate, no API calls
    python llm_forecast.py                  # run both arms (needs ANTHROPIC_API_KEY)
    python llm_forecast.py --arms anon      # run one arm
Results are cached in data/llm_forecasts.jsonl, so an interrupted run resumes.
"""
import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import RAW, OUT, HORIZON
from features import FEATURES, build_panel
from fetch import load

MODEL = "claude-opus-5-5"
CUTOFF_NOTE = "Jun 2026 (Anthropic models overview, training data cutoff)"
TEST_START = "2026-07-01"
STEP = HORIZON                       # sample every 5th trading day
CACHE = RAW.parent / "llm_forecasts.jsonl"
RESULTS = RAW.parent / "llm_forecast.json"
ML_PRED = RAW.parent / "ml_predictions.csv"
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
         "GOOGL": "Alphabet (Google)", "TSLA": "Tesla", "JPM": "JPMorgan Chase", "XOM": "ExxonMobil"}

SYSTEM = (
    "You are a quantitative equity analyst. You will see recent data for one large-cap "
    "US stock as of a market close. Estimate the probability that the stock's total "
    "return over the NEXT 5 trading days is positive. Over the past decade about 56% of "
    "5-day windows for stocks like this were positive. Be calibrated: short-term returns "
    "are very hard to predict, so stay close to the base rate unless the data gives a "
    "real reason to move. Give a probability between 0 and 1 and a rationale of at most 40 words."
)
# Same task with no base rate and no advice to stay near it: does Claude get bolder,
# and does boldness help?
SYSTEM_NOHINT = (
    "You are a quantitative equity analyst. You will see recent data for one large-cap "
    "US stock as of a market close. Estimate the probability that the stock's total "
    "return over the NEXT 5 trading days is positive. Give a probability between 0 and 1 "
    "and a rationale of at most 40 words."
)
SYSTEMS = {"anon": SYSTEM, "named": SYSTEM, "nohint": SYSTEM_NOHINT}
PRICE_IN, PRICE_OUT = 4.0, 20.0   # $ per million tokens, claude-opus-5-5
ARMS = ["anon", "named", "nohint"]

SCHEMA = {
    "type": "object",
    "properties": {"prob_up": {"type": "number"}, "rationale": {"type": "string"}},
    "required": ["prob_up", "rationale"],
    "additionalProperties": False,
}


def pct(x):
    return f"{x * 100:+.2f}%"


def make_prompt(row, recent, arm):
    """Turn one (stock, date) row into a plain-text briefing."""
    head = (f"Company: {NAMES[row.ticker]} ({row.ticker})\nAs of market close on: {row.date:%Y-%m-%d}\n"
            if arm == "named" else "Company: an anonymized large-cap US stock\nDate: hidden\n")  # anon + nohint
    eps = "n/a" if pd.isna(row.eps_surprise) else f"{row.eps_surprise:+.1f}%"
    return head + f"""
Last 20 daily returns, oldest to newest: {", ".join(f"{r * 100:+.1f}" for r in recent)} (%)

Stock
- 5-day return {pct(row.ret_5)}, 20-day return {pct(row.ret_20)}
- 20-day volatility (annualized) {row.vol_20 * 100:.0f}%
- RSI(14) {row.rsi_14:.1f}
- Price vs 50-day average {pct(row.dist_sma50)}, vs 200-day average {pct(row.dist_sma200)}
- Volume vs 20-day norm: {row.volume_z:+.2f} standard deviations
- 20-day return minus S&P 500: {pct(row.rel_str_20)}

Market and macro
- S&P 500 5-day return {pct(row.spx_ret_5)}
- VIX {row.vix:.1f} (5-day change {row.vix_chg_5:+.1f})
- 10-year Treasury yield, 20-day change {row.tnx_chg_20:+.2f} percentage points
- US dollar index 20-day {pct(row.dxy_ret_20)}, oil 20-day {pct(row.oil_ret_20)}, gold 20-day {pct(row.gold_ret_20)}

Fundamentals
- Last quarterly EPS surprise {eps}, reported {row.days_since_er:.0f} days ago

What is the probability that this stock's return over the next 5 trading days is positive?"""


def sample(panel):
    """Test rows: after the cutoff, labeled, every STEP-th trading day."""
    p = panel.dropna(subset=FEATURES + ["fwd_ret_h"])
    days = sorted(p.loc[p["date"] >= TEST_START, "date"].unique())[::STEP]
    return p[p["date"].isin(days)].sort_values(["date", "ticker"])


def recent_returns(panel, ticker, date, n=20):
    s = panel[(panel["ticker"] == ticker) & (panel["date"] <= date)].sort_values("date")
    return s["ret_1"].tail(n).tolist()


def ask(client, prompt, system=SYSTEM):
    """One forecast. Returns (prob, rationale, model_used, usage dict) or raises."""
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],   # retry on a fallback model if declined
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={resp.stop_reason}")
    data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    return float(np.clip(data["prob_up"], 0.0, 1.0)), data["rationale"], resp.model, usage


def load_cache():
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            out[(r["arm"], r["ticker"], r["date"])] = r
    return out


def run(panel, rows, arms, client, workers=4):
    import anthropic

    cache = load_cache()
    jobs = []
    for arm in arms:
        for r in rows.itertuples():
            k = (arm, r.ticker, f"{r.date:%Y-%m-%d}")
            if k not in cache:
                jobs.append((k, make_prompt(r, recent_returns(panel, r.ticker, r.date), arm), SYSTEMS[arm]))
    print(f"{len(jobs)} forecasts to request ({sum(len(rows) for _ in arms) - len(jobs)} cached)")
    done = failed = 0
    with CACHE.open("a") as f, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(ask, client, prompt, system): k for k, prompt, system in jobs}
        for fut in as_completed(futs):
            arm, t, d = futs[fut]
            try:
                prob, why, used, usage = fut.result()
            except (anthropic.APIError, RuntimeError, ValueError, KeyError) as e:
                failed += 1
                print(f"  failed {arm} {t} {d}: {e}")
                continue
            rec = {"arm": arm, "ticker": t, "date": d, "prob": round(prob, 4), "rationale": why, "model": used,
                   "tokens_in": usage["in"], "tokens_out": usage["out"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)} done")
    print(f"finished: {done} new, {failed} failed (re-run to retry failures)")


# ------------------------------------------------------------------ evaluation
def metrics(y, p, rng, n_boot=2000):
    """Accuracy, AUC, Brier score + 95% bootstrap intervals.
    Brier = mean squared error of the probability; lower is better, 0.25 = always saying 50%."""
    y, p = np.asarray(y, float), np.asarray(p, float)

    def one(idx):
        yy, pp = y[idx], p[idx]
        auc = roc_auc_score(yy, pp) if 0 < yy.mean() < 1 and np.ptp(pp) > 0 else np.nan
        return ((pp > 0.5) == yy).mean(), auc, ((pp - yy) ** 2).mean()

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN AUC for constant predictors
    point = one(np.arange(len(y)))
    boots = np.array([one(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
    r = lambda x: None if not np.isfinite(x) else round(float(x), 4)
    return {"acc": r(point[0]), "acc_ci": [r(lo[0]), r(hi[0])],
            "auc": r(point[1]), "auc_ci": [r(lo[1]), r(hi[1])],
            "brier": r(point[2]), "brier_ci": [r(lo[2]), r(hi[2])], "n": int(len(y))}


def cost_summary(cache):
    """Actual spend from recorded token counts (runs before tracking have none)."""
    out = {}
    for arm in ARMS:
        recs = [r for k, r in cache.items() if k[0] == arm and "tokens_in" in r]
        if recs:
            tin, tout = sum(r["tokens_in"] for r in recs), sum(r["tokens_out"] for r in recs)
            out[arm] = {"calls": len(recs), "tokens_in": tin, "tokens_out": tout,
                        "usd": round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 2)}
    return out


def evaluate(rows):
    cache = load_cache()
    if not cache:
        return None
    df = rows[["date", "ticker", "fwd_ret_h"]].copy()
    df["date_s"] = df["date"].dt.strftime("%Y-%m-%d")
    df["y"] = (df["fwd_ret_h"] > 0).astype(int)
    ml = pd.read_csv(ML_PRED, parse_dates=["date"])
    df = df.merge(ml[["date", "ticker", "p_gbm", "p_logit", "base_rate"]], on=["date", "ticker"], how="left")
    for arm in ARMS:
        df[f"p_{arm}"] = [cache.get((arm, t, d), {}).get("prob") for t, d in zip(df["ticker"], df["date_s"])]
    rng = np.random.default_rng(0)
    contenders = {"claude_anon": "p_anon", "claude_named": "p_named", "claude_nohint": "p_nohint", "ml_boosting": "p_gbm",
                  "ml_logistic": "p_logit", "always_up": "base_rate"}
    table = {}
    for name, col in contenders.items():
        d = df.dropna(subset=[col])
        if len(d) >= 20:
            table[name] = metrics(d["y"], d[col], rng)
    examples = []
    for arm in ARMS:
        for k, r in list(cache.items()):
            if k[0] == arm and len([e for e in examples if e["arm"] == arm]) < 4:
                y = df.loc[(df.ticker == k[1]) & (df.date_s == k[2]), "fwd_ret_h"]
                examples.append({**r, "actual": None if y.empty else round(float(y.iloc[0]), 4)})
    return {
        "model": MODEL, "cutoff": CUTOFF_NOTE, "test_start": TEST_START,
        "test_end": df["date_s"].max(), "n_dates": int(df["date_s"].nunique()),
        "up_share": round(float(df["y"].mean()), 4), "table": table,
        "corr_claude_ml": None if df[["p_anon", "p_gbm"]].dropna().shape[0] < 10
        else round(float(df[["p_anon", "p_gbm"]].dropna().corr().iloc[0, 1]), 3),
        "mean_prob": {a: None if df[f"p_{a}"].isna().all() else round(float(df[f"p_{a}"].mean()), 4)
                      for a in ARMS},
        "examples": examples,
        # how boldly Claude used its probabilities: range and share of "up" calls
        "spread": {a: None if df[f"p_{a}"].isna().all() else {
            "min": round(float(df[f"p_{a}"].min()), 3), "max": round(float(df[f"p_{a}"].max()), 3),
            "share_up": round(float((df[f"p_{a}"] > 0.5).mean()), 3)} for a in ARMS},
        "cost": cost_summary(cache),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["anon", "named"], choices=ARMS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    close, volume, macro, earnings, _ = load()
    panel = build_panel(close, volume, macro, earnings)
    rows = sample(panel)
    n = len(rows) * len(a.arms)
    print(f"test window {TEST_START} → {rows['date'].max():%Y-%m-%d}: "
          f"{rows['date'].nunique()} dates × {rows['ticker'].nunique()} stocks = {len(rows)} rows, {n} forecasts")
    # rough cost: ~900 input tokens + ~1,500 output/thinking tokens per call at $4/$20 per M
    print(f"estimated cost ≈ ${n * (900 * 4 + 1500 * 20) / 1e6:.2f} (actual depends on thinking length)")

    if a.dry_run:
        r = rows.iloc[0]
        for arm in a.arms:
            print(f"\n----- sample prompt ({arm}) -----\n" + make_prompt(r, recent_returns(panel, r.ticker, r.date), arm))
        return
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("No Anthropic credentials. Set ANTHROPIC_API_KEY first.")
    if not ML_PRED.exists():
        sys.exit("Run build.py first so ML predictions exist.")

    import anthropic
    run(panel, rows, a.arms, anthropic.Anthropic(), a.workers)
    res = evaluate(rows)
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    # patch the dashboard data so no full rebuild is needed
    dj = OUT / "data.json"
    if dj.exists():
        d = json.loads(dj.read_text())
        d["llm_forecast"] = res
        dj.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
    for k, v in res["table"].items():
        print(f"{k:14s} acc {v['acc']}  AUC {v['auc']}  Brier {v['brier']}  (n={v['n']})")


if __name__ == "__main__":
    main()

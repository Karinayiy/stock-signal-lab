"""Portfolio analysis and live quotes from Yahoo Finance (free, no key).

Used by app.py, so holdings stay on this computer. Everything here describes
risk; nothing here predicts direction or says what to buy or sell.

    holdings = [{"ticker": "AAPL", "shares": 10, "cost": 150}, {"ticker": "CASH", "shares": 2000}]
    analyse(holdings)
"""
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-=^]{0,14}$")
CASH = "CASH"
BENCH = "SPY"
MAX_HOLDINGS = 50
QUOTE_TTL = 30          # seconds a live quote is reused
HIST_TTL = 3600         # seconds price history is reused

_lock = threading.Lock()
_quotes, _hist, _fx = {}, {}, {}


def clean_ticker(t):
    t = str(t or "").strip().upper().replace("/", "-")
    t = re.sub(r"\.([A-C])$", r"-\1", t)      # US share classes: BRK.B -> BRK-B (Yahoo style)
    return t if TICKER_RE.match(t) else None


def _quote(t):
    try:
        fi = yf.Ticker(t).fast_info
        price, prev = fi.last_price, fi.previous_close
        if price is None or not math.isfinite(price):
            return t, None
        return t, {"price": float(price), "prev": float(prev) if prev else None,
                   "currency": fi.currency or "USD", "time": time.time()}
    except Exception:
        return t, None


def quotes(tickers):
    """Latest price and previous close for each ticker; None if Yahoo has nothing."""
    now, need = time.time(), []
    with _lock:
        for t in tickers:
            q = _quotes.get(t)
            if not q or now - q["time"] > QUOTE_TTL:
                need.append(t)
    if need:
        with ThreadPoolExecutor(8) as ex:
            got = dict(ex.map(_quote, need))
        with _lock:
            for t, q in got.items():
                if q:
                    _quotes[t] = q
                else:
                    _quotes.pop(t, None)
    with _lock:
        return {t: _quotes.get(t) for t in tickers}


def fx_to_usd(cur):
    if not cur or cur == "USD":
        return 1.0
    pence = cur in ("GBp", "GBX", "ZAc", "ILA")          # quoted in 1/100 of the currency
    base = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}.get(cur, cur)
    with _lock:
        r = _fx.get(base)
    if r is None or time.time() - r[1] > HIST_TTL:
        q = _quote(f"{base}USD=X")[1]
        r = (q["price"] if q else float("nan"), time.time())
        with _lock:
            _fx[base] = r
    return r[0] / (100 if pence else 1)


def history(tickers, period="1y"):
    """Daily adjusted closes, cached for an hour."""
    key = (tuple(sorted(tickers)), period)
    with _lock:
        h = _hist.get(key)
    if h and time.time() - h[1] < HIST_TTL:
        return h[0]
    d = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, group_by="column")
    close = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.ffill()
    with _lock:
        _hist[key] = (close, time.time())
    return close


def max_drawdown(series):
    s = series.dropna()
    return float((s / s.cummax() - 1).min()) if len(s) else None


def _r(x, k=4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def analyse(holdings, forecasts=None):
    """Value, today's change, and risk of a portfolio.

    forecasts: optional {ticker: forecast annual volatility} from the study,
    used for the eight tickers it covers.
    """
    forecasts = forecasts or {}
    rows, errors = {}, []
    for h in holdings[:MAX_HOLDINGS]:
        t = clean_ticker(h.get("ticker"))
        try:
            sh = float(h.get("shares"))
        except (TypeError, ValueError):
            sh = float("nan")
        if not t or not math.isfinite(sh):
            errors.append(f"Skipped {h.get('ticker')!r}: needs a ticker and a number of shares.")
            continue
        cost = h.get("cost")
        try:
            cost = float(cost) if cost not in (None, "") else None
        except (TypeError, ValueError):
            cost = None
        r = rows.setdefault(t, {"shares": 0.0, "cost_total": 0.0, "cost_known": True})
        r["shares"] += sh
        if cost is None:
            r["cost_known"] = False
        else:
            r["cost_total"] += cost * sh
    if len(holdings) > MAX_HOLDINGS:
        errors.append(f"Only the first {MAX_HOLDINGS} rows were used.")

    stocks = [t for t in rows if t != CASH]
    q = quotes(stocks + [BENCH])
    out, missing = [], []
    for t, r in rows.items():
        if t == CASH:
            out.append({"ticker": CASH, "shares": r["shares"], "price": 1.0, "prev": 1.0, "currency": "USD",
                        "value": r["shares"], "day_pnl": 0.0, "cost_total": r["shares"]})
            continue
        qt = q.get(t)
        if not qt:
            missing.append(t)
            continue
        fx = fx_to_usd(qt["currency"])
        if not math.isfinite(fx):
            missing.append(t)
            continue
        value = r["shares"] * qt["price"] * fx
        day = r["shares"] * (qt["price"] - qt["prev"]) * fx if qt["prev"] else None
        out.append({"ticker": t, "shares": r["shares"], "price": qt["price"], "prev": qt["prev"],
                    "currency": qt["currency"], "value": value, "day_pnl": day,
                    "cost_total": r["cost_total"] * fx if r["cost_known"] else None})
    if missing:
        errors.append("No price found for " + ", ".join(missing) + ". Check the symbol (Yahoo style, e.g. BRK-B, 0700.HK).")
    if not out:
        return {"errors": errors or ["No holdings."], "holdings": []}

    total = sum(o["value"] for o in out)
    gross = sum(abs(o["value"]) for o in out) or 1.0
    for o in out:
        o["weight"] = o["value"] / total if total else None
        o["pnl"] = o["value"] - o["cost_total"] if o.get("cost_total") is not None else None

    # ---- risk from one year of daily returns ----
    names = [o["ticker"] for o in out if o["ticker"] != CASH]
    risk = {}
    if names:
        close = history(sorted(set(names + [BENCH])))
        rets = close.pct_change(fill_method=None).iloc[1:]
        w = pd.Series({o["ticker"]: o["value"] / total for o in out if o["ticker"] != CASH})
        R = rets[w.index].fillna(0.0)
        cov = R.cov() * 252
        port_r = R @ w
        port_vol = float(np.sqrt(w @ cov @ w))
        mrc = cov @ w
        contrib = (w * mrc) / (port_vol ** 2) if port_vol > 0 else w * 0
        bench = rets[BENCH]
        beta_p = float(port_r.cov(bench) / bench.var())
        corr = R.corr()
        for o in out:
            t = o["ticker"]
            if t == CASH:
                o.update(vol_1y=0.0, vol_20=0.0, beta=0.0, risk_share=0.0)
                continue
            s = rets[t].dropna()
            o["vol_1y"] = _r(s.std() * math.sqrt(252))
            o["vol_20"] = _r(s.tail(20).std() * math.sqrt(252))
            o["beta"] = _r(s.cov(bench) / bench.var(), 2)
            o["ret_1y"] = _r(close[t].dropna().iloc[-1] / close[t].dropna().iloc[0] - 1)
            o["max_dd_1y"] = _r(max_drawdown(close[t]))
            o["risk_share"] = _r(float(contrib[t]))
            o["forecast_vol"] = forecasts.get(t)
        growth = (1 + port_r).cumprod()
        day_sd = port_vol / math.sqrt(252)
        pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(corr.index) for b in corr.index[i + 1:]]
        avg_vol = sum(abs(w[t]) * (rets[t].std() * math.sqrt(252)) for t in w.index)
        risk = {
            "vol": _r(port_vol), "beta": _r(beta_p, 2),
            "bad_day": _r(-1.645 * day_sd * total, 0),               # 1-in-20 day, normal approx.
            "market_down_10": _r(-0.10 * beta_p * total, 0),
            "max_dd_1y": _r(max_drawdown(growth)),
            "ret_1y": _r(float(growth.iloc[-1] - 1)),
            "diversification": _r(avg_vol / port_vol, 2) if port_vol else None,
            "bench_vol": _r(bench.std() * math.sqrt(252)),
            "corr": {"labels": list(corr.index), "values": [[_r(v, 2) for v in row] for row in corr.values]},
            "top_pairs": [{"a": a, "b": b, "corr": _r(c, 2)} for a, b, c in sorted(pairs, key=lambda x: -x[2])[:3]],
            "history": {"dates": [d.strftime("%Y-%m-%d") for d in growth.index], "port": [_r(v) for v in growth],
                        "bench": [_r(v) for v in (1 + bench.fillna(0)).cumprod()]},
        }

    # ---- plain observations (describe risk; no buy/sell calls) ----
    notes = []
    stock_rows = sorted([o for o in out if o["ticker"] != CASH], key=lambda o: -abs(o["value"]))
    if stock_rows and gross and abs(stock_rows[0]["value"]) / gross > 0.25:
        o = stock_rows[0]
        notes.append(f"{o['ticker']} is {abs(o['value']) / gross:.0%} of the portfolio, so one company's news moves the whole thing.")
    big = [o for o in out if (o.get("risk_share") or 0) > 0.4]
    for o in big:
        notes.append(f"{o['ticker']} is {o['weight']:.0%} of the money but {o['risk_share']:.0%} of the risk.")
    if risk.get("top_pairs") and risk["top_pairs"][0]["corr"] and risk["top_pairs"][0]["corr"] > 0.75:
        p = risk["top_pairs"][0]
        notes.append(f"{p['a']} and {p['b']} move together (correlation {p['corr']:.2f}), so they diversify each other little.")
    if risk.get("vol") and risk.get("bench_vol"):
        ratio = risk["vol"] / risk["bench_vol"]
        if ratio > 1.3:
            notes.append(f"The portfolio swings about {ratio:.1f}× as much as the S&P 500 (SPY).")
        elif ratio < 0.8:
            notes.append(f"The portfolio swings less than the S&P 500 (about {ratio:.1f}× as much).")
    for o in stock_rows:
        f, v = o.get("forecast_vol"), o.get("vol_1y")
        if f and v and f > 1.3 * v:
            notes.append(f"The study's forecast for {o['ticker']} next week ({f:.0%}) is well above its 1-year average ({v:.0%}).")

    return {
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S"), "currency": "USD", "errors": errors,
        "total": _r(total, 2), "day_pnl": _r(sum(o["day_pnl"] or 0 for o in out), 2),
        "pnl": _r(sum(o["pnl"] for o in out if o["pnl"] is not None), 2) if any(o["pnl"] is not None for o in out) else None,
        "holdings": sorted(out, key=lambda o: -abs(o["value"])), "risk": risk, "notes": notes,
    }


if __name__ == "__main__":
    import json
    demo = [{"ticker": "AAPL", "shares": 10, "cost": 150}, {"ticker": "NVDA", "shares": 40},
            {"ticker": "VOO", "shares": 5}, {"ticker": "CASH", "shares": 2000}]
    res = analyse(demo)
    res["risk"].pop("history", None)
    print(json.dumps(res, indent=1, default=str)[:3000])

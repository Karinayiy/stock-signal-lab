"""Download and cache raw data from Yahoo Finance (free, no key)."""
import json
import time
import pandas as pd
import yfinance as yf

from config import RAW, TICKERS, MACRO, START


def _download(symbols):
    d = yf.download(symbols, start=START, progress=False, auto_adjust=True, group_by="column")
    return d


def fetch_all(refresh=False):
    RAW.mkdir(parents=True, exist_ok=True)
    px_path, vol_path, mac_path = RAW / "close.csv", RAW / "volume.csv", RAW / "macro.csv"
    if refresh or not px_path.exists():
        d = _download(TICKERS)
        d["Close"][TICKERS].to_csv(px_path)
        d["Volume"][TICKERS].to_csv(vol_path)
        m = _download(list(MACRO))["Close"].rename(columns=MACRO)
        m.to_csv(mac_path)

    for t in TICKERS:
        e_path = RAW / f"earnings_{t}.csv"
        if refresh or not e_path.exists():
            e = yf.Ticker(t).get_earnings_dates(limit=100)
            e.to_csv(e_path)
        n_path = RAW / f"news_{t}.json"
        # news refreshes every run (>6h old) so the archive grows daily
        if refresh or not n_path.exists() or time.time() - n_path.stat().st_mtime > 6 * 3600:
            news = yf.Ticker(t).get_news(count=50)
            rows = [
                {
                    "title": x["content"].get("title", ""),
                    "summary": x["content"].get("summary", ""),
                    "date": x["content"].get("pubDate", ""),
                    "source": (x["content"].get("provider") or {}).get("displayName", ""),
                }
                for x in news
                if x.get("content")
            ]
            n_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def load():
    close = pd.read_csv(RAW / "close.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(RAW / "volume.csv", index_col=0, parse_dates=True)
    macro = pd.read_csv(RAW / "macro.csv", index_col=0, parse_dates=True)
    earnings = {}
    for t in TICKERS:
        e = pd.read_csv(RAW / f"earnings_{t}.csv", index_col=0)
        e.index = pd.to_datetime(e.index, utc=True).tz_convert("America/New_York").tz_localize(None)
        earnings[t] = e
    news = {t: json.loads((RAW / f"news_{t}.json").read_text()) for t in TICKERS}
    return close, volume, macro, earnings, news


if __name__ == "__main__":
    fetch_all(refresh=True)
    c, v, m, e, n = load()
    print(c.shape, m.shape, {k: len(x) for k, x in n.items()})

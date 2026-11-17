"""Market-wide volatility forecasts for every S&P 500 stock.

The in-depth study (direction models, news, Claude) covers 8 stocks. This adds
one thing for the whole index: next week's volatility forecast, which the
study showed is predictable. One pooled model is trained on all ~500 stocks with
price and macro features only (no per-stock earnings data), checked walk-forward,
and applied to each stock's latest day.

    python universe.py            # use cached prices
    python universe.py --refresh  # re-download the S&P 500 list and prices

Writes web/universe.json (for the website) and data/universe_model.joblib
(for the local app's any-ticker forecast).
"""
import io
import json
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from config import HORIZON, OUT, RAW, ROOT
from features import build_panel

SP500_CSV = ROOT / "data" / "sp500.csv"
U_CLOSE, U_VOLUME = RAW / "universe_close.csv.gz", RAW / "universe_volume.csv.gz"
MODEL_PATH = ROOT / "data" / "universe_model.joblib"
UNI_START = "2015-01-01"
FIRST_TEST_YEAR = 2019
UNI_FEATURES = ["ret_1", "ret_5", "ret_20", "vol_20", "rsi_14", "dist_sma50", "dist_sma200", "volume_z",
                "rel_str_20", "spx_ret_5", "vix", "vix_chg_5", "tnx_chg_20", "dxy_ret_20", "oil_ret_20",
                "gold_ret_20", "vol_5"]
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def r(x, k=4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def constituents(refresh=False):
    """Ticker, company name and sector for the current S&P 500 (cached in data/sp500.csv)."""
    if refresh or not SP500_CSV.exists():
        try:
            html = requests.get(WIKI, headers={"User-Agent": "stock-signal-lab/1.0 (research script)"}, timeout=30).text
            t = pd.read_html(io.StringIO(html))[0]
            df = pd.DataFrame({"ticker": t["Symbol"].str.replace(".", "-", regex=False).str.strip(),
                               "name": t["Security"].str.strip(), "sector": t["GICS Sector"].str.strip()})
            if len(df) > 450:
                df.sort_values("ticker").to_csv(SP500_CSV, index=False)
        except Exception as e:  # keep the last good list
            print("  could not refresh the S&P 500 list:", e)
    return pd.read_csv(SP500_CSV)


def prices(tickers, refresh=False):
    if refresh or not U_CLOSE.exists():
        closes, vols = [], []
        for i in range(0, len(tickers), 100):
            chunk = tickers[i:i + 100]
            d = yf.download(chunk, start=UNI_START, auto_adjust=True, progress=False, group_by="column", threads=True)
            closes.append(d["Close"])
            vols.append(d["Volume"])
            print(f"  prices {min(i + 100, len(tickers))}/{len(tickers)}")
        pd.concat(closes, axis=1).to_csv(U_CLOSE)
        pd.concat(vols, axis=1).to_csv(U_VOLUME)
    close = pd.read_csv(U_CLOSE, index_col=0, parse_dates=True)
    volume = pd.read_csv(U_VOLUME, index_col=0, parse_dates=True)
    keep = [t for t in close.columns if close[t].notna().sum() > 260]
    return close[keep], volume[keep]


def macro_frame():
    from fetch import load
    return load()[2]


def new_model():
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300,
                                         min_samples_leaf=500, random_state=0)


def fit(X, y):
    """Fit on log volatility, then rescale so forecasts match the average week,
    not the median one (Duan's smearing factor from the training residuals)."""
    m = new_model().fit(X, y)
    return m, float(np.mean(np.exp(y - m.predict(X))))


def weekly(df):
    """Every 5th trading day: 5-day targets then don't overlap, and training is 5× faster."""
    dates = np.sort(df["date"].unique())
    return df[df["date"].isin(dates[::HORIZON])]


def study(panel):
    data = panel.dropna(subset=UNI_FEATURES + ["fwd_vol_h"])
    data = weekly(data[data["fwd_vol_h"] > 0]).copy()
    data["y"] = np.log(data["fwd_vol_h"])
    oos, per_year = [], []
    for year in range(FIRST_TEST_YEAR, data["date"].dt.year.max() + 1):
        start = pd.Timestamp(f"{year}-01-01")
        tr = data[data["date"] < start - pd.tseries.offsets.BDay(HORIZON)]   # embargo: no label overlaps the test year
        te = data[data["date"].dt.year == year]
        if te.empty:
            continue
        m, k = fit(tr[UNI_FEATURES].values, tr["y"].values)
        res = te[["date", "ticker", "fwd_vol_h", "vol_20"]].copy()
        res["gbm"] = k * np.exp(m.predict(te[UNI_FEATURES].values))
        oos.append(res)
        y = np.log(res["fwd_vol_h"])
        per_year.append({"year": year, "n": int(len(res)), "r2_gbm": r(r2_score(y, np.log(res["gbm"]))),
                         "r2_naive": r(r2_score(y, np.log(res["vol_20"].clip(lower=1e-4))))})
        print(f"  {year}: R² {per_year[-1]['r2_gbm']} vs naive {per_year[-1]['r2_naive']}")
    p = pd.concat(oos)
    y = np.log(p["fwd_vol_h"])
    overall = {"gbm": r(r2_score(y, np.log(p["gbm"]))), "naive": r(r2_score(y, np.log(p["vol_20"].clip(lower=1e-4)))),
               "corr": r(np.corrcoef(y, np.log(p["gbm"]))[0, 1]),
               "calibration": r(p["fwd_vol_h"].mean() / p["gbm"].mean(), 3),   # 1.0 = right level on average
               "n": int(len(p)), "stocks": int(p["ticker"].nunique())}
    final = fit(data[UNI_FEATURES].values, data["y"].values)
    return final, p, overall, per_year


def main():
    refresh = "--refresh" in sys.argv
    t0 = time.time()
    names = constituents(refresh)
    close, volume = prices(names["ticker"].tolist(), refresh)
    macro = macro_frame()
    print(f"universe: {close.shape[1]} stocks, {close.index.min():%Y-%m-%d} to {close.index.max():%Y-%m-%d}")
    panel = build_panel(close, volume, macro)
    print("panel", panel.shape)
    (model, scale), oos, overall, per_year = study(panel)
    print("overall", overall)
    joblib.dump({"model": model, "scale": scale, "features": UNI_FEATURES,
                 "trained": datetime.now().isoformat(timespec="minutes")}, MODEL_PATH)

    info = names.set_index("ticker")
    last = panel.dropna(subset=UNI_FEATURES).sort_values("date").groupby("ticker").tail(1)
    last = last[last["date"] >= last["date"].max() - pd.Timedelta(days=7)]     # skip stale or delisted
    preds = scale * np.exp(model.predict(last[UNI_FEATURES].values))
    rows = []
    for (_, row), pred in zip(last.iterrows(), preds):
        t = row["ticker"]
        hist = oos.loc[oos["ticker"] == t, "gbm"].values
        c = close[t].dropna()
        rows.append({
            "t": t, "name": info.loc[t, "name"] if t in info.index else t,
            "sector": info.loc[t, "sector"] if t in info.index else "",
            "date": row["date"].strftime("%Y-%m-%d"), "close": r(c.iloc[-1], 2), "prev": r(c.iloc[-2], 2),
            "ret_20": r(row["ret_20"]), "vol_20": r(row["vol_20"]), "pred": r(float(pred)),
            "rank": r(float((hist <= pred).mean()), 3) if len(hist) else None,
        })
    out = {"meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "horizon": HORIZON,
                    "first_test_year": FIRST_TEST_YEAR, "features": UNI_FEATURES,
                    "overall": overall, "per_year": per_year,
                    "sectors": sorted({x["sector"] for x in rows if x["sector"]})},
           "rows": sorted(rows, key=lambda x: x["t"])}
    path = OUT / "universe.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({len(rows)} stocks, {path.stat().st_size / 1e3:.0f} KB) in {time.time() - t0:.0f}s")


# ---- any ticker, on demand (used by the local app) ----
_cache = {}


def forecast_one(ticker):
    """Volatility forecast for any Yahoo ticker, using the market-wide model."""
    hit = _cache.get(ticker)
    if hit and time.time() - hit[1] < 3600:
        return hit[0]
    bundle = joblib.load(MODEL_PATH)
    d = yf.download([ticker], period="3y", auto_adjust=True, progress=False, group_by="column")
    if d.empty:
        return None
    close, volume = d["Close"], d["Volume"]
    if isinstance(close, pd.Series):
        close, volume = close.to_frame(ticker), volume.to_frame(ticker)
    if close[ticker].notna().sum() < 260:
        return None
    panel = build_panel(close, volume, macro_frame())
    ok = panel.dropna(subset=bundle["features"]).sort_values("date")
    if ok.empty:
        return None
    preds = bundle["scale"] * np.exp(bundle["model"].predict(ok[bundle["features"]].values))
    last, c = ok.iloc[-1], close[ticker].dropna()
    past = preds[:-1][::HORIZON]
    res = {"t": ticker, "name": ticker, "sector": "", "date": last["date"].strftime("%Y-%m-%d"),
           "close": r(c.iloc[-1], 2), "prev": r(c.iloc[-2], 2), "ret_20": r(last["ret_20"]),
           "vol_20": r(last["vol_20"]), "pred": r(float(preds[-1])),
           "rank": r(float((past <= preds[-1]).mean()), 3) if len(past) else None,
           "note": "Not in the S&P 500 list; regime compares with this stock's own past 2 years (in-sample)."}
    _cache[ticker] = (res, time.time())
    return res


if __name__ == "__main__":
    main()

"""Feature engineering. Every feature on row t uses only information known
at the close of trading day t — no peeking into the future (no look-ahead bias)."""
import numpy as np
import pandas as pd

from config import HORIZON

# Human-readable names + group, used by the dashboard.
FEATURE_INFO = {
    # technical
    "ret_1":      ("1-day return", "technical"),
    "ret_5":      ("5-day return", "technical"),
    "ret_20":     ("20-day return", "technical"),
    "vol_20":     ("20-day volatility", "technical"),
    "rsi_14":     ("RSI(14)", "technical"),
    "dist_sma50": ("Price vs 50-day avg", "technical"),
    "dist_sma200":("Price vs 200-day avg", "technical"),
    "volume_z":   ("Volume surprise", "technical"),
    "rel_str_20": ("20-day return minus S&P", "technical"),
    # macro
    "spx_ret_5":  ("S&P 500 5-day return", "macro"),
    "vix":        ("VIX level", "macro"),
    "vix_chg_5":  ("VIX 5-day change", "macro"),
    "tnx_chg_20": ("10y yield 20-day change", "macro"),
    "dxy_ret_20": ("Dollar 20-day return", "macro"),
    "oil_ret_20": ("Oil 20-day return", "macro"),
    "gold_ret_20":("Gold 20-day return", "macro"),
    # fundamental
    "eps_surprise":  ("Last EPS surprise %", "fundamental"),
    "days_since_er": ("Days since earnings", "fundamental"),
}
FEATURES = list(FEATURE_INFO)


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def macro_features(macro, index):
    m = macro.reindex(index).ffill()
    f = pd.DataFrame(index=index)
    f["spx_ret_5"] = m["spx"].pct_change(5)
    f["vix"] = m["vix"]
    f["vix_chg_5"] = m["vix"].diff(5)
    f["tnx_chg_20"] = m["tnx"].diff(20)
    f["dxy_ret_20"] = m["dxy"].pct_change(20)
    f["oil_ret_20"] = m["oil"].pct_change(20)
    f["gold_ret_20"] = m["gold"].pct_change(20)
    f["spx_ret_20"] = m["spx"].pct_change(20)
    f["spx_fwd_h"] = m["spx"].shift(-HORIZON) / m["spx"] - 1  # target only
    return f


def earnings_features(earn, index):
    """Earnings are released ~16:00 after the close, so a report dated day d is
    only usable from the next trading day."""
    e = earn.dropna(subset=["Surprise(%)"]).copy()
    e = e[~e.index.duplicated()].sort_index()
    avail = pd.DataFrame({
        "avail": e.index.normalize() + pd.Timedelta(days=1),
        "eps_surprise": e["Surprise(%)"].clip(-100, 100).values,
        "er_date": e.index.normalize(),
    }).sort_values("avail")
    left = pd.DataFrame({"date": index})
    out = pd.merge_asof(left, avail, left_on="date", right_on="avail", direction="backward")
    out["days_since_er"] = (out["date"] - out["er_date"]).dt.days.clip(upper=120)
    return out.set_index("date")[["eps_surprise", "days_since_er"]]


def build_panel(close, volume, macro, earnings=None):
    """Return one long table: (date, ticker) rows, features + targets.
    earnings may be None (or miss a ticker); those rows get NaN earnings features."""
    mf = macro_features(macro, close.index)
    frames = []
    for t in close.columns:
        c = close[t].dropna()
        v = volume[t].reindex(c.index)
        f = pd.DataFrame(index=c.index)
        r = c.pct_change()
        f["ret_1"] = r
        f["ret_5"] = c.pct_change(5)
        f["ret_20"] = c.pct_change(20)
        f["vol_20"] = r.rolling(20).std() * np.sqrt(252)
        f["rsi_14"] = rsi(c)
        f["dist_sma50"] = c / c.rolling(50).mean() - 1
        f["dist_sma200"] = c / c.rolling(200).mean() - 1
        lv = np.log(v.replace(0, np.nan))
        f["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
        f = f.join(mf)
        f["rel_str_20"] = f["ret_20"] - f["spx_ret_20"]
        if earnings is not None and t in earnings and len(earnings[t]):
            f = f.join(earnings_features(earnings[t], c.index))
        else:  # market-wide universe: no per-stock earnings history
            f["eps_surprise"] = np.nan
            f["days_since_er"] = np.nan
        # targets (used only for training and scoring, never as features)
        f["fwd_ret_1"] = c.shift(-1) / c - 1
        f["fwd_ret_h"] = c.shift(-HORIZON) / c - 1
        # realized volatility over the next H days (annualized), for the volatility study
        f["fwd_vol_h"] = r.rolling(HORIZON).std().shift(-HORIZON) * np.sqrt(252)
        f["vol_5"] = r.rolling(5).std() * np.sqrt(252)
        f["fwd_excess_h"] = f["fwd_ret_h"] - f["spx_fwd_h"]  # beat the S&P?
        f["close"] = c
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    return panel.reset_index()

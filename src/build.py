"""Run the whole study and write web/data.json for the dashboard.

    python build.py            # use cached data
    python build.py --refresh  # re-download everything
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import OUT, RAW, TICKERS, HORIZON, FIRST_TEST_YEAR, COST_BPS
from features import FEATURES, FEATURE_INFO, build_panel
from fetch import fetch_all, load
from sentiment import score_news

RNG = 42


def r(x, k=4):
    """Round + turn NaN into null for JSON."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), k)


def rl(a, k=4):
    return [r(x, k) for x in a]


# ---------------------------------------------------------------- analysis
def correlation_matrix(close, macro):
    m = macro.reindex(close.index).ffill()
    rets = close.pct_change()
    rets["S&P500"] = m["spx"].pct_change()
    rets["VIX"] = m["vix"].pct_change()
    rets["10y yield"] = m["tnx"].diff()
    rets["Dollar"] = m["dxy"].pct_change()
    rets["Oil"] = m["oil"].pct_change()
    rets["Gold"] = m["gold"].pct_change()
    rets = rets[rets.index >= f"{FIRST_TEST_YEAR}-01-01"].replace([np.inf, -np.inf], np.nan)
    c = rets.corr()
    return {"labels": list(c.columns), "values": [rl(row, 3) for row in c.values]}, rets


def same_day_vs_next_day(daily):
    """Same-day correlation 'explains' moves; next-day correlation is what
    prediction needs."""
    macros = ["S&P500", "VIX", "10y yield", "Dollar", "Oil", "Gold"]
    out = {}
    for t in TICKERS:
        nxt = daily[t].shift(-1)
        out[t] = {mname: [r(daily[t].corr(daily[mname]), 3), r(nxt.corr(daily[mname]), 3)]
                  for mname in macros}
    return out


def information_coefficients(panel):
    """Spearman rank correlation between each feature today and the next-H-day
    return. Also: share of years where the sign was the same as overall."""
    p = panel[panel["date"] >= f"{FIRST_TEST_YEAR}-01-01"].dropna(subset=FEATURES + ["fwd_ret_h"])
    pooled, per_t, stab = [], {t: [] for t in TICKERS}, []
    years = p["date"].dt.year
    for f in FEATURES:
        ic = p[f].corr(p["fwd_ret_h"], method="spearman")
        pooled.append(r(ic, 4))
        yearly = p.groupby(years).apply(lambda g: g[f].corr(g["fwd_ret_h"], method="spearman"))
        stab.append(r((np.sign(yearly) == np.sign(ic)).mean(), 3))
        for t in TICKERS:
            q = p[p["ticker"] == t]
            per_t[t].append(r(q[f].corr(q["fwd_ret_h"], method="spearman"), 4))
    return {"pooled": pooled, "per_ticker": per_t, "stability": stab}


def rolling_corr(daily, win=120):
    out = {}
    for t in TICKERS:
        d = pd.DataFrame({
            "spx": daily[t].rolling(win).corr(daily["S&P500"]),
            "vix": daily[t].rolling(win).corr(daily["VIX"]),
            "tnx": daily[t].rolling(win).corr(daily["10y yield"]),
            "oil": daily[t].rolling(win).corr(daily["Oil"]),
        }).dropna().iloc[::5]  # weekly sampling keeps file small
        out[t] = {"dates": [x.strftime("%Y-%m-%d") for x in d.index],
                  **{k: rl(d[k], 3) for k in d.columns}}
    return out


# ---------------------------------------------------------------- modeling
def make_models():
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        "gbm": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=300,
            l2_regularization=1.0, random_state=RNG),
    }


def split(data, year):
    """Train on everything before `year`, minus an embargo gap so the last
    training targets (which look H days ahead) don't overlap the test year."""
    start = pd.Timestamp(f"{year}-01-01")
    embargo = start - pd.Timedelta(days=HORIZON * 2 + 4)
    tr = data[data["date"] < embargo]
    te = data[(data["date"] >= start) & (data["date"] < f"{year + 1}-01-01")]
    return tr, te


def walk_forward(panel, feats=FEATURES, models=("logit", "gbm"), shuffle=False, target="fwd_ret_h"):
    data = panel.dropna(subset=feats).copy()
    data["fwd_ret_h"] = data[target]
    data["y"] = (data["fwd_ret_h"] > 0).astype(int)
    last_year = data["date"].dt.year.max()
    preds = []
    for year in range(FIRST_TEST_YEAR, last_year + 1):
        tr, te = split(data, year)
        tr = tr[tr["fwd_ret_h"].notna()]
        y_tr = tr["y"].values
        if shuffle:
            y_tr = np.random.default_rng(year).permutation(y_tr)
        res = te[["date", "ticker"]].copy()
        res["base_rate"] = y_tr.mean()
        for name in models:
            m = make_models()[name].fit(tr[feats].values, y_tr)
            res[f"p_{name}"] = m.predict_proba(te[feats].values)[:, 1]
        preds.append(res)
    out = pd.concat(preds)
    out = out.merge(data[["date", "ticker", "y", "fwd_ret_h", "fwd_ret_1"]], on=["date", "ticker"])
    out.loc[out["fwd_ret_h"].isna(), "y"] = np.nan
    return out


def scores(df, col):
    d = df.dropna(subset=["y"])
    return {"acc": r(accuracy_score(d["y"], d[col] > 0.5), 4),
            "auc": r(roc_auc_score(d["y"], d[col]), 4)}


def per_year_table(pred):
    rows = []
    d = pred.dropna(subset=["y"])
    for year, g in d.groupby(d["date"].dt.year):
        rows.append({
            "year": int(year), "n": int(len(g)),
            "up_share": r(g["y"].mean(), 4),
            "base_acc": r(accuracy_score(g["y"], np.ones(len(g))), 4),
            **{f"{m}_{k}": v for m in ("logit", "gbm") for k, v in scores(g, f"p_{m}").items()},
        })
    return rows


def backtest(pred_t, thresh=0.5):
    """Long when model says P(up) > thresh, else hold cash. Daily rebalancing,
    COST_BPS charged on every switch."""
    d = pred_t.sort_values("date").dropna(subset=["fwd_ret_1"])
    pos = (d["p_gbm"] > thresh).astype(float).values
    ret = d["fwd_ret_1"].values
    cost = np.abs(np.diff(np.r_[0, pos])) * COST_BPS / 1e4
    strat = pos * ret - cost

    def stats(x):
        eq = np.cumprod(1 + x)
        yrs = len(x) / 252
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        return {"cagr": r(eq[-1] ** (1 / yrs) - 1), "vol": r(x.std() * np.sqrt(252)),
                "sharpe": r(x.mean() / x.std() * np.sqrt(252), 3), "maxdd": r(dd),
                "total": r(eq[-1] - 1)}

    return {
        "dates": [x.strftime("%Y-%m-%d") for x in d["date"]],
        "strat": rl(np.cumprod(1 + strat), 4),
        "bh": rl(np.cumprod(1 + ret), 4),
        "stats_strat": stats(strat), "stats_bh": stats(ret),
        "time_in_market": r(pos.mean(), 3), "trades": int(np.abs(np.diff(pos)).sum()),
    }


def importance(panel):
    """Permutation importance of the GBM on the latest full test year: how much AUC drops
    when one feature is shuffled."""
    data = panel.dropna(subset=FEATURES + ["fwd_ret_h"]).copy()
    data["y"] = (data["fwd_ret_h"] > 0).astype(int)
    year = data["date"].dt.year.max() - 1
    tr, te = split(data, year)
    te = pd.concat([te, data[data["date"] >= f"{year + 1}-01-01"]])
    m = make_models()["gbm"].fit(tr[FEATURES].values, tr["y"].values)
    pi = permutation_importance(m, te[FEATURES].values, te["y"].values, scoring="roc_auc",
                                n_repeats=10, random_state=RNG)
    return {"test_from": f"{year}-01-01", "mean": rl(pi.importances_mean, 5),
            "std": rl(pi.importances_std, 5)}


# ---------------------------------------------------------------- volatility
VOL_FEATURES = FEATURES + ["vol_5"]


def volatility_study(panel):
    """Predict how much a stock will move (realized volatility over the next H
    days) instead of which way. Volatility clusters, so this should be far more
    predictable than direction. Same walk-forward splits as the direction model.
    Models work on log volatility, which is closer to normally distributed."""
    data = panel.dropna(subset=VOL_FEATURES + ["fwd_vol_h"]).copy()
    data = data[data["fwd_vol_h"] > 0]
    data["y"] = np.log(data["fwd_vol_h"])
    preds = []
    for year in range(FIRST_TEST_YEAR, data["date"].dt.year.max() + 1):
        tr, te = split(data, year)
        if te.empty:
            continue
        res = te[["date", "ticker", "fwd_vol_h", "vol_20", "fwd_ret_1"]].copy()
        res["naive"] = te["vol_20"]          # "next week looks like the last month"
        X_tr, X_te = tr[VOL_FEATURES].values, te[VOL_FEATURES].values
        lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, tr["y"])
        gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                            min_samples_leaf=200, random_state=RNG).fit(X_tr, tr["y"])
        res["linear"] = np.exp(lin.predict(X_te))
        res["gbm"] = np.exp(gbm.predict(X_te))
        preds.append(res)
    p = pd.concat(preds)

    def score(col):
        y, yhat = np.log(p["fwd_vol_h"]), np.log(p[col].clip(lower=1e-4))
        return {"r2": r(r2_score(y, yhat), 4), "corr": r(np.corrcoef(y, yhat)[0, 1], 4),
                "mae": r((p["fwd_vol_h"] - p[col]).abs().mean(), 4)}

    overall = {m: score(m) for m in ("naive", "linear", "gbm")}
    series = {}
    for t in TICKERS:
        q = p[p["ticker"] == t].sort_values("date").iloc[::5]   # weekly points keep the file small
        series[t] = {"dates": [x.strftime("%Y-%m-%d") for x in q["date"]],
                     "realized": rl(q["fwd_vol_h"], 3), "gbm": rl(q["gbm"], 3), "naive": rl(q["naive"], 3)}
    latest = {}
    last = panel.dropna(subset=VOL_FEATURES)
    tr = data  # train on everything labeled for the live forecast
    gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                        min_samples_leaf=200, random_state=RNG).fit(tr[VOL_FEATURES].values, tr["y"])
    for t in TICKERS:
        row = last[last["ticker"] == t].iloc[-1]
        latest[t] = {"date": row["date"].strftime("%Y-%m-%d"),
                     "pred": r(float(np.exp(gbm.predict(row[VOL_FEATURES].values.reshape(1, -1).astype(float))[0])), 4),
                     "vol_20": r(row["vol_20"], 4)}
    return {"overall": overall, "n": int(len(p)), "series": series, "latest": latest,
            "sizing": vol_sizing(p)}


def perf_stats(x):
    """Annual return, volatility, Sharpe ratio and max drawdown of daily returns."""
    x = np.asarray(x, float)
    eq = np.cumprod(1 + x)
    yrs = len(x) / 252
    return {"cagr": r(eq[-1] ** (1 / yrs) - 1), "vol": r(x.std() * np.sqrt(252)),
            "sharpe": r(x.mean() / x.std() * np.sqrt(252), 3),
            "maxdd": r((eq / np.maximum.accumulate(eq) - 1).min()), "total": r(eq[-1] - 1)}


def vol_sizing(p, cap=1.0):
    """Use the volatility forecast to decide HOW MUCH to hold, not whether to hold.
    Weight = (the stock's typical forecast volatility so far) / (this week's forecast),
    capped at 100% (no borrowing). Calm weeks -> fully invested, stormy weeks -> less.
    The "typical" level is an expanding median of past forecasts only, so there is no
    look-ahead. Rebalanced weekly; COST_BPS charged on every change in weight.
    Compared with buy-and-hold and with the same rule driven by the naive 20-day volatility."""
    out, agg = {}, {k: [] for k in ("bh", "model", "naive")}
    for t in TICKERS:
        q = p[p["ticker"] == t].sort_values("date").dropna(subset=["fwd_ret_1"]).reset_index(drop=True)
        ret = q["fwd_ret_1"].values
        series = {"bh": ret}
        for name, col in (("model", "gbm"), ("naive", "naive")):
            f = q[col]
            typical = f.expanding().median().shift(1).fillna(f)
            w = (typical / f).clip(upper=cap)
            w = w.where(q.index % HORIZON == 0).ffill().values   # weekly rebalance
            cost = np.abs(np.diff(np.r_[0, w])) * COST_BPS / 1e4
            series[name] = w * ret - cost
            series[name + "_w"] = w
        stats = {k: perf_stats(series[k]) for k in ("bh", "model", "naive")}
        for k in agg:
            agg[k].append(stats[k])
        idx = np.arange(0, len(q), 5)
        out[t] = {"dates": [q["date"].iloc[i].strftime("%Y-%m-%d") for i in idx],
                  **{k: rl(np.cumprod(1 + series[k])[idx], 4) for k in ("bh", "model", "naive")},
                  "weight": rl(series["model_w"][idx], 3),
                  "stats": stats, "avg_weight": r(series["model_w"].mean(), 3)}
    summary = {k: {m: r(np.mean([s[m] for s in v]), 4) for m in ("cagr", "vol", "sharpe", "maxdd")}
               for k, v in agg.items()}
    wins = sum(out[t]["stats"]["model"]["sharpe"] > out[t]["stats"]["bh"]["sharpe"] for t in TICKERS)
    dd_wins = sum(out[t]["stats"]["model"]["maxdd"] > out[t]["stats"]["bh"]["maxdd"] for t in TICKERS)
    return {"per_ticker": out, "summary": summary, "sharpe_wins": int(wins), "dd_wins": int(dd_wins)}


# ---------------------------------------------------------------- main
def main():
    fetch_all(refresh="--refresh" in sys.argv)
    close, volume, macro, earnings, news = load()
    panel = build_panel(close, volume, macro, earnings)
    print("panel", panel.shape)

    corr, daily = correlation_matrix(close, macro)
    sameday = same_day_vs_next_day(daily)
    ic = information_coefficients(panel)
    roll = rolling_corr(daily)
    print("analysis done")

    pred = walk_forward(panel)
    # saved so llm_forecast.py can compare Claude with the ML model on the same rows
    pred.to_csv(RAW.parent / "ml_predictions.csv", index=False)
    overall = {m: scores(pred, f"p_{m}") for m in ("logit", "gbm")}
    lab = pred.dropna(subset=["y"])
    overall["base"] = {"acc": r(lab["y"].mean(), 4), "auc": 0.5}
    print("overall", overall)

    groups = {g: [f for f in FEATURES if FEATURE_INFO[f][1] == g]
              for g in ("technical", "macro", "fundamental")}
    ablation = {}
    for g, feats in groups.items():
        ablation[g] = scores(walk_forward(panel, feats, models=("gbm",)), "p_gbm")
    ablation["all"] = overall["gbm"]
    shuffled = scores(walk_forward(panel, models=("gbm",), shuffle=True), "p_gbm")
    pred_x = walk_forward(panel, target="fwd_excess_h")
    excess = {m: scores(pred_x, f"p_{m}") for m in ("logit", "gbm")}
    excess["base"] = {"acc": r(pred_x.dropna(subset=["y"])["y"].mean(), 4), "auc": 0.5}
    print("ablation", ablation, "shuffled", shuffled, "excess", excess)
    imp = importance(panel)
    vol = volatility_study(panel)
    print("volatility", vol["overall"])
    print("sizing", vol["sizing"]["summary"], "sharpe wins", vol["sizing"]["sharpe_wins"], "dd wins", vol["sizing"]["dd_wins"])

    tickers = {}
    for t in TICKERS:
        p_t = panel[(panel["ticker"] == t) & (panel["date"] >= f"{FIRST_TEST_YEAR}-01-01")]
        pr = pred[pred["ticker"] == t]
        c = p_t.set_index("date")["close"]
        prob = pr.set_index("date")["p_gbm"].reindex(c.index)
        sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
        full_c = panel[panel["ticker"] == t].set_index("date")["close"]
        sma50, sma200 = full_c.rolling(50).mean().reindex(c.index), full_c.rolling(200).mean().reindex(c.index)
        last = panel[panel["ticker"] == t].dropna(subset=FEATURES).iloc[-1]
        tickers[t] = {
            "dates": [x.strftime("%Y-%m-%d") for x in c.index],
            "close": rl(c, 2), "sma50": rl(sma50, 2), "sma200": rl(sma200, 2),
            "rsi": rl(p_t["rsi_14"], 1), "prob": rl(prob, 3),
            "scores": scores(pr, "p_gbm"),
            "backtest": backtest(pr),
            "latest": {"date": last["date"].strftime("%Y-%m-%d"),
                       "prob": r(pr.sort_values("date")["p_gbm"].iloc[-1], 3),
                       "features": {f: r(last[f], 4) for f in FEATURES}},
        }

    out = {
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "tickers": TICKERS,
                 "horizon": HORIZON, "first_test_year": FIRST_TEST_YEAR, "cost_bps": COST_BPS,
                 "features": FEATURES,
                 "feature_names": [FEATURE_INFO[f][0] for f in FEATURES],
                 "feature_groups": [FEATURE_INFO[f][1] for f in FEATURES],
                 "n_predictions": int(len(lab))},
        "corr": corr, "sameday": sameday, "ic": ic, "rolling": roll,
        "model": {"overall": overall, "per_year": per_year_table(pred), "ablation": ablation,
                  "shuffled": shuffled, "excess": excess, "importance": imp},
        "tickers": tickers,
        "volatility": vol,
        "sentiment": score_news(news),
        "llm_forecast": (json.loads((RAW.parent / "llm_forecast.json").read_text())
                         if (RAW.parent / "llm_forecast.json").exists() else None),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("wrote", OUT / "data.json", round((OUT / "data.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()

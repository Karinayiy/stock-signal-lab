# Stock Signal Lab

Can technical indicators, macro data, earnings surprises, news sentiment, or a large language model predict where a stock goes over the next 5 trading days? This project tests that honestly, with walk-forward backtests on data the models never saw.

## Website

The dashboard is a static site in `web/`. It opens with a **risk planner**: pick a stock, an amount and the volatility you accept, and it sizes the position from next week's volatility forecast and shows the likely price range. Everything runs in the browser.

`.github/workflows/site.yml` rebuilds the study after every US trading day (22:30 UTC, Monday to Friday), saves the growing news archive back to the repo, and publishes `web/` to GitHub Pages. It needs no API key. To turn it on: repo **Settings → Pages → Source: GitHub Actions**, then **Actions → Update website → Run workflow**. (GitHub Pages on a free account needs a public repo; `.env` is never committed.)

## Your portfolio and live prices (local app)

In the local app, the **Portfolio** section takes your holdings (type `AAPL 10 150` per line, or import a broker CSV with Symbol and Quantity columns) and shows value, today's change, gain vs cost, volatility vs the S&P 500, a bad-day estimate, each holding's share of the risk, correlations, and how today's mix did over the past year. Prices come from Yahoo Finance: free, no key, and usually real-time or up to 15 minutes late. The planner also switches to live prices, refreshed every minute. Holdings are saved only in your browser and sent only to the app on your own computer (`src/portfolio.py`), never to the public website. It describes risk; it does not say what to buy or sell.

## Start the app

Double-click **`Stock Signal Lab.command`** in Finder, or run:

```bash
.venv/bin/python app.py
```

It opens the dashboard at http://127.0.0.1:8770 and adds live buttons:

- **Refresh data**: re-download prices and news, then rerun the whole study.
- **Ask Claude about a stock now**: a live Claude forecast for the latest trading day.
- **Run Claude experiment**: the full Claude-vs-ML forecasting test (about $2).

The app only listens on this computer.

### Turn on the Claude features

```bash
cp .env.example .env
```

Open `.env` and paste your Anthropic API key after `ANTHROPIC_API_KEY=`, then restart the app. `.env` is ignored by git, so the key stays on this computer. Without a key, everything else still works and news is scored with the VADER word list.

## Run the pipeline without the app

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd src && ../.venv/bin/python build.py --refresh   # download, analyse, train, write web/data.json
../.venv/bin/python llm_forecast.py --dry-run      # preview the Claude experiment (free)
../.venv/bin/python llm_forecast.py                # run it (needs the API key)
```

## Files

| File | What it does |
|---|---|
| `app.py` | Local app: serves the dashboard and runs live actions |
| `src/config.py` | Tickers, macro series, horizon, test years; loads `.env` |
| `src/fetch.py` | Download and cache Yahoo Finance data |
| `src/features.py` | 18 features, all known at that day's close (no look-ahead) |
| `src/build.py` | Correlations, IC, walk-forward direction models, backtest, volatility study, JSON export |
| `src/sentiment.py` | News sentiment: Claude scorer with VADER fallback, daily headline archive |
| `src/portfolio.py` | Live quotes and portfolio risk analysis for the local app |
| `src/llm_forecast.py` | Claude-as-forecaster experiment vs the ML models (3 groups, cost tracking) |
| `tests/test_no_lookahead.py` | Proves no feature uses future data |
| `Save API Key.command` | Saves the API key on the clipboard into `.env` |
| `web/index.html` | Website: risk planner plus the full study |
| `.github/workflows/site.yml` | Daily rebuild and deploy to GitHub Pages |

## Findings so far

- **Direction is not predictable from this data.** Over 2016–2026 (about 21,800 out-of-sample predictions), gradient boosting scores 53.3% accuracy vs 56.9% for "always guess up", with AUC 0.49. Predicting which stock beats the S&P 500 does no better.
- **Same day is not next day.** Stocks move with the S&P 500 (correlation about 0.64) and against VIX on the same day, but next-day correlations are near zero.
- **No data leak.** Training on shuffled labels gives AUC 0.48, as it should.
- **Volatility is predictable.** The same features predict next week's volatility with out-of-sample R² 0.41 (boosting) vs 0.23 for the naive "last 20 days" guess.
- **Claude doesn't beat the base rate either.** On 152 post-cutoff samples (Jul–Nov 2026), Claude kept every forecast between 50% and 58% and called "up" 99% of the time, matching "always guess up" (accuracy 56.6% vs 55.9%, AUC 0.50 anonymous, 0.42 named). A third run with no base-rate hint (cost $0.76) stayed just as cautious (48%–57%) and was slightly worse (accuracy 53.3%, AUC 0.46), so the caution is Claude's own judgment, not the prompt.
- **Volatility forecasts are useful for sizing.** Holding less when high volatility is forecast cut average volatility from 35% to 27% and the worst drop from −53% to −43%, and raised the Sharpe ratio for 7 of 8 stocks, at the cost of a lower total return (23% vs 28% a year). The simple "last 20 days" rule does about as well.
- **Leak test passes.** `tests/test_no_lookahead.py` rebuilds all features from data cut off at three past dates; nothing on or before the cutoff changes.
- **Claude reads news better than a word list.** Only 10 of 50 headlines in Yahoo's AAPL feed are actually about Apple; Claude flags the rest as irrelevant, and for MSFT, TSLA and JPM it disagrees with VADER on whether the news is positive or negative.
- **Most stable direction signal:** high VIX is followed by slightly higher returns (IC +0.03, same sign in all 11 test years). It's too weak to trade on its own.

## Claude as forecaster

`src/llm_forecast.py` gives Claude the same 18 features as a short analyst briefing and asks for the probability that the stock rises over the next 5 days.

- Only dates after Claude Opus 5.5's training-data cutoff (June 2026) are tested, so it can't remember outcomes.
- One date every 5 trading days, so the 5-day windows don't overlap (152 samples per group).
- Two groups: **anonymous** (numbers only) and **named** (adds company and date).
- Scored by accuracy, AUC and Brier score with 95% bootstrap intervals. With about 150 samples per group, accuracy is only known to within about ±8 points.

## Ideas to extend

1. Once the news archive (grown daily by the workflow) covers a few months, test Claude news sentiment as a feature.
2. Add more stocks (for example all of the S&P 500) to reduce survivorship bias.
3. Try longer horizons (1–3 months), where valuation and earnings may matter more.
4. Combine Claude's news relevance filter with the daily archive to test whether only-relevant news sentiment predicts anything.

Not investment advice.

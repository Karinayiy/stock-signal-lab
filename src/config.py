"""Project-wide settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "web"
ENV_FILE = ROOT / ".env"


def load_env():
    """Read KEY=value lines from the project's .env file (kept out of git) into the
    environment, without overriding variables that are already set. This lets the
    Anthropic API key live in one private file instead of every terminal window."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


load_env()

# Stocks studied (mix of sectors so results are not just "big tech").
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "JPM", "XOM"]

# Macro / market series.
MACRO = {
    "^GSPC": "spx",      # S&P 500 index
    "^VIX": "vix",       # volatility "fear" index
    "^TNX": "tnx",       # 10-year Treasury yield (%)
    "DX-Y.NYB": "dxy",   # US dollar index
    "CL=F": "oil",       # WTI crude futures
    "GC=F": "gold",      # gold futures
}

START = "2010-01-01"
HORIZON = 5            # predict sign of next-5-trading-day return
FIRST_TEST_YEAR = 2016 # walk-forward test starts here
COST_BPS = 5           # trading cost per position change

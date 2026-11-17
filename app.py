"""Stock Signal Lab as a local app.

Serves the dashboard at http://127.0.0.1:8770 and adds live actions the static
page can't do:
  * Refresh data      re-download prices/news and rerun the whole study
  * Ask Claude now    live Claude forecast for one stock, using today's numbers
  * Run experiment    the full Claude-vs-ML forecasting experiment (llm_forecast.py)

Run:  .venv/bin/python app.py      (or double-click "Stock Signal Lab.command")
Only listens on this computer (127.0.0.1); nothing is exposed to the network.
"""
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC, WEB = ROOT / "src", ROOT / "web"
sys.path.insert(0, str(SRC))

from config import TICKERS  # noqa: E402  (also loads .env)

PORT = int(os.environ.get("SSL_PORT", "8770"))
JOB = {"name": None, "running": False, "log": [], "started": None, "finished": None, "ok": None}
JOB_LOCK = threading.Lock()
_PANEL = {"panel": None, "mtime": None}


def has_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def start_job(name, args):
    """Run a pipeline script in the background and collect its output."""
    with JOB_LOCK:
        if JOB["running"]:
            return False
        JOB.update(name=name, running=True, log=[], started=time.time(), finished=None, ok=None)

    def work():
        # -u: unbuffered output, so progress lines reach the page as they happen
        proc = subprocess.Popen([sys.executable, "-u", *args], cwd=SRC, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        for line in proc.stdout:
            if "Warning" in line or line.strip().startswith("return fnb"):
                continue
            with JOB_LOCK:
                JOB["log"] = (JOB["log"] + [line.rstrip()])[-40:]
        proc.wait()
        with JOB_LOCK:
            JOB.update(running=False, finished=time.time(), ok=proc.returncode == 0)
        _PANEL["panel"] = None   # data changed; rebuild the feature table on next use

    threading.Thread(target=work, daemon=True).start()
    return True


def panel():
    """Feature table built from the cached raw data, reused between requests."""
    from features import build_panel
    from fetch import load
    mtime = (ROOT / "data" / "raw" / "close.csv").stat().st_mtime
    if _PANEL["panel"] is None or _PANEL["mtime"] != mtime:
        close, volume, macro, earnings, _ = load()
        _PANEL.update(panel=build_panel(close, volume, macro, earnings), mtime=mtime)
    return _PANEL["panel"]


def live_forecast(ticker):
    """Ask Claude about the latest trading day for one stock (named version)."""
    import anthropic
    from features import FEATURES
    from llm_forecast import ask, make_prompt, recent_returns
    p = panel()
    row = p[p["ticker"] == ticker].dropna(subset=FEATURES).sort_values("date").iloc[-1]
    prompt = make_prompt(row, recent_returns(p, ticker, row["date"]), "named")
    prob, why, model, _ = ask(anthropic.Anthropic(), prompt)
    return {"ticker": ticker, "date": row["date"].strftime("%Y-%m-%d"), "prob": prob,
            "rationale": why, "model": model}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):   # keep the console quiet
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        if self.path.endswith((".json", ".html", "/")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path == "/api/status":
            with JOB_LOCK:
                job = dict(JOB)
            return self.send_json({"app": True, "has_key": has_key(), "job": job, "tickers": TICKERS})
        return super().do_GET()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self.send_json({"error": "Request body must be JSON."}, 400)

        if self.path == "/api/refresh":
            ok = start_job("Refresh data", ["build.py", "--refresh"])
            return self.send_json({"started": ok} if ok else {"error": "Another job is still running."}, 200 if ok else 409)

        if self.path == "/api/experiment":
            if not has_key():
                return self.send_json({"error": "No Anthropic API key. Add it to the .env file and restart the app."}, 400)
            ok = start_job("Claude experiment", ["llm_forecast.py"])
            return self.send_json({"started": ok} if ok else {"error": "Another job is still running."}, 200 if ok else 409)

        if self.path == "/api/forecast":
            t = body.get("ticker")
            if t not in TICKERS:
                return self.send_json({"error": f"Unknown ticker {t!r}."}, 400)
            if not has_key():
                return self.send_json({"error": "No Anthropic API key. Add it to the .env file and restart the app."}, 400)
            try:
                return self.send_json(live_forecast(t))
            except Exception as e:  # report API or data problems to the page instead of crashing
                return self.send_json({"error": f"{type(e).__name__}: {e}"}, 502)

        return self.send_json({"error": "Not found."}, 404)


def main():
    if not (WEB / "data.json").exists():
        print("No data yet; running the pipeline once (about a minute)...")
        subprocess.run([sys.executable, "build.py"], cwd=SRC, check=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Stock Signal Lab is running at {url}")
    print(f"Claude features: {'on' if has_key() else 'off (add ANTHROPIC_API_KEY to .env)'}")
    print("Press Ctrl+C to stop.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

"""Download 10 years of adjusted daily prices and store them as the offline snapshot.

Run once (and again before the presentation) on a machine with internet access:
    python make_snapshot.py
then commit data/prices_snapshot.csv to GitHub. The app uses it when Yahoo fails
or when "Snapshot (offline)" is selected in the sidebar.
"""
from pathlib import Path

import yfinance as yf

TICKERS = ["VGSH", "VGIT", "VMBS", "VCIT", "VTI", "VEA"]
OUTPUT = Path(__file__).parent / "data" / "prices_snapshot.csv"

raw = yf.download(TICKERS, period="10y", interval="1d", auto_adjust=True, progress=False)
close = raw["Close"][TICKERS].dropna(how="all")
close.index = close.index.tz_localize(None) if close.index.tz is not None else close.index
missing = [ticker for ticker in TICKERS if close[ticker].dropna().empty]
if missing:
    raise SystemExit(f"No data for {missing}; snapshot not written.")
OUTPUT.parent.mkdir(exist_ok=True)
close.round(6).to_csv(OUTPUT)
print(f"Saved {len(close)} rows from {close.index.min().date()} to {close.index.max().date()} -> {OUTPUT}")

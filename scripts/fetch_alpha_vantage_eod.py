"""
Fetch missing EOD daily price history from Alpha Vantage and save parquet
files in the schema the engine expects.

API key is read from the ALPHAVANTAGE_API_KEY environment variable. The key
is never read from a config file or written to disk — pass it via:

    export ALPHAVANTAGE_API_KEY="your_key_here"
    python scripts/fetch_alpha_vantage_eod.py

or inline for one-off use:

    ALPHAVANTAGE_API_KEY=xyz python scripts/fetch_alpha_vantage_eod.py --max 25

Schema (matches existing `*.US.eod.parquet`):
  date              datetime64[ns]
  open / high / low / close   float64
  adjusted_close    float64
  volume            int64

Strategy:
  1. Try TIME_SERIES_DAILY_ADJUSTED first (gives true split- + dividend-adjusted
     close). This is premium-only on AV since late 2023; if the API returns an
     "Information" message, we fall back to the free TIME_SERIES_DAILY endpoint
     and set adjusted_close = close (split-adjusted only — accept the small
     dividend-adjustment error for delisted names where this is the best we have).
  2. Respect rate limits — sleep `--sleep-ms` between calls (default 13s for
     the 5/min free-tier limit), and stop if AV returns the daily-cap "Note".
  3. Idempotent — never overwrites an existing parquet.
  4. Targets are the in-window R1000 missing tickers, ranked by snapshot count.

Usage:
  python scripts/fetch_alpha_vantage_eod.py                 # fetch up to --max 25
  python scripts/fetch_alpha_vantage_eod.py --max 25        # explicit
  python scripts/fetch_alpha_vantage_eod.py ATVI VMW SGEN   # specific tickers
  python scripts/fetch_alpha_vantage_eod.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_alpha.config import get_cfg                            # noqa: E402
from project_alpha.data import load_russell_panel                   # noqa: E402

API_URL = "https://www.alphavantage.co/query"


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------
def to_alpha_symbol(eod_ticker: str) -> str:
    """Class-share dot to hyphen: BRK.B → BRK-B (matches Yahoo's convention)."""
    return eod_ticker.replace(".", "-")


def is_clean_ticker(t: str) -> bool:
    if not isinstance(t, str) or not t:
        return False
    if not re.match(r"^[A-Z]", t):
        return False
    if any(suffix in t for suffix in (".I", ".1", "WS", "WI")):
        return False
    return bool(re.match(r"^[A-Z][A-Z0-9.\-]{0,5}$", t))


# ---------------------------------------------------------------------------
# Single-ticker fetch
# ---------------------------------------------------------------------------
def _request_json(params: dict, timeout: int = 30) -> dict:
    r = requests.get(API_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_one(eod_ticker: str, api_key: str) -> Tuple[pd.DataFrame, str]:
    """Returns (DataFrame, status).

    Uses only TIME_SERIES_DAILY (free tier — TIME_SERIES_DAILY_ADJUSTED became
    premium-only in Nov 2023). adjusted_close is set to raw close, which omits
    dividend adjustments (~2%/yr error for typical dividend stocks). For
    delisted-name back-fill this is the best free option.
    """
    sym = to_alpha_symbol(eod_ticker)
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": sym,
        "outputsize": "full",
        "apikey": api_key,
        "datatype": "json",
    }
    try:
        data = _request_json(params)
    except Exception as e:
        return pd.DataFrame(), f"error: {e}"

    # Rate-limit / daily-cap notices come back as Note / Information
    if "Note" in data:
        return pd.DataFrame(), f"rate_limited: {data['Note'][:120]}"
    if "Information" in data:
        return pd.DataFrame(), f"rate_limited: {data['Information'][:120]}"
    if "Error Message" in data:
        return pd.DataFrame(), f"error: {data['Error Message'][:120]}"

    ts = data.get("Time Series (Daily)", {})
    if not ts:
        return pd.DataFrame(), "no_data"

    rows = []
    for date, v in ts.items():
        rows.append({
            "date":           pd.Timestamp(date),
            "open":           float(v["1. open"]),
            "high":           float(v["2. high"]),
            "low":             float(v["3. low"]),
            "close":          float(v["4. close"]),
            "adjusted_close": float(v["4. close"]),   # split-adjusted close (no div adj on free tier)
            "volume":         int(float(v["5. volume"])),
        })
    df = (pd.DataFrame(rows)
            .sort_values("date")
            .reset_index(drop=True))
    return df, "ok"


# ---------------------------------------------------------------------------
# Discovery — what's still missing AND in the active backtest window
# ---------------------------------------------------------------------------
def discover_in_window_missing(prices_dir: str, russell_xlsx: str,
                                window_start: pd.Timestamp) -> List[str]:
    panel = load_russell_panel(russell_xlsx)
    universe = sorted(set(panel["ticker"].astype(str)))

    present = set()
    for p in Path(prices_dir).glob("*.US.eod.parquet"):
        present.add(p.name.replace(".US.eod.parquet", ""))

    in_window = panel[panel["snapshot_date"] >= window_start]
    relevant_missing = sorted(set(in_window["ticker"].astype(str)) - present)

    # Rank by snapshot count (impact)
    counts = (in_window[in_window["ticker"].isin(relevant_missing)]
              .groupby("ticker").size().sort_values(ascending=False))
    return [t for t in counts.index.tolist() if is_clean_ticker(t)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*",
                    help="Specific tickers (default: in-window missing R1000 names)")
    ap.add_argument("--window-start", default="2020-06-30",
                    help="Only fetch tickers in R1000 snapshots ≥ this date (default 2020-06-30)")
    ap.add_argument("--max", type=int, default=25,
                    help="Cap to N tickers (default 25 — matches AV free-tier daily limit)")
    ap.add_argument("--sleep-ms", type=int, default=13_000,
                    help="Sleep between calls (default 13s for 5/min free-tier limit)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        print("ERROR: ALPHAVANTAGE_API_KEY is not set in the environment.",
              file=sys.stderr)
        return 1

    cfg = get_cfg()
    prices_dir = cfg["paths"]["prices_dir"]
    russell_xlsx = cfg["paths"]["russell_xlsx"]
    window_start = pd.Timestamp(args.window_start)

    if args.tickers:
        targets = [t for t in args.tickers if is_clean_ticker(t)]
    else:
        targets = discover_in_window_missing(prices_dir, russell_xlsx, window_start)

    if args.max:
        targets = targets[: args.max]

    print(f"Prices dir:  {prices_dir}")
    print(f"Targets:     {len(targets)} tickers (max={args.max})")
    if args.dry_run:
        for t in targets:
            print(f"  {t}  → AV: {to_alpha_symbol(t)}")
        return 0

    n_ok, n_skip, n_empty, n_rate, n_err = 0, 0, 0, 0, 0
    for i, t in enumerate(targets, 1):
        out_path = Path(prices_dir) / f"{t}.US.eod.parquet"
        if out_path.exists():
            n_skip += 1
            continue

        df, status = fetch_one(t, api_key)
        if status == "ok" and not df.empty:
            df.to_parquet(out_path, engine="pyarrow", index=False)
            print(f"  [{i:>3}/{len(targets)}] {t:<8}  rows={len(df):>5}  → {out_path.name}")
            n_ok += 1
        elif status.startswith("rate_limited"):
            print(f"  [{i:>3}/{len(targets)}] {t:<8}  RATE LIMIT — stopping. ({status})")
            n_rate += 1
            break
        elif status.startswith("no_data") or df.empty:
            print(f"  [{i:>3}/{len(targets)}] {t:<8}  empty  ({status})")
            n_empty += 1
        else:
            print(f"  [{i:>3}/{len(targets)}] {t:<8}  ERR  {status}")
            n_err += 1

        # Politeness sleep — respects 5/min free-tier limit
        time.sleep(args.sleep_ms / 1000.0)

    print()
    print(f"Done. ok={n_ok}  empty={n_empty}  err={n_err}  "
          f"rate_limited_stop={n_rate}  skipped(existing)={n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

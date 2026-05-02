"""
Fetch missing EOD daily price history from Yahoo Finance and save parquet
files in the format the engine expects.

Schema (matches existing `*.US.eod.parquet`):
  date              datetime64[ns]
  open / high / low / close   float64
  adjusted_close    float64        (Yahoo "Adj Close")
  volume            int64

Usage:
  python scripts/fetch_yahoo_eod.py                    # fetch all missing
  python scripts/fetch_yahoo_eod.py MSFT GOOGL META    # fetch a specific list
  python scripts/fetch_yahoo_eod.py --max 50           # cap to top-N by impact

The script never overwrites an existing parquet — it skips any ticker whose
file already exists in the prices directory.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Repo path setup so we can import project_alpha when running from anywhere
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_alpha.config import get_cfg                            # noqa: E402
from project_alpha.data import load_russell_panel                   # noqa: E402


# ---------------------------------------------------------------------------
# Yahoo ticker normalization
# ---------------------------------------------------------------------------
# EOD-Historical uses some symbols that need translation to Yahoo's format:
#   BRK.B       → BRK-B   (class-share dot becomes hyphen on Yahoo)
#   BF.B        → BF-B
# We keep the file name in the EOD-Historical format ({TICKER}.US.eod.parquet).

def to_yahoo_symbol(eod_ticker: str) -> str:
    """Translate an EOD-Historical-style ticker to Yahoo's format."""
    return eod_ticker.replace(".", "-")


def is_clean_ticker(t: str) -> bool:
    """Heuristic — exclude numeric IDs and obvious corp-action variants."""
    if not isinstance(t, str) or not t:
        return False
    if not re.match(r"^[A-Z]", t):
        return False
    if any(suffix in t for suffix in (".I", ".1", "WS", "WI")):
        return False
    if t.endswith("Q") and len(t) >= 4:  # bankruptcy suffix
        return False
    return bool(re.match(r"^[A-Z][A-Z0-9.\-]{0,5}$", t))


# ---------------------------------------------------------------------------
# Fetch one ticker
# ---------------------------------------------------------------------------

def fetch_one(yf_module, eod_ticker: str, *, start: str, end: str) -> pd.DataFrame:
    """Returns an empty DataFrame on failure (caller logs)."""
    sym = to_yahoo_symbol(eod_ticker)
    try:
        df = yf_module.download(
            sym,
            start=start, end=end,
            auto_adjust=False,           # keep raw close + Adj Close separate
            actions=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"  ERR  {eod_ticker:<8} yfinance.download raised: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return a MultiIndex column when symbol is a single str — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Yahoo column names → engine column names
    rename = {
        "Open":      "open",
        "High":      "high",
        "Low":       "low",
        "Close":     "close",
        "Adj Close": "adjusted_close",
        "Volume":    "volume",
    }
    df = df.rename(columns=rename)
    needed = ["open", "high", "low", "close", "adjusted_close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  ERR  {eod_ticker:<8} missing columns from Yahoo: {missing}")
        return pd.DataFrame()

    out = df[needed].copy()
    out.insert(0, "date", pd.to_datetime(out.index).tz_localize(None))
    out = out.reset_index(drop=True)
    out["volume"] = out["volume"].fillna(0).astype("int64")
    for c in ("open", "high", "low", "close", "adjusted_close"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = out.dropna(subset=["adjusted_close"]).sort_values("date").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_missing(prices_dir: str, russell_xlsx: str) -> List[str]:
    """Return clean R1000 tickers with no .US.eod.parquet on disk."""
    panel = load_russell_panel(russell_xlsx)
    universe = sorted(set(panel["ticker"].astype(str)))

    present = set()
    for p in Path(prices_dir).glob("*.US.eod.parquet"):
        present.add(p.name.replace(".US.eod.parquet", ""))

    # Rank by # snapshots — most-frequently-in-universe first
    snap_counts = (panel[panel["ticker"].isin(set(universe) - present)]
                   .groupby("ticker").size().sort_values(ascending=False))

    clean = [t for t in snap_counts.index.tolist() if is_clean_ticker(t)]
    return clean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*",
                    help="Specific tickers to fetch (default: all missing R1000 names)")
    ap.add_argument("--start", default="2014-01-01",
                    help="Earliest date to fetch (default 2014-01-01 — covers full backtest history)")
    ap.add_argument("--end", default=None,
                    help="Latest date (default today)")
    ap.add_argument("--max", type=int, default=None,
                    help="Cap to top-N tickers by snapshot count (impact)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be fetched without doing it")
    ap.add_argument("--sleep-ms", type=int, default=200,
                    help="Sleep between calls (politeness; default 200ms)")
    args = ap.parse_args()

    cfg = get_cfg()
    prices_dir = cfg["paths"]["prices_dir"]
    russell_xlsx = cfg["paths"]["russell_xlsx"]
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    if args.tickers:
        targets = list(args.tickers)
    else:
        targets = discover_missing(prices_dir, russell_xlsx)

    if args.max:
        targets = targets[: args.max]

    print(f"Prices dir: {prices_dir}")
    print(f"Window:     {args.start} → {end}")
    print(f"Targets:    {len(targets)} tickers")
    if args.dry_run:
        print("DRY-RUN — would fetch:")
        for t in targets:
            print(f"  {t}  (Yahoo: {to_yahoo_symbol(t)})")
        return 0

    try:
        import yfinance
    except ImportError:
        print("ERROR: yfinance not installed.  pip install yfinance", file=sys.stderr)
        return 1

    n_ok, n_skip, n_empty, n_err = 0, 0, 0, 0
    for i, t in enumerate(targets, 1):
        out_path = Path(prices_dir) / f"{t}.US.eod.parquet"
        if out_path.exists():
            n_skip += 1
            continue
        df = fetch_one(yfinance, t, start=args.start, end=end)
        if df.empty:
            print(f"  [{i:>3}/{len(targets)}] {t:<8}  empty  (delisted / not on Yahoo)")
            n_empty += 1
        else:
            df.to_parquet(out_path, engine="pyarrow", index=False)
            print(f"  [{i:>3}/{len(targets)}] {t:<8}  rows={len(df):>5}  → {out_path.name}")
            n_ok += 1
        time.sleep(args.sleep_ms / 1000.0)

    print()
    print(f"Done. ok={n_ok}  empty={n_empty}  err={n_err}  skipped(existing)={n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

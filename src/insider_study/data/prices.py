"""yfinance price adapter with a per-ticker parquet cache.

KNOWN BIAS: Yahoo drops most delisted tickers, so the price panel is survivorship
biased. Coverage is measured and reported (docs/HYPOTHESES.md section 7).

Derived columns:
- adj_open / adj_close: split- and dividend-adjusted (total-return) prices.
- raw_close: as-traded price (undoes later splits), for validation against the
  price reported on Form 4 and for market-cap computation.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf
import yfinance.shared as yf_shared
from yfinance.exceptions import (
    YFPricesMissingError,
    YFRateLimitError,
    YFTickerMissingError,
    YFTzMissingError,
)

from insider_study.logging_setup import get_logger

log = get_logger(__name__)

CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart"
_SPLIT_TOKENS = re.compile(r"[,;\s]+")
_YF_FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits")


_OTC_SUFFIX = re.compile(r"[.\-](OB|PK|OTC|QB|QX)$")
_VALID = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")


def to_yahoo(ticker: str) -> str:
    """'BRK.B' -> 'BRK-B'; 'ACME, ACMEW' -> 'ACME'; 'XYZ.OB' -> 'XYZ'; junk -> ''."""
    first = _SPLIT_TOKENS.split(ticker.strip().strip('"').upper().lstrip("$"))[0]
    first = _OTC_SUFFIX.sub("", first.strip('"'))
    out = first.replace(".", "-").replace("/", "-")
    return out if _VALID.match(out) else ""


def _safe(ticker: str) -> str:
    return re.sub(r"[^A-Z0-9\-\^=]", "_", ticker)


def parse_chart(payload: dict[str, Any]) -> pd.DataFrame:
    """Yahoo v8 chart JSON -> OHLCV frame with dividends and split ratios."""
    result = (payload.get("chart") or {}).get("result") or []
    if not result or not result[0].get("timestamp"):
        return pd.DataFrame()
    r = result[0]
    tz = r.get("meta", {}).get("exchangeTimezoneName") or "America/New_York"
    idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    q = r["indicators"]["quote"][0]
    adj = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose", q["close"])
    frame = pd.DataFrame(
        {
            "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
            "adj_close": adj, "volume": q["volume"],
        },
        index=idx,
        dtype=float,
    )
    frame["dividends"] = 0.0
    frame["splits"] = 0.0
    events = r.get("events", {})

    def _day(ts: int) -> pd.Timestamp:
        return pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(tz).tz_localize(None).normalize()

    for d in events.get("dividends", {}).values():
        day = _day(d["date"])
        if day in frame.index:
            frame.loc[day, "dividends"] += float(d["amount"])
    for sp in events.get("splits", {}).values():
        day = _day(sp["date"])
        if day in frame.index and sp.get("denominator"):
            frame.loc[day, "splits"] = float(sp["numerator"]) / float(sp["denominator"])
    frame = frame[~frame.index.duplicated(keep="last")].dropna(subset=["close"])
    frame.index.name = "date"
    return frame


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    """Add adjusted and raw prices to a single-ticker OHLCV frame."""
    df = frame.sort_index().copy()
    factor = (df["adj_close"] / df["close"]).where(df["close"] > 0)
    df["adj_open"] = df["open"] * factor
    splits = df["splits"].replace(0.0, 1.0).fillna(1.0)
    # Cumulative product of splits strictly AFTER each date undoes yfinance's
    # retroactive split adjustment of close prices.
    later = splits[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    df["raw_close"] = df["close"] * later
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


@dataclass
class YFinancePriceProvider:
    cache_dir: Path
    batch_size: int = 50
    pause_seconds: float = 2.0
    history_start: date = date(2004, 1, 1)
    history_end: date = date(2026, 10, 20)

    def _path(self, ticker: str) -> Path:
        return self.cache_dir / f"{_safe(ticker)}.parquet"

    def download(self, tickers: Sequence[str]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        todo = sorted({t for t in tickers if not self._path(t).exists()})
        log.info("yf_download_start", n=len(todo))
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i : i + self.batch_size]
            data = None
            for attempt in range(4):
                try:
                    data = yf.download(
                        batch,
                        start=self.history_start.isoformat(),
                        end=self.history_end.isoformat(),
                        auto_adjust=False,
                        actions=True,
                        group_by="ticker",
                        progress=False,
                        threads=True,
                    )
                    break
                except Exception as exc:  # network / rate limit
                    log.warning("yf_retry", error=str(exc), attempt=attempt)
                    time.sleep(5 * 2**attempt)
            errors = {k: str(v) for k, v in getattr(yf_shared, "_ERRORS", {}).items()}
            n_limited = sum("rate" in errors.get(t, "").lower() for t in batch)
            if n_limited:
                log.warning("yf_rate_limited", n=n_limited, backoff_s=60)
                time.sleep(60)
            for t in batch:
                if "rate" in errors.get(t, "").lower():
                    # Rate-limited: leave uncached so a later pass retries it.
                    continue
                frame = pd.DataFrame()
                if data is not None and not data.empty and t in data.columns.get_level_values(0):
                    sub = data[t].dropna(subset=["Close"])
                    if not sub.empty:
                        frame = sub.reindex(columns=list(_YF_FIELDS))
                        frame.columns = [
                            "open", "high", "low", "close", "adj_close", "volume",
                            "dividends", "splits",
                        ]
                        frame.index = pd.to_datetime(frame.index).tz_localize(None)
                        frame.index.name = "date"
                # An empty file marks "not available" so we don't retry forever.
                frame.to_parquet(self._path(t))
            log.info("yf_batch", done=min(i + self.batch_size, len(todo)), total=len(todo))
            time.sleep(self.pause_seconds)

    def download_robust(self, tickers: Sequence[str], pause_seconds: float = 0.6) -> dict[str, int]:
        """One request per ticker; distinguishes 'not on Yahoo' from rate limiting.

        Batch downloads can silently return empty frames when Yahoo throttles, which
        would wrongly mark live tickers as unavailable. Here only an explicit
        missing-ticker error (or an empty history) writes the 'unavailable' marker;
        rate limits back off and retry, other errors are left for a later pass.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Keep the caller's order (most-used tickers first).
        todo = list(dict.fromkeys(t for t in tickers if t and not self._path(t).exists()))
        counts = {"ok": 0, "missing": 0, "error": 0}
        backoff = 60.0
        i = 0
        while i < len(todo):
            t = todo[i]
            try:
                hist = yf.Ticker(t).history(
                    start=self.history_start.isoformat(),
                    end=self.history_end.isoformat(),
                    auto_adjust=False,
                    actions=True,
                    raise_errors=True,
                )
            except YFRateLimitError:
                log.warning("yf_rate_limited", ticker=t, backoff_s=backoff, done=i)
                time.sleep(backoff)
                backoff = min(backoff * 2, 900.0)
                continue
            except (YFPricesMissingError, YFTzMissingError, YFTickerMissingError):
                pd.DataFrame().to_parquet(self._path(t))
                counts["missing"] += 1
                i += 1
                continue
            except Exception as exc:
                log.warning("yf_error", ticker=t, error=str(exc)[:200])
                counts["error"] += 1
                i += 1
                continue
            backoff = 60.0
            frame = hist.dropna(subset=["Close"]).reindex(columns=list(_YF_FIELDS))
            frame.columns = [
                "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits",
            ]
            frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
            frame.index.name = "date"
            frame.to_parquet(self._path(t))
            counts["ok" if len(frame) else "missing"] += 1
            i += 1
            if i % 200 == 0:
                log.info("yf_progress", done=i, total=len(todo), **counts)
            time.sleep(pause_seconds)
        return counts

    def download_chart(self, tickers: Sequence[str], pause_seconds: float = 0.25) -> dict[str, int]:
        """Direct calls to Yahoo's public v8 chart endpoint (no crumb needed).

        Same data yfinance parses (split-adjusted OHLC, adjclose, dividends, splits).
        404 'No data found' -> 'unavailable' marker; 429 -> back off and retry;
        other failures are left uncached for a later pass.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        todo = list(dict.fromkeys(t for t in tickers if t and not self._path(t).exists()))
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (research; insider-study)"})
        p1 = int(pd.Timestamp(self.history_start).timestamp())
        p2 = int(pd.Timestamp(self.history_end).timestamp())
        counts = {"ok": 0, "missing": 0, "error": 0}
        backoff, i = 30.0, 0
        while i < len(todo):
            t = todo[i]
            url = f"{CHART_URL}/{t}"
            try:
                resp = session.get(
                    url,
                    params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,split"},
                    timeout=30,
                )
            except requests.RequestException as exc:
                log.warning("chart_error", ticker=t, error=str(exc)[:200])
                counts["error"] += 1
                i += 1
                continue
            if resp.status_code == 429:
                log.warning("chart_rate_limited", ticker=t, backoff_s=backoff, done=i)
                time.sleep(backoff)
                backoff = min(backoff * 2, 900.0)
                continue
            backoff = 30.0
            if resp.status_code == 404:
                pd.DataFrame().to_parquet(self._path(t))
                counts["missing"] += 1
            elif resp.status_code == 200:
                frame = parse_chart(resp.json())
                frame.to_parquet(self._path(t))
                counts["ok" if len(frame) else "missing"] += 1
            else:
                log.warning("chart_status", ticker=t, status=resp.status_code)
                counts["error"] += 1
            i += 1
            if i % 250 == 0:
                log.info("chart_progress", done=i, total=len(todo), **counts)
            time.sleep(pause_seconds)
        return counts

    def load(self, ticker: str) -> pd.DataFrame | None:
        path = self._path(ticker)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        if frame.empty:
            return None
        return derive(frame)

    def get_prices(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        parts = []
        for t in tickers:
            frame = self.load(t)
            if frame is None:
                continue
            frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
            parts.append(frame.assign(ticker=t).reset_index())
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

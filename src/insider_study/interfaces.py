"""Component contracts for the research pipeline.

EDGARIngestor -> Form4Parser -> InsiderNormalizer -> TransactionFilter ->
SignalBuilder -> PortfolioConstructor -> RiskManager -> Backtester -> Evaluator -> Reporter

Data sources (prices, factors, shares outstanding) sit behind adapter protocols so
yfinance can be swapped for CRSP/Norgate without touching downstream code.
Concrete implementations arrive in later phases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

# ---------------------------------------------------------------- data adapters


@runtime_checkable
class EDGARIngestor(Protocol):
    def fetch(self, years: Sequence[int]) -> list[Path]:
        """Download raw SEC artifacts for the given years; return local paths."""
        ...


@runtime_checkable
class PriceProvider(Protocol):
    def get_prices(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Long frame: date, ticker, open, high, low, close, adj_close, volume."""
        ...


@runtime_checkable
class FactorProvider(Protocol):
    def get_factors(self, start: date, end: date, frequency: str) -> pd.DataFrame:
        """Columns: mkt_rf, smb, hml, rmw, cma, umd, rf (decimal returns)."""
        ...

    def get_size_breakpoints(self, start: date, end: date) -> pd.DataFrame:
        """Monthly NYSE market-cap percentile breakpoints."""
        ...


@runtime_checkable
class SharesOutstandingProvider(Protocol):
    def get_shares_outstanding(self, ciks: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Point-in-time shares outstanding, keyed by the date they became public."""
        ...


# ---------------------------------------------------------------- pipeline stages


@runtime_checkable
class Form4Parser(Protocol):
    def parse(self, raw_paths: Sequence[Path]) -> pd.DataFrame:
        """Return rows matching domain.RAW_TRANSACTION_COLUMNS."""
        ...


@runtime_checkable
class InsiderNormalizer(Protocol):
    def normalize(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Add canonical insider identity and role columns."""
        ...


@dataclass(frozen=True)
class FilterResult:
    eligible: pd.DataFrame
    sales: pd.DataFrame
    plan_10b5_1: pd.DataFrame
    exclusion_counts: Mapping[str, int] = field(default_factory=dict)


@runtime_checkable
class TransactionFilter(Protocol):
    def apply(self, transactions: pd.DataFrame) -> FilterResult: ...


@runtime_checkable
class SignalBuilder(Protocol):
    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        """Return events with entry_date (strictly after filing_date) and features."""
        ...


@runtime_checkable
class PortfolioConstructor(Protocol):
    def target_weights(self, signals: pd.DataFrame, rebalance_date: date) -> pd.Series:
        """Weights indexed by ticker using only signals known before rebalance_date."""
        ...


@runtime_checkable
class RiskManager(Protocol):
    def apply(self, weights: pd.Series, market_data: pd.DataFrame, aum_usd: float) -> pd.Series:
        """Enforce position caps and liquidity (ADV participation) limits."""
        ...


@runtime_checkable
class CostModel(Protocol):
    def trade_costs(self, trades_usd: pd.Series, market_data: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True)
class BacktestResult:
    returns_gross: pd.Series
    returns_net: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


@runtime_checkable
class Backtester(Protocol):
    def run(self, signals: pd.DataFrame, start: date, end: date) -> BacktestResult: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, result: BacktestResult, factors: pd.DataFrame) -> Mapping[str, float]: ...


@runtime_checkable
class Reporter(Protocol):
    def render(self, results: Mapping[str, Mapping[str, float]], out_dir: Path) -> Path: ...

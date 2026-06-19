# Enhanced MarketData class with:
# - Batch fetching for multiple tickers
# - Rate limiting (1 second between batches)
# - Parquet caching per ticker
# - Missing data handling (forward fill, then drop)
# - Return aligned price matrix for all tickers

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import config

logger = logging.getLogger(__name__)


class MarketData:
    """Fetch and manage market price data using yfinance.

    Supports single-ticker historical fetches (original interface) plus
    batch multi-ticker downloads with Parquet caching and aligned price
    matrices suitable for cross-sectional quantitative research.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        if cache_dir is None:
            self.cache_dir = config.DATA_DIR / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-ticker helpers (backward-compatible)
    # ------------------------------------------------------------------

    def get_historical_prices(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """Fetch historical price data for a single ticker."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                logger.warning("No data found for %s", ticker)
                return pd.DataFrame()

            # Standardize columns
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            df.index.name = "date"

            # Cache to parquet
            cache_path = self.cache_dir / f"{ticker}_prices.parquet"
            df.to_parquet(cache_path)

            logger.info("Fetched %d rows for %s", len(df), ticker)
            return df

        except Exception as exc:
            logger.error("Error fetching data for %s: %s", ticker, exc)
            return pd.DataFrame()

    def get_current_price(self, ticker: str) -> dict:
        """Get current price and basic info for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "ticker": ticker,
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "volume": info.get("volume"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception as exc:
            logger.error("Error fetching current price for %s: %s", ticker, exc)
            return {}

    def get_multiple_tickers(self, tickers: list[str], period: str = "5y") -> dict:
        """Fetch historical data for multiple tickers (sequential)."""
        results: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            results[ticker] = self.get_historical_prices(ticker, period)
        return results

    def get_sp500_tickers(self) -> list[str]:
        """Get list of top S&P 500 tickers (legacy convenience method)."""
        from data.universe import UniverseManager

        return UniverseManager(cache_dir=self.cache_dir).get_sp500_tickers()

    # ------------------------------------------------------------------
    # Multi-asset batch helpers
    # ------------------------------------------------------------------

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        field: str = "Close",
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for multiple tickers.

        Returns a DataFrame with dates as the index and ticker symbols as
        columns.  Missing data is forward-filled then any column that is
        still entirely NaN is dropped.

        Parameters
        ----------
        tickers : list[str]
            Yahoo Finance ticker symbols.
        start, end : str
            Date strings understood by ``yfinance`` (e.g. ``"2020-01-01"``).
        field : str
            Price field to extract (default ``"Close"``).
        """
        cache_file = self.cache_dir / f"prices_{start}_{end}.parquet"

        # Check cache – serve immediately if all requested tickers present
        if cache_file.exists():
            try:
                cached = pd.read_parquet(cache_file)
                missing = set(tickers) - set(cached.columns)
                if not missing:
                    return cached[tickers].copy()
            except Exception as exc:
                logger.warning("Cache read failed, re-fetching: %s", exc)

        # Batch download with rate limiting
        batch_size = 10
        all_data: dict[str, pd.Series] = {}

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            try:
                data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if data.empty:
                    continue

                field_data = data[field] if field in data.columns else data

                if isinstance(field_data, pd.DataFrame):
                    for ticker in batch:
                        if ticker in field_data.columns:
                            all_data[ticker] = field_data[ticker]
                elif isinstance(field_data, pd.Series) and len(batch) == 1:
                    all_data[batch[0]] = field_data

            except Exception as exc:
                logger.error("Error fetching batch %s: %s", batch, exc)

            # Rate-limit between batches
            if i + batch_size < len(tickers):
                time.sleep(1)

        if not all_data:
            logger.warning("No price data retrieved for any ticker")
            return pd.DataFrame()

        # Create aligned DataFrame
        prices = pd.DataFrame(all_data)

        # Handle missing data
        prices = prices.ffill()  # Forward fill gaps
        prices = prices.dropna(axis=1, how="all")  # Drop tickers with no data

        # Cache to disk
        try:
            prices.to_parquet(cache_file)
        except Exception as exc:
            logger.warning("Failed to cache prices: %s", exc)

        logger.info(
            "Built price matrix %s x %s tickers (%s – %s)",
            prices.shape[0],
            prices.shape[1],
            start,
            end,
        )
        return prices

    def get_returns(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Fetch daily simple returns for multiple tickers."""
        prices = self.get_prices(tickers, start, end)
        returns = prices.pct_change().dropna(how="all")
        return returns

    def get_log_returns(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Fetch daily log returns for multiple tickers."""
        prices = self.get_prices(tickers, start, end)
        log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
        return log_ret

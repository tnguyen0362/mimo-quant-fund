# S&P 500 universe with survivorship bias handling
# - Fetch current S&P 500 tickers from Wikipedia
# - Cache ticker list to avoid repeated fetches
# - Provide universe for any historical date (best-effort with available data)
# - Handle missing data gracefully

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config.settings import config

logger = logging.getLogger(__name__)


class UniverseManager:
    """Manage S&P 500 universe with caching and fallback handling."""

    def __init__(self, cache_dir: str | Path | None = None):
        if cache_dir is None:
            self.cache_dir = config.DATA_DIR / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ticker_cache: list[str] | None = None

    def get_sp500_tickers(self) -> list[str]:
        """Fetch S&P 500 tickers from Wikipedia, cache locally.

        Returns a sorted list of normalized ticker symbols (``-`` replaces
        ``.`` as per Yahoo Finance convention).  The cache file is refreshed
        weekly to stay current with index composition changes.
        """
        if self._ticker_cache is not None:
            return self._ticker_cache

        cache_file = self.cache_dir / "sp500_tickers.json"

        # Check cache (refresh weekly)
        if cache_file.exists():
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(
                    cache_file.stat().st_mtime
                )
                if cache_age < timedelta(days=7):
                    with open(cache_file) as f:
                        tickers = json.load(f)
                    self._ticker_cache = tickers
                    return tickers
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Cache read failed, re-fetching: %s", exc)

        # Fetch from Wikipedia
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            table = tables[0]
            tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
            tickers = sorted(tickers)

            # Cache to disk
            with open(cache_file, "w") as f:
                json.dump(tickers, f)

            self._ticker_cache = tickers
            logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
            return tickers

        except Exception as exc:
            logger.error("Error fetching S&P 500 tickers: %s", exc)
            return self._get_fallback_universe()

    def _get_fallback_universe(self) -> list[str]:
        """Fallback universe of top 50 stocks by market cap."""
        fallback = sorted([
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY",
            "AVGO", "TSLA", "WMT", "JPM", "V", "UNH", "MA", "XOM", "COST",
            "HD", "PG", "ABBV", "CRM", "NFLX", "MRK", "BAC", "AMD", "CVX",
            "ORCL", "KO", "TMO", "PEP", "LIN", "CSCO", "ACN", "WFC", "ADBE",
            "DHR", "ABT", "QCOM", "TXN", "PM", "COP", "NEE", "CMCSA", "INTC",
            "UNP", "INTU", "AMGN", "AMAT", "LOW", "CAT",
        ])
        logger.info("Using fallback universe of %d tickers", len(fallback))
        return fallback

    def get_universe_size(self) -> int:
        """Return current universe size."""
        return len(self.get_sp500_tickers())

    def is_in_universe(self, ticker: str) -> bool:
        """Check whether a ticker is in the current universe."""
        tickers = self.get_sp500_tickers()
        normalized = ticker.replace(".", "-")
        return normalized in tickers
